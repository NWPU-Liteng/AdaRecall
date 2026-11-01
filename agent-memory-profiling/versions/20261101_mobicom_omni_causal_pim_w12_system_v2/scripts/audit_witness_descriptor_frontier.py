"""Audit descriptor-frontier recall against prior source-located witness branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from omni_multi_witness import branch_source_families, has_independent_branch_pair
from omni_proof_runtime import file_digest, write_json
from omni_provenance_optimizer import FrozenRelevanceModel
from omni_witness_descriptor import (rank_source_family_frontier,
                                     validate_descriptor_index)


def query_vectors(texts, tokenizer, model):
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to('cuda:0')
    if encoded.input_ids.shape[1] > 2048:
        raise ValueError('query exceeds frozen text encoder context')
    with torch.inference_mode():
        output = model(**encoded, use_cache=False).last_hidden_state
    positions = encoded.attention_mask.shape[1] - 1 - encoded.attention_mask.flip(dims=[1]).argmax(dim=1)
    rows = torch.arange(output.shape[0], device=output.device)
    return torch.nn.functional.normalize(output[rows, positions].float(), dim=-1).cpu().numpy()


def reference_branches(row):
    shadow = row.get('proof_plan', {}).get('multi_witness_shadow') or {}
    branches = {}
    for witness in shadow.get('witnesses', []):
        branches.setdefault(witness['demand_id'], []).append(witness)
    return branches


def independent_pair_families(branches):
    families = [branch_source_families(item) for item in branches]
    pairs = []
    for left in range(len(families)):
        if not families[left]:
            continue
        for right in range(left + 1, len(families)):
            if families[right] and families[left].isdisjoint(families[right]):
                pairs.append((families[left], families[right]))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--descriptor-index', type=Path, required=True)
    parser.add_argument('--reference-results', type=Path, required=True)
    parser.add_argument('--design-results', type=Path, required=True)
    parser.add_argument('--embedder', type=Path, required=True)
    parser.add_argument('--reranker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prerank-family-limit', type=int, default=32)
    parser.add_argument('--per-demand-limit', type=int, default=8)
    args = parser.parse_args()
    if (args.output.exists() or args.prerank_family_limit < args.per_demand_limit
            or args.per_demand_limit < 1):
        raise ValueError('fresh output and valid frontier limits required')
    started = time.perf_counter()
    meta = json.loads((args.index / 'index.json').read_text())
    records = json.loads((args.index / 'records.json').read_text())
    vectors = np.load(args.index / 'vectors.npy')
    descriptors = validate_descriptor_index(
        json.loads(args.descriptor_index.read_text()), records,
        person_id=meta['person_id'],
        source_index_sha256=file_digest(args.index / 'index.json'),
        records_sha256=file_digest(args.index / 'records.json'))
    records_by_id = {record['record_id']: record for record in records}
    reference = {row['question_id']: row
                 for row in json.loads(args.reference_results.read_text())['rows']}
    design_rows = json.loads(args.design_results.read_text())['rows']
    tokenizer = AutoTokenizer.from_pretrained(str(args.embedder), local_files_only=True,
                                              padding_side='left')
    embedder = AutoModel.from_pretrained(str(args.embedder), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    reranker = FrozenRelevanceModel(args.reranker, 'cuda:0')
    if len(records) != len(vectors):
        raise ValueError('proof index records/vectors mismatch')

    rows, independent_count, pair_recall, exact_touch = [], 0, 0, 0
    for row in design_rows:
        question_id = row['question_id']
        demands = row['proof_plan']['demands']
        views = [json.loads(value) for value in row['proof_plan']['retrieval']['views']]
        if len(demands) != len(views):
            raise ValueError(f'demand/view count mismatch for {question_id}')
        query_vectors_value = query_vectors(
            [json.dumps(value, ensure_ascii=False, sort_keys=True) for value in views],
            tokenizer, embedder)
        reference_by_demand = reference_branches(reference[question_id])
        per_demand = []
        for index, demand in enumerate(demands):
            scores = (vectors @ query_vectors_value[index]).tolist()
            frontier = rank_source_family_frontier(
                records, scores, descriptors, limit=args.prerank_family_limit,
                required_kind=demand.get('evidence_scope', {}).get('kind'))
            branches = frontier['branches']
            pre_rerank_family_count = len(branches)
            if branches:
                values = reranker.score(
                    json.dumps(views[index], ensure_ascii=False, sort_keys=True),
                    [records_by_id[item['record_id']]['retrieval_text'] for item in branches])
                for branch, value in zip(branches, values):
                    branch['cosine_score'] = branch['score']
                    branch['reranker_score'] = float(value)
                    branch['score'] = float(value)
                branches.sort(key=lambda item: (
                    -item['reranker_score'], item['source_family'], item['record_id']))
            branches = branches[:args.per_demand_limit]
            retained_families = {item['source_family'] for item in branches}
            retained_records = {item['record_id'] for item in branches}
            reference_items = reference_by_demand.get(demand['demand_id'], [])
            reference_independent = has_independent_branch_pair(reference_items)
            pairs = independent_pair_families(reference_items) if reference_independent else []
            recalled_pair = any(
                bool(left & retained_families) and bool(right & retained_families)
                for left, right in pairs)
            touched = any(
                set(item.get('retrieval_record_ids') or item.get('record_ids') or [])
                & retained_records for item in reference_items)
            independent_count += reference_independent
            pair_recall += recalled_pair
            exact_touch += touched
            per_demand.append({
                'demand_id': demand['demand_id'],
                'pre_rerank_family_count': pre_rerank_family_count,
                'retained_family_count': len(branches),
                'retained_families': sorted(retained_families),
                'retained_records': sorted(retained_records),
                'branch_seconds': None,
                'reference_independent': reference_independent,
                'reference_pair_family_recalled': recalled_pair,
                'reference_exact_record_touched': touched,
            })
        rows.append({'question_id': question_id, 'per_demand': per_demand})
    result = {
        'schema': 'witness-descriptor-frontier-audit-v1',
        'source_index_sha256': file_digest(args.index / 'index.json'),
        'descriptor_index_sha256': file_digest(args.descriptor_index),
        'reference_results_sha256': file_digest(args.reference_results),
        'design_results_sha256': file_digest(args.design_results),
        'prerank_family_limit': args.prerank_family_limit,
        'per_demand_limit': args.per_demand_limit,
        'reference_independent_demand_count': independent_count,
        'reference_pair_family_recall_count': pair_recall,
        'reference_pair_family_recall': (pair_recall / independent_count) if independent_count else None,
        'reference_exact_record_touch_count': exact_touch,
        'seconds_including_model_startup': time.perf_counter() - started,
        'rows': rows,
        'boundary': 'candidate-frontier recall only; no semantic sufficiency or quality claim',
    }
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        'reference_independent_demand_count', 'reference_pair_family_recall_count',
        'reference_pair_family_recall', 'reference_exact_record_touch_count',
        'seconds_including_model_startup')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
