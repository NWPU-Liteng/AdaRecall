"""Build the label-free text index for source records and visual locators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from omni_proof_runtime import authorize_history_users, checksum, file_digest, write_json
from omni_proof_sources import build_sources
from extract_omni_visual_locators import valid_regions


def completed_regions(row):
    if row.get('regions'):
        regions = valid_regions({'regions': row['regions']}, image_size=row.get('image_size'))
        return regions, 'complete_json' if regions else 'invalid_placeholder_only'
    decoder, regions = json.JSONDecoder(), []
    text = row.get('raw_locator', '')
    for position, character in enumerate(text):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except ValueError:
            continue
        if isinstance(value, dict) and set(value) >= {'box', 'text'}:
            for region in valid_regions({'regions': [value]}, image_size=row.get('image_size')):
                if region not in regions:
                    regions.append(region)
    return regions, 'completed_regions_from_partial_json' if regions else 'invalid_no_completed_region'


def token_vector(texts, tokenizer, model):
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to('cuda:0')
    if encoded.input_ids.shape[1] > 2048:
        raise ValueError('index document exceeds frozen text encoder context')
    with torch.inference_mode():
        output = model(**encoded, use_cache=False).last_hidden_state
    positions = encoded.attention_mask.shape[1] - 1 - encoded.attention_mask.flip(dims=[1]).argmax(dim=1)
    rows = torch.arange(output.shape[0], device=output.device)
    return torch.nn.functional.normalize(output[rows, positions].float(), dim=-1).cpu().numpy()


def all_locator_records(root):
    records, hashes, statuses = {}, {}, []
    if (root / 'records.jsonl').is_file() and (root / 'status.json').is_file():
        status = json.loads((root / 'status.json').read_text())
        if (not status.get('complete')
                or status['records_sha256'] != file_digest(root / 'records.jsonl')
                or status['image_hashes_sha256'] != file_digest(root / 'image_hashes.json')):
            raise ValueError('merged visual locator source not sealed or hash differs')
        hashes = json.loads((root / 'image_hashes.json').read_text())
        with (root / 'records.jsonl').open() as stream:
            records = {row['image_sha256']: row for row in map(json.loads, stream)}
        if set(records) != set(hashes.values()):
            raise ValueError('merged locator source lost images from its denominator')
        return hashes, records, [status]
    for directory in sorted(root.glob('visual_shard_*')):
        if not directory.is_dir():
            continue
        status_path = directory / 'status.json'
        if not status_path.is_file():
            raise ValueError(f'missing locator status: {directory}')
        status = json.loads(status_path.read_text())
        statuses.append(status)
        if not status.get('complete'):
            raise ValueError(f'incomplete locator shard: {directory}')
        hashes.update(json.loads((directory / 'image_hashes.json').read_text()))
        with (directory / 'records.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                old = records.get(row['image_sha256'])
                if old is not None and old != row:
                    raise ValueError('duplicate image locator identity differs')
                records[row['image_sha256']] = row
    if not statuses or set(records) != set(hashes.values()):
        raise ValueError('complete locator shards and denominator required')
    return hashes, records, statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--locator-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--version-root', type=Path, required=True)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--allowed-splits', nargs='+', default=['development'],
                        choices=['development', 'validation', 'holdout'])
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError('fresh output and positive batch size required')
    started = time.perf_counter()
    graphs = authorize_history_users(args.prepared, [args.user_id],
                                     allowed_splits=set(args.allowed_splits))
    graph = graphs[0]
    image_hashes, locators, locator_statuses = all_locator_records(args.locator_root)
    source = build_sources(graph, image_hashes)
    records = []
    for record in source['records']:
        if record['kind'] == 'source_text':
            context = '\n'.join(source['objects'][identity]['text']
                                for identity in record['retrieval_context_object_ids'])
            records.append({**record, 'retrieval_text': record['locator'] + ('\n' + context if context else ''),
                'support_status': 'original_text_source', 'claim_source': 'quoted_original_text'})
        elif record['kind'] == 'image_locator':
            locator = locators.get(source['objects'][record['image_object_id']]['source_sha256'])
            if locator is None:
                continue
            regions, parse_status = completed_regions(locator)
            if not regions:
                continue
            for region_index, region in enumerate(regions):
                records.append({**record, 'record_id': record['record_id'] + ':region:' + str(region_index),
                    'retrieval_text': region['text'] + '\n' + record['context'],
                    'locator_sha256': locator['locator_sha256'], 'source_region_index': region_index,
                    'locator_regions': [region], 'locator_parse_status': parse_status,
                    'locator_original_parse_valid': locator['parse_valid'],
                    'locator_original_eos': locator['eos'],
                    'support_status': 'locator_for_original_image',
                    'claim_source': 'raw_image_required_not_locator_fact'})
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side='left')
    model = AutoModel.from_pretrained(str(args.model), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    selected, rejected = [], []
    for record in records:
        try:
            token_count = len(tokenizer.encode(record['retrieval_text'], add_special_tokens=False))
        except Exception as error:
            rejected.append({'record_id': record['record_id'], 'reason': str(error)})
            continue
        if token_count > 2048:
            rejected.append({'record_id': record['record_id'], 'reason': 'encoder_context', 'tokens': token_count})
        else:
            selected.append({**record, 'retrieval_tokens': token_count})
    vectors = []
    for begin in range(0, len(selected), args.batch_size):
        vectors.append(token_vector([record['retrieval_text'] for record in selected[begin:begin + args.batch_size]],
                                    tokenizer, model))
    vectors = np.concatenate(vectors, axis=0) if vectors else np.empty((0, 1024), dtype=np.float32)
    if len(vectors) != len(selected) or (len(vectors) and not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4)):
        raise ValueError('invalid proof index vectors')
    costs = {identity: int(locator['reader_image_tokens']) for identity, locator in locators.items()
             if isinstance(locator.get('reader_image_tokens'), int)
             and locator['reader_image_tokens'] > 0}
    cost_source_sha256 = file_digest(args.locator_root / 'records.jsonl')
    missing_costs = sorted({source['objects'][object_id]['source_sha256'] for record in selected
        for object_id in record['object_ids'] if source['objects'][object_id]['kind'] == 'image'} - set(costs))
    if missing_costs:
        raise ValueError(f'missing actual Reader processor costs for {len(missing_costs)} images')
    args.output.mkdir(parents=True)
    np.save(args.output / 'vectors.npy', vectors, allow_pickle=False)
    write_json(args.output / 'records.json', selected)
    write_json(args.output / 'source.json', source)
    write_json(args.output / 'image_costs.json', costs)
    payload = {'complete': True, 'schema': 'proof-index-v2', 'person_id': args.user_id,
        'prepared_audit_sha256': file_digest(args.prepared / 'audit.json'),
        'locator_root': str(args.locator_root.resolve()), 'locator_statuses': locator_statuses,
        'locator_records': len(locators), 'records': len(selected), 'rejected_records': rejected,
        'vectors_sha256': file_digest(args.output / 'vectors.npy'),
        'records_sha256': file_digest(args.output / 'records.json'),
        'source_sha256': file_digest(args.output / 'source.json'),
        'image_costs_sha256': file_digest(args.output / 'image_costs.json'),
        'model_files_sha256': {path.name: file_digest(path) for path in args.model.iterdir()
            if path.suffix in {'.json', '.safetensors', '.txt'}},
        'image_processor_cost_source_sha256': cost_source_sha256,
        'image_cost_count': len(costs), 'construction_seconds': time.perf_counter() - started,
        'retrieval_contract': 'same frozen text encoder for query, original text and visual locator; raw image only at Reader',
        'boundary': 'offline no-label index; locator is not an answer fact; source containers and unique object IDs remain explicit'}
    write_json(args.output / 'index.json', payload)
    print(json.dumps({key: payload[key] for key in ('complete', 'person_id', 'records', 'rejected_records',
                                                     'construction_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
