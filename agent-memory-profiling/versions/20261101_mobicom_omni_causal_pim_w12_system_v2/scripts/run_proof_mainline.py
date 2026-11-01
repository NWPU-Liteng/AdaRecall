"""One frozen proof-connected Reader run for an authorized development user."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from omni_proof_planner import (PARSER_PROMPT, QUERY_GRAPH_SCHEMA, parse_query_graph,
                                select_retrieval_package)
from omni_proof_join import solve_witness_join
from omni_multi_witness import (INDEPENDENT_WITNESS_PROMPT, MULTI_WITNESS_PROMPT,
                                audit_multi_witness, branch_source_families,
                                has_independent_branch_pair, independent_of_existing,
                                limit_witness_alternatives,
                                source_family)
from omni_witness_descriptor import (audit_descriptor_frontiers,
                                     rank_source_family_frontier,
                                     validate_descriptor_index)
from omni_proof_lifecycle import compile_lifecycle
from omni_proof_backend import choose_backend
from omni_proof_grounding import (WITNESS_PROMPT, compact_record, expand_compact_sources,
                                   validate_witnesses, WITNESS_SCHEMA)
from omni_proof_runtime import (authorize_development_users, authorize_users, file_digest, greedy, load_reader,
                                prepare_inputs, structured_call, write_json)
from residual_profile import ReaderOperatorTrace
from omni_proof_query_contract import (QUERY_CONTRACT_PROMPT, QUERY_CONTRACT_SCHEMA,
                                       parse_query_contract, reader_context,
                                       render_operation_plan, retrieval_views)
from audit_proof_direction import selected_jsonl
from omni_proof_sources import image_source_header, materialize
from omni_provenance_optimizer import FrozenRelevanceModel
from pim_backend.evidence_frontier import (build_frontier_levels, candidates_for_frontier_level,
                                           compile_candidate_frontier)
from pim_backend.evidence_state import plan_evidence_route
from pim_backend.proof_query import (audit_host_grounding_coverage, compile_candidate_table,
                                     compile_proof_query, execute_proof_query)
from pim_backend.evidence_prefilter import (audit_evidence_prefilter,
                                            enforce_host_witness_coverage,
                                            plan_evidence_state_prefilter)
from pim_backend.functional_decode import materialize_w12a16_primary
from pim_backend.lm_head_functional import LmHeadBankDispatcher
from pim_backend.w12_functional_dispatch import W12FunctionalDispatch


READER_SYSTEM = ('你是端侧长期多模态记忆助手。只依据本次输入中给出的原始对话和原图回答。'
                 '原图定位记录不是答案事实，必须以原始对话或原图为准；区分用户陈述与助手建议。'
                 '证据不足时明确说明。回答简洁，不展示推理过程。')
PARSER_SYSTEM = PARSER_PROMPT


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


W12_STAGES = ('query_parser', 'source_grounding', 'reader_generation')
W12_COUNTER_FIELDS = (
    'bank_calls',
    'decode_fallback_calls',
    'prefill_fallback_calls',
    'fallback_rows',
    'bank_external_bytes',
    'fallback_external_bytes',
)


def resolve_reader_quantization(functional_route, requested, lm_head_mode):
    """Resolve the Host Reader contract without conflating it with bank execution."""
    if requested is None:
        resolved = 'w12a16' if functional_route != 'none' else 'bf16'
    else:
        resolved = requested
    if resolved not in ('bf16', 'w12a16'):
        raise ValueError('Reader quantization must be bf16 or w12a16')
    if functional_route != 'none' and resolved != 'w12a16':
        raise ValueError('W12 functional routes require the W12A16 contract')
    if resolved == 'w12a16' and lm_head_mode != 'off':
        raise ValueError('Host W12A16 and lm_head dispatch target the same modules')
    return resolved


def aggregate_w12_stage_dispatch(rows):
    ledgers = [row['w12_stage_dispatch'] for row in rows
               if row.get('w12_stage_dispatch') is not None]
    if not ledgers:
        return None
    identities = {
        ledger[stage]['layout_instance_sha256']
        for ledger in ledgers for stage in W12_STAGES
    }
    if len(identities) != 1:
        raise RuntimeError('W12 layout identity changed across requests or stages')
    stages = {}
    for stage in W12_STAGES:
        stage_rows = [ledger[stage] for ledger in ledgers]
        item = {
            field: sum(row[field] for row in stage_rows)
            for field in W12_COUNTER_FIELDS
        }
        item.update({
            'requests_with_bank_calls': sum(row['bank_calls'] > 0
                                            for row in stage_rows),
            'requests_with_decode_fallback': sum(
                row['decode_fallback_calls'] > 0 for row in stage_rows),
            'selected_modules': ledgers[0][stage]['selected_modules'],
            'unique_layouts': ledgers[0][stage]['unique_layouts'],
        })
        stages[stage] = item
    return {
        'requests': len(ledgers),
        'stages': stages,
        'all_requests_all_stages_hit_bank': all(
            ledger[stage]['bank_calls'] > 0
            for ledger in ledgers for stage in W12_STAGES),
        'decode_fallback_zero': all(
            ledger[stage]['decode_fallback_calls'] == 0
            for ledger in ledgers for stage in W12_STAGES),
        'single_dispatcher': True,
        'single_layout': True,
        'layout_instance_sha256': next(iter(identities)),
        'layout_identity_unchanged': True,
        'physical_pim_executed': False,
        'performance_comparable': False,
        'boundary': ('stage deltas from one functional W12 dispatcher; Torch '
                     'wall-clock is not a PIM latency measurement'),
    }


def structured_call_with_profile(model, processor, system, payload, *, enabled,
                                 lm_head_dispatch=None, **kwargs):
    trace = ReaderOperatorTrace(model) if enabled else None
    try:
        dispatch_scope = (lm_head_dispatch.decode() if lm_head_dispatch is not None
                          else nullcontext())
        with dispatch_scope:
            result = structured_call(model, processor, system, payload,
                                     trace=trace, profile_operator_kernels=enabled, **kwargs)
    finally:
        if trace is not None:
            trace.close()
    if trace is not None:
        result['structured_linear_trace'] = trace.as_dict()
    return result


def load_image(archive, ref):
    member = ref.get('resolved', ref.get('normalized'))
    with zipfile.ZipFile(archive, metadata_encoding='gb18030') as source:
        payload = source.read(member)
    return Image.open(io.BytesIO(payload)).convert('RGB'), digest_bytes(payload)


def query_vectors(texts, tokenizer, model):
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to('cuda:0')
    if encoded.input_ids.shape[1] > 2048:
        raise ValueError('query exceeds frozen text encoder context')
    with torch.inference_mode():
        output = model(**encoded, use_cache=False).last_hidden_state
    positions = encoded.attention_mask.shape[1] - 1 - encoded.attention_mask.flip(dims=[1]).argmax(dim=1)
    rows = torch.arange(output.shape[0], device=output.device)
    return torch.nn.functional.normalize(output[rows, positions].float(), dim=-1).cpu().numpy()


def load_persistent_replay(path, person_id):
    queries = {}
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get('person_id')) != str(person_id):
                continue
            question_id = str(row['question_id'])
            if question_id in queries:
                raise ValueError('persistent replay has duplicate question: ' + question_id)
            if row.get('runtime_labels_read') is not False:
                raise ValueError('persistent replay is not label-separated: ' + question_id)
            queries[question_id] = row
    if not queries:
        raise ValueError(f'persistent replay has no queries for user {person_id}')
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--embedder', type=Path, required=True)
    parser.add_argument('--reranker', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--evidence-state-index', type=Path)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--input-budget', type=int, default=8192)
    parser.add_argument('--total-text-tokens', type=int, default=3600)
    parser.add_argument('--max-images', type=int, default=4)
    parser.add_argument('--planner-time-limit', type=float, default=2.0)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--question-ids-file', type=Path)
    parser.add_argument('--persistent-replay', type=Path,
                        help='label-separated replay queries for dynamic visible-index serving')
    parser.add_argument('--visible-index-mode', choices=['filter', 'mask'], default='filter',
                        help='filter copies visible vectors; mask scans the full index with future rows masked')
    parser.add_argument('--persistent-replay-order', action='store_true',
                        help='serve selected replay queries in checkpoint order')
    parser.add_argument('--capture-reader-inputs', type=Path,
                        help='save final label-free Reader tensors and skip answer generation')
    parser.add_argument('--maintenance-progress', type=Path,
                        help='atomic progress file consumed by a background memory-maintenance worker')
    parser.add_argument('--query-contract', action='store_true')
    parser.add_argument('--candidate-frontier-seed-per-demand', type=int, default=0)
    parser.add_argument('--proof-query-shadow', action='store_true')
    parser.add_argument('--evidence-state-prefilter-shadow', action='store_true')
    parser.add_argument('--multi-witness-shadow', action='store_true')
    parser.add_argument('--witness-descriptor-index', type=Path)
    parser.add_argument('--witness-descriptor-shadow', action='store_true')
    parser.add_argument('--witness-descriptor-per-demand-limit', type=int, default=8)
    parser.add_argument('--witness-descriptor-prerank-family-limit', type=int, default=32)
    parser.add_argument('--independent-witness-rounds', type=int, default=2)
    parser.add_argument('--independent-witness-candidate-limit', type=int, default=24)
    parser.add_argument('--initial-witness-per-demand-limit', type=int, default=8)
    parser.add_argument('--pre-reader-operator-profile', action='store_true')
    parser.add_argument('--lm-head-functional-mode', choices=['off', 'shadow', 'replace'],
                        default='off')
    parser.add_argument('--w12-functional-route', choices=['none', 'shadow', 'replace'],
                        default='none')
    parser.add_argument('--reader-quantization', choices=['bf16', 'w12a16'],
                        default=None,
                        help=('Host Reader contract when the functional route is none; '
                              'W12 functional routes resolve to w12a16'))
    parser.add_argument('--w12-numeric-mode', choices=['fp32_bank', 'bf16_reference'],
                        default='bf16_reference')
    parser.add_argument('--w12-bank-count', type=int, default=256)
    parser.add_argument('--w12-tile-rows', type=int, default=16)
    parser.add_argument('--w12-host-contract-rows', type=int, default=None,
                        help=('Host W12A16 row tile; defaults to bank_count x '
                              'tile_rows so the control shares the bank contract'))
    parser.add_argument('--w12a16-group-size', type=int,
                        choices=(8, 16, 32, 64, 128, 256), default=128)
    parser.add_argument('--w12-release-source', action='store_true')
    parser.add_argument('--w12-allow-decode-fallback', action='store_true')
    parser.add_argument('--allowed-splits', nargs='+', default=['development'],
                        choices=['development', 'validation', 'holdout'])
    args = parser.parse_args()
    args.reader_quantization = resolve_reader_quantization(
        args.w12_functional_route, args.reader_quantization,
        args.lm_head_functional_mode)
    if args.w12_host_contract_rows is None:
        args.w12_host_contract_rows = args.w12_bank_count * args.w12_tile_rows
    if args.w12_host_contract_rows <= 0:
        raise ValueError('positive W12 Host contract row tile required')
    if (args.w12_functional_route != 'none'
            and args.w12_host_contract_rows
            != args.w12_bank_count * args.w12_tile_rows):
        raise ValueError('bank route requires the bank-tile-aligned Host contract')
    partial_path = args.output.with_suffix('.partial.json')
    if (args.output.exists() or args.max_new_tokens < 1
            or args.candidate_frontier_seed_per_demand < 0
            or args.witness_descriptor_per_demand_limit < 1
            or args.witness_descriptor_prerank_family_limit < args.witness_descriptor_per_demand_limit
            or not 0 <= args.shard_index < args.shard_count):
        raise ValueError('fresh output and positive generation limit required')
    if args.witness_descriptor_shadow and not args.witness_descriptor_index:
        raise ValueError('witness descriptor shadow requires a descriptor index')
    if (args.w12_functional_route != 'none'
            and (args.w12_bank_count <= 0 or args.w12_tile_rows <= 0)):
        raise ValueError('positive W12 bank count and row tile required')
    if (args.w12_functional_route != 'none'
            and args.lm_head_functional_mode != 'off'):
        raise ValueError('W12 dispatch and lm_head dispatch target the same modules')
    if (args.w12_functional_route == 'none'
            and (args.w12_release_source or args.w12_allow_decode_fallback)):
        raise ValueError('W12 bank options require an enabled functional route')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.allowed_splits == ['development']:
        authorize_development_users(args.prepared, [args.user_id])
    else:
        authorize_users(args.prepared, [args.user_id], allowed_splits=set(args.allowed_splits))
    index_meta = json.loads((args.index / 'index.json').read_text())
    if not index_meta.get('complete') or index_meta['person_id'] != args.user_id:
        raise ValueError('complete proof index for this user required')
    if index_meta.get('prepared_audit_sha256') != file_digest(args.prepared / 'audit.json'):
        raise ValueError('proof index refers to another prepared source release')
    if index_meta.get('schema') != 'proof-index-v2':
        raise ValueError('atom/region proof index v2 required; do not reuse session-level vectors')
    for name, field in [('records.json', 'records_sha256'), ('vectors.npy', 'vectors_sha256'),
                        ('source.json', 'source_sha256'), ('image_costs.json', 'image_costs_sha256')]:
        if file_digest(args.index / name) != index_meta[field]:
            raise ValueError('proof index payload identity changed: ' + name)
    records = json.loads((args.index / 'records.json').read_text())
    vectors = np.load(args.index / 'vectors.npy')
    base_records = records
    base_vectors = vectors
    source = json.loads((args.index / 'source.json').read_text())
    objects = source['objects']
    image_costs = {key: int(value) for key, value in json.loads((args.index / 'image_costs.json').read_text()).items()}
    if len(records) != len(vectors):
        raise ValueError('proof index records/vectors mismatch')
    descriptor_by_record = {}
    descriptor_index_sha256 = None
    if args.witness_descriptor_index:
        descriptor_index = json.loads(args.witness_descriptor_index.read_text())
        descriptor_by_record = validate_descriptor_index(
            descriptor_index, records, person_id=args.user_id,
            source_index_sha256=file_digest(args.index / 'index.json'),
            records_sha256=file_digest(args.index / 'records.json'))
        descriptor_index_sha256 = file_digest(args.witness_descriptor_index)
    replay_queries = (load_persistent_replay(args.persistent_replay, args.user_id)
                      if args.persistent_replay else None)
    evidence_state_by_record = {}
    if args.evidence_state_index:
        evidence_state = json.loads(args.evidence_state_index.read_text())
        if (evidence_state.get('schema') != 'evidence-state-index-v1'
                or evidence_state.get('person_id') != args.user_id
                or evidence_state.get('source_index_sha256') != file_digest(args.index / 'index.json')
                or evidence_state.get('records_sha256') != file_digest(args.index / 'records.json')
                or evidence_state.get('label_free') is not True
                or evidence_state.get('semantic_claims') is not False):
            raise ValueError('evidence-state sidecar does not match the proof index')
        for item in evidence_state.get('records', []):
            identity = item.get('record_id')
            if not isinstance(identity, str) or identity in evidence_state_by_record:
                raise ValueError('evidence-state sidecar has duplicate/invalid record identity')
            evidence_state_by_record[identity] = item
        if set(evidence_state_by_record) != {record['record_id'] for record in records}:
            raise ValueError('evidence-state sidecar record set does not match proof index')
        if evidence_state.get('record_count') != len(records):
            raise ValueError('evidence-state sidecar record count does not match proof index')
        exact_fields = ('kind', 'session_id', 'container_id', 'primary_object_id',
                        'object_ids', 'source_atom_ids', 'source_span')
        for record in records:
            sidecar_record = evidence_state_by_record[record['record_id']]
            if any(sidecar_record.get(field) != record.get(field) for field in exact_fields):
                raise ValueError('evidence-state sidecar metadata differs from proof index')
    cohort = json.loads(args.cohort.read_text())
    full_question_ids = cohort['question_ids_by_user'][args.user_id]
    if args.question_ids_file:
        requested_ids = json.loads(args.question_ids_file.read_text())
        if (not isinstance(requested_ids, list) or not requested_ids
                or len(requested_ids) != len(set(requested_ids))
                or not set(requested_ids) <= set(full_question_ids)):
            raise ValueError('question subset must be unique and belong to the registered user cohort')
        full_question_ids = [identity for identity in full_question_ids if identity in set(requested_ids)]
        if args.pre_reader_operator_profile and len(requested_ids) > 16:
            raise ValueError('pre-reader operator profiling is limited to 16 explicit questions')
        if args.lm_head_functional_mode != 'off' and len(requested_ids) > 4:
            raise ValueError('lm_head functional qualification is limited to 4 explicit questions')
    elif args.pre_reader_operator_profile:
        raise ValueError('pre-reader operator profiling requires an explicit --question-ids-file subset')
    if args.lm_head_functional_mode != 'off' and not args.question_ids_file:
        raise ValueError('lm_head functional qualification requires an explicit --question-ids-file subset')
    if replay_queries is not None:
        missing_replay = sorted(set(full_question_ids) - set(replay_queries))
        if missing_replay:
            raise ValueError('selected questions are absent from persistent replay: ' + ','.join(missing_replay[:5]))
    question_ids = [identity for index, identity in enumerate(full_question_ids)
                    if index % args.shard_count == args.shard_index]
    if args.persistent_replay_order:
        if replay_queries is None:
            raise ValueError('persistent replay order requires --persistent-replay')
        question_ids.sort(key=lambda identity: (replay_queries[identity]['checkpoint_time'], identity))
    manifests = selected_jsonl(args.prepared / 'manifests.jsonl', question_ids)
    if any(row['split'] not in args.allowed_splits or row['person_id'] != args.user_id
           for row in manifests.values()):
        raise ValueError('question manifest contains a user or split outside the authorized serving set')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if file_digest(args.prepared / 'manifests.jsonl') != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('serving query manifest identity changed')
    rows = []
    completed_ids = set()
    if partial_path.exists():
        checkpoint = json.loads(partial_path.read_text())
        if (checkpoint.get('complete') or checkpoint.get('person_id') != args.user_id
                or checkpoint.get('question_ids') != question_ids
                or checkpoint.get('shard_index') != args.shard_index
                or checkpoint.get('shard_count') != args.shard_count
                or checkpoint.get('pre_reader_operator_profile', False)
                != args.pre_reader_operator_profile
                or checkpoint.get('reader_quantization', 'bf16')
                != args.reader_quantization
                or checkpoint.get('w12_functional_route', 'none')
                != args.w12_functional_route
                or checkpoint.get('w12_numeric_mode', 'bf16_reference')
                != args.w12_numeric_mode
                or checkpoint.get('w12_bank_count', 256)
                != args.w12_bank_count
                or checkpoint.get('w12_tile_rows', 16)
                != args.w12_tile_rows
                or checkpoint.get('w12_host_contract_rows', 4096)
                != args.w12_host_contract_rows
                or checkpoint.get('w12a16_group_size', 128)
                != args.w12a16_group_size
                or checkpoint.get('w12_release_source', False)
                != args.w12_release_source
                or checkpoint.get('cohort_sha256') != file_digest(args.cohort)):
            raise ValueError('partial checkpoint does not match the frozen request stream')
        rows = checkpoint.get('rows', [])
        completed_ids = {row.get('question_id') for row in rows}
        if (len(rows) != len(completed_ids) or not completed_ids <= set(question_ids)
                or any(row.get('person_id') != args.user_id for row in rows)):
            raise ValueError('partial checkpoint contains duplicate or unauthorized rows')
    processor, reader = load_reader(args.reader)
    w12_dispatch = None
    w12_materialization = None
    reader_quantization_materialization = None
    if args.w12_functional_route != 'none':
        with torch.inference_mode():
            w12_dispatch = W12FunctionalDispatch(
                reader, mode=args.w12_functional_route,
                bank_count=args.w12_bank_count, tile_rows=args.w12_tile_rows,
                group_size=args.w12a16_group_size,
                numeric_mode=args.w12_numeric_mode,
                release_source=args.w12_release_source,
                fail_on_decode_fallback=not args.w12_allow_decode_fallback,
            ).__enter__()
            w12_materialization = w12_dispatch.materialization_summary()
            reader_quantization_materialization = w12_materialization
    elif args.reader_quantization == 'w12a16':
        with torch.inference_mode():
            reader_quantization_materialization = materialize_w12a16_primary(
                reader, group_size=args.w12a16_group_size,
                row_chunk=args.w12_host_contract_rows)
    tokenizer = AutoTokenizer.from_pretrained(str(args.embedder), local_files_only=True, padding_side='left')
    embedder = AutoModel.from_pretrained(str(args.embedder), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    reranker = FrozenRelevanceModel(args.reranker, 'cuda:0')
    lm_head_dispatch = (LmHeadBankDispatcher(reader, mode=args.lm_head_functional_mode)
                        if args.lm_head_functional_mode != 'off' else None)
    if lm_head_dispatch is not None:
        lm_head_dispatch.__enter__()
    render_cache = {}
    startup_seconds = time.perf_counter() - started

    def w12_snapshot():
        return (w12_dispatch.counter_snapshot()
                if w12_dispatch is not None else None)

    def w12_stage_delta(before, after):
        return (W12FunctionalDispatch.counter_delta(before, after)
                if w12_dispatch is not None else None)

    for position, question_id in enumerate(question_ids):
        if question_id in completed_ids:
            continue
        render_cache.clear()
        image_cache = {}
        manifest = manifests[question_id]
        query = manifest['question']
        request_started = time.perf_counter()
        w12_request_started = w12_snapshot()
        visible_index_seconds = 0.0
        visible_index_mode = 'static_full_history'
        visible_session_ids = None
        visible_record_count = len(base_records)
        visible_vector_bytes = int(base_vectors.nbytes)
        if replay_queries is not None:
            replay = replay_queries[question_id]
            if replay['query_text'] != query:
                raise ValueError(f'replay/manifest query mismatch for {question_id}')
            visible_session_ids = set(map(str, replay['visible_session_ids']))
            filter_started = time.perf_counter()
            if args.visible_index_mode == 'filter':
                keep = np.asarray([
                    str(record.get('session_id')) in visible_session_ids
                    for record in base_records
                ], dtype=bool)
                records = [record for record, selected in zip(base_records, keep) if selected]
                vectors = base_vectors[keep]
                visible_index_mode = 'dynamic_visible_filter'
            else:
                records = base_records
                vectors = base_vectors
                visible_index_mode = 'dynamic_visible_mask'
            visible_index_seconds = time.perf_counter() - filter_started
            visible_record_count = int(sum(
                str(record.get('session_id')) in visible_session_ids for record in base_records
            ))
            visible_vector_bytes = int(visible_record_count * base_vectors.shape[1] * base_vectors.dtype.itemsize)
            if args.maintenance_progress:
                write_json(args.maintenance_progress, {
                    'person_id': args.user_id,
                    'question_id': question_id,
                    'request_index': position,
                    'checkpoint_time': replay['checkpoint_time'],
                    'visible_session_ids': sorted(visible_session_ids),
                    'visible_record_count': visible_record_count,
                    'runtime_labels_read': False,
                })
        parser_prompt = QUERY_CONTRACT_PROMPT if args.query_contract else PARSER_SYSTEM
        parser_schema = QUERY_CONTRACT_SCHEMA if args.query_contract else QUERY_GRAPH_SCHEMA
        parse_result = structured_call_with_profile(reader, processor, parser_prompt,
            {'question': query, 'output_contract': 'JSON only'}, max_tokens=1024,
            budget=args.input_budget, json_schema=parser_schema,
            enabled=args.pre_reader_operator_profile, lm_head_dispatch=lm_head_dispatch)
        w12_after_parser = w12_snapshot()
        query_vector_started = time.perf_counter()
        if args.query_contract:
            query_contract = parse_query_contract(parse_result.get('parsed'), query)
            demands = query_contract['source_facts']
            relations = query_contract['relations']
            parse_status = query_contract['status']
            operation_plan = render_operation_plan(query_contract)
        else:
            query_contract = None
            demands, relations, parse_status = parse_query_graph(parse_result.get('parsed'), query)
            operation_plan = ''
        demand_texts = (retrieval_views(query, query_contract) if args.query_contract
                        else [demand['text'] for demand in demands])
        retrieval_by_demand = dict(zip((demand['demand_id'] for demand in demands), demand_texts))
        vectors_for_query = query_vectors(demand_texts, tokenizer, embedder)
        query_vector_seconds = time.perf_counter() - query_vector_started
        candidate_pool_records = []
        dense_search_started = time.perf_counter()
        candidate_width = 48
        for demand_index, demand_vector in enumerate(vectors_for_query):
            demand = demands[demand_index]
            scores = vectors @ demand_vector
            if replay_queries is not None and args.visible_index_mode == 'mask':
                scores = scores.copy()
                scores[np.asarray([
                    str(record.get('session_id')) not in visible_session_ids
                    for record in records
                ], dtype=bool)] = -np.inf
            finite = np.isfinite(scores)
            required_kind = demand.get('evidence_scope', {}).get('kind')
            if required_kind in {'source_text', 'image_locator'}:
                modality = np.asarray([record.get('kind') == required_kind
                                       for record in records], dtype=bool)
                primary = np.flatnonzero(finite & modality)
                primary = primary[np.argsort(-scores[primary])][:min(32, candidate_width)]
                remaining = np.flatnonzero(finite)
                remaining = remaining[np.argsort(-scores[remaining])]
                order = list(primary)
                seen = set(int(value) for value in primary)
                for value in remaining:
                    if int(value) not in seen:
                        order.append(int(value))
                        seen.add(int(value))
                    if len(order) >= candidate_width:
                        break
                order = np.asarray(order[:candidate_width], dtype=np.int64)
            else:
                order = np.argsort(-scores)
                order = order[np.isfinite(scores[order])][:candidate_width]
            for retrieval_rank, rank in enumerate(order):
                base = records[int(rank)]
                candidate_pool_records.append({**base, 'witness_id': f"{base['record_id']}#d{demand_index}",
                    'demand_id': demand['demand_id'], 'base_record_id': base['record_id'],
                    'query_rank': int(rank), 'retrieval_rank': retrieval_rank,
                    'score': float(scores[int(rank)]),
                    'bindings': base.get('bindings', {}),
                    'source_relations': base.get('source_relations', []),
                    'evidence_metadata': evidence_state_by_record.get(base['record_id'])})
        dense_search_seconds = time.perf_counter() - dense_search_started
        reranker_started = time.perf_counter()
        for demand in demands:
            candidates = [record for record in candidate_pool_records
                          if record['demand_id'] == demand['demand_id']]
            values = reranker.score(retrieval_by_demand[demand['demand_id']],
                                    [record['retrieval_text'] for record in candidates])
            for record, score in zip(candidates, values):
                record['retrieval_cosine'] = record['score']
                record['score'] = float(score)
        reranker_seconds = time.perf_counter() - reranker_started
        if args.evidence_state_prefilter_shadow:
            witness_records = [record for record in candidate_pool_records
                               if record['retrieval_rank'] < 48]
        else:
            witness_records = candidate_pool_records
        frontier_started = time.perf_counter()
        grounding_frontier = compile_candidate_frontier(witness_records, question_id=question_id)
        grounding_route = plan_evidence_route(grounding_frontier)
        frontier_levels = build_frontier_levels(
            grounding_frontier,
            seed_per_demand=args.candidate_frontier_seed_per_demand)
        frontier_compile_seconds = time.perf_counter() - frontier_started
        parse_vector_seconds = time.perf_counter() - request_started
        grounding_started = time.perf_counter()
        grounding_attempts = []
        grounding_budget_prepare_seconds = 0.0
        grounding_reader_seconds = 0.0
        grounding_validation_seconds = 0.0
        proof_query = None
        if args.proof_query_shadow or args.evidence_state_prefilter_shadow:
            proof_query = compile_proof_query(
                demands=demands,
                relations=relations,
                operations=(query_contract or {}).get('operations', []) if query_contract else [],
                answer_mode=(query_contract or {}).get('answer_mode', 'structured')
                if query_contract else 'structured')
        proof_query_shadow = None
        if args.proof_query_shadow:
            proof_query_started = time.perf_counter()
            candidate_table = compile_candidate_table(witness_records)
            proof_query_result = execute_proof_query(proof_query, candidate_table)
            proof_query_shadow = {
                'plan': proof_query,
                'candidate_table': {
                    'row_count': len(candidate_table['rows']),
                    'row_width_bytes': candidate_table['row_width_bytes'],
                    'descriptor_bytes': candidate_table['descriptor_bytes'],
                    'sidecar_identity_count': candidate_table['sidecar_identity_count'],
                },
                'result': proof_query_result,
                'seconds': time.perf_counter() - proof_query_started,
                'execution_backend': 'host_reference',
                'physical_pim_execution': False,
                'safe_to_replace_host_grounding': False,
            }
        evidence_state_prefilter_shadow = None
        if args.evidence_state_prefilter_shadow:
            prefilter_started = time.perf_counter()
            evidence_state_prefilter_shadow = plan_evidence_state_prefilter(
                demands=demands,
                relations=relations,
                operations=(query_contract or {}).get('operations', []) if query_contract else [],
                candidates=candidate_pool_records,
                answer_mode=(query_contract or {}).get('answer_mode', 'structured')
                if query_contract else 'structured')
            evidence_state_prefilter_shadow.pop('candidates', None)
            evidence_state_prefilter_shadow['seconds'] = time.perf_counter() - prefilter_started
            evidence_state_prefilter_shadow['answer_path_unchanged'] = True
            evidence_state_prefilter_shadow['grounding_input_replaced'] = False
        score_by_demand_record = {(record['demand_id'], record['base_record_id']): record['score']
                                  for record in witness_records}
        witness_descriptor_shadow = None
        if (args.witness_descriptor_shadow
                and not (args.query_contract and query_contract['answer_mode'] == 'contextual')):
            descriptor_started = time.perf_counter()
            records_by_id = {record['record_id']: record for record in records}
            eligible_record_ids = None
            if replay_queries is not None and args.visible_index_mode == 'mask':
                eligible_record_ids = {
                    record['record_id'] for record in records
                    if str(record.get('session_id')) in visible_session_ids}
            frontiers = {}
            for demand_index, demand in enumerate(demands):
                scores = (vectors @ vectors_for_query[demand_index]).tolist()
                frontier = rank_source_family_frontier(
                    records, scores, descriptor_by_record,
                    limit=args.witness_descriptor_prerank_family_limit,
                    required_kind=demand.get('evidence_scope', {}).get('kind'),
                    eligible_record_ids=eligible_record_ids)
                branches = frontier['branches']
                frontier['pre_rerank_family_count'] = len(branches)
                if branches:
                    rerank_scores = reranker.score(
                        retrieval_by_demand[demand['demand_id']],
                        [records_by_id[item['record_id']]['retrieval_text']
                         for item in branches])
                    for branch, score in zip(branches, rerank_scores):
                        branch['cosine_score'] = branch['score']
                        branch['reranker_score'] = float(score)
                        branch['score'] = float(score)
                    branches.sort(key=lambda item: (
                        -item['reranker_score'], item['source_family'], item['record_id']))
                branches = branches[:args.witness_descriptor_per_demand_limit]
                frontier['branches'] = branches
                frontier['source_family_count'] = len(branches)
                frontier['candidate_or_available'] = len(branches) >= 2
                frontiers[demand['demand_id']] = frontier
            descriptor_audit = audit_descriptor_frontiers(frontiers)
            witness_descriptor_shadow = {
                'schema': 'witness-descriptor-shadow-v1',
                **descriptor_audit,
                'descriptor_index_sha256': descriptor_index_sha256,
                'per_demand_limit': args.witness_descriptor_per_demand_limit,
                'prerank_family_limit': args.witness_descriptor_prerank_family_limit,
                'online_llm_calls': 0,
                'seconds': time.perf_counter() - descriptor_started,
                'answer_path_unchanged': True,
                'grounding_input_replaced': False,
                'boundary': ('disk descriptor lookup plus frozen reranker; branches remain candidate-only '
                             'and cannot replace source-located or semantic witnesses'),
            }
        attempt_levels = ([] if args.query_contract and query_contract['answer_mode'] == 'contextual'
                          else frontier_levels)
        if not attempt_levels and not (args.query_contract
                                       and query_contract['answer_mode'] == 'contextual'):
            attempt_levels = [{'level': 0, 'quota_per_demand': 0,
                               'selected_candidate_indices': [], 'candidate_count': 0,
                               'covered_demand_count': 0, 'is_complete': True}]
        grounded_witnesses = []
        grounding_rejections = []
        grounding_result = None
        grounding_candidates, grounding_aliases, omitted = [], {}, []
        selected_frontier_level = None
        grounding_source_records = []
        for frontier_level in attempt_levels:
            selected_records = candidates_for_frontier_level(witness_records, frontier_level)
            grounding_source_records = selected_records
            grounding_candidates, grounding_aliases, omitted, included = [], {}, [], set()
            ranked_domains = [sorted((record for record in selected_records
                if record['demand_id'] == demand['demand_id']),
                key=lambda record: (-record['score'], record['record_id'])) for demand in demands]
            for rank in range(max(map(len, ranked_domains), default=0)):
                for domain in ranked_domains:
                    if rank >= len(domain):
                        continue
                    candidate = domain[rank]
                    if candidate['base_record_id'] in included:
                        continue
                    alias = f"r{len(grounding_candidates)}"
                    compact = compact_record(candidate, objects, alias)
                    proposed = grounding_candidates + [compact]
                    payload = {'demands': demands, 'relations': relations, 'candidates': proposed,
                               'output_contract': 'JSON only'}
                    if operation_plan:
                        payload['query_operations'] = operation_plan
                    if args.query_contract:
                        payload['original_question'] = query
                    proposal_started = time.perf_counter()
                    proposal_inputs = prepare_inputs(processor, [
                        {'role': 'system', 'content': WITNESS_PROMPT},
                        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
                    grounding_budget_prepare_seconds += time.perf_counter() - proposal_started
                    if proposal_inputs.input_ids.shape[1] > args.input_budget:
                        omitted.append(candidate['witness_id'])
                        continue
                    grounding_aliases[alias] = {**candidate, 'record_id': alias}
                    grounding_candidates.append(compact)
                    included.add(candidate['base_record_id'])
            grounding_payload = {'demands': demands, 'relations': relations,
                                 'candidates': grounding_candidates, 'output_contract': 'JSON only'}
            if operation_plan:
                grounding_payload['query_operations'] = operation_plan
            if args.query_contract:
                grounding_payload['original_question'] = query
            grounding_result = structured_call_with_profile(reader, processor, WITNESS_PROMPT,
                grounding_payload, max_tokens=4096,
                budget=args.input_budget, json_schema=WITNESS_SCHEMA,
                enabled=args.pre_reader_operator_profile, lm_head_dispatch=lm_head_dispatch)
            grounding_reader_seconds += grounding_result.get('reader_seconds', 0.0)
            grounding_validation_started = time.perf_counter()
            grounded, grounding_rejections = validate_witnesses(expand_compact_sources(
                grounding_result.get('parsed'), grounding_aliases, objects),
                demands, relations, grounding_aliases, objects)
            grounded_witnesses = []
            for witness in grounded:
                values = [grounding_aliases[alias] for alias in witness['record_ids']
                          if alias in grounding_aliases]
                if not values:
                    continue
                demand = next(item for item in demands if item['demand_id'] == witness['demand_id'])
                support_scores = []
                for value in values:
                    score_key = (demand['demand_id'], value['base_record_id'])
                    if score_key not in score_by_demand_record:
                        score_by_demand_record[score_key] = float(reranker.score(
                            retrieval_by_demand[demand['demand_id']], [value['retrieval_text']])[0])
                    support_scores.append(score_by_demand_record[score_key])
                combined_score = min(support_scores)
                witness = {**witness, 'score': float(combined_score),
                           'base_record_id': '|'.join(sorted(value['base_record_id'] for value in values)),
                           'retrieval_record_ids': [value['base_record_id'] for value in values]}
                grounded_witnesses.append(witness)
            covered_demands = {witness['demand_id'] for witness in grounded_witnesses}
            required_demands = {demand['demand_id'] for demand in demands}
            required_relations = {relation['relation_id'] for relation in relations
                                  if relation['kind'] == 'source_relation'}
            provided_relations = {relation_id for witness in grounded_witnesses
                                  for relation_id in witness.get('source_relations', [])}
            contract_complete = (required_demands <= covered_demands
                                 and required_relations <= provided_relations)
            grounding_attempts.append({
                'level': frontier_level['level'],
                'quota_per_demand': frontier_level['quota_per_demand'],
                'frontier_candidate_count': frontier_level['candidate_count'],
                'grounding_candidate_count': len(grounding_candidates),
                'omitted_candidate_count': len(omitted),
                'covered_demand_count': len(covered_demands),
                'required_demand_count': len(required_demands),
                'provided_relation_count': len(provided_relations),
                'required_relation_count': len(required_relations),
                'contract_complete': contract_complete,
                'reader_timing': {
                    'input_tokens': grounding_result.get('structured_input_tokens'),
                    'prefill_seconds': grounding_result.get('structured_reader_prefill_seconds'),
                    'decode_seconds': grounding_result.get('structured_reader_decode_seconds'),
                    'first_token_seconds': grounding_result.get('structured_first_token_seconds'),
                    'output_tokens': len(grounding_result.get('answer_token_ids', [])),
                    'decode_phase_profile': grounding_result.get('decode_phase_profile'),
                },
                'linear_operator_trace': grounding_result.get('structured_linear_trace'),
                'operator_profile': grounding_result.get('structured_operator_profile'),
            })
            grounding_validation_seconds += time.perf_counter() - grounding_validation_started
            selected_frontier_level = frontier_level
            if contract_complete or frontier_level.get('is_complete'):
                break

        def materialize_shadow(witness, aliases):
            values = [aliases[alias] for alias in witness['record_ids']
                      if alias in aliases]
            if not values:
                return None
            demand = next(item for item in demands
                          if item['demand_id'] == witness['demand_id'])
            support_scores = []
            for value in values:
                score_key = (demand['demand_id'], value['base_record_id'])
                if score_key not in score_by_demand_record:
                    score_by_demand_record[score_key] = float(reranker.score(
                        retrieval_by_demand[demand['demand_id']],
                        [value['retrieval_text']])[0])
                support_scores.append(score_by_demand_record[score_key])
            return {
                **witness, 'score': float(min(support_scores)),
                'base_record_id': '|'.join(sorted(
                    value['base_record_id'] for value in values)),
                'retrieval_record_ids': [value['base_record_id'] for value in values],
            }

        multi_witness_shadow = None
        multi_witness_shadow_witnesses = []
        if (args.multi_witness_shadow
                and not (args.query_contract and query_contract['answer_mode'] == 'contextual')):
            multi_witness_started = time.perf_counter()
            shadow_result = structured_call_with_profile(
                reader, processor, MULTI_WITNESS_PROMPT, grounding_payload,
                max_tokens=4096, budget=args.input_budget, json_schema=WITNESS_SCHEMA,
                enabled=args.pre_reader_operator_profile, lm_head_dispatch=lm_head_dispatch)
            shadow_grounded, shadow_rejections = validate_witnesses(
                expand_compact_sources(shadow_result.get('parsed'), grounding_aliases, objects),
                demands, relations, grounding_aliases, objects)
            for witness in shadow_grounded:
                materialized = materialize_shadow(witness, grounding_aliases)
                if materialized is not None:
                    multi_witness_shadow_witnesses.append(materialized)
            initial_shadow_witness_count = len(multi_witness_shadow_witnesses)
            multi_witness_shadow_witnesses = limit_witness_alternatives(
                multi_witness_shadow_witnesses,
                args.initial_witness_per_demand_limit)

            search_rounds = []
            search_rejections = []
            search_reader_seconds = 0.0
            search_output_tokens = 0
            for round_index in range(args.independent_witness_rounds):
                by_demand = {demand['demand_id']: [] for demand in demands}
                for witness in multi_witness_shadow_witnesses:
                    by_demand[witness['demand_id']].append(witness)
                targets = [demand for demand in demands
                           if not has_independent_branch_pair(
                               by_demand[demand['demand_id']])]
                if not targets:
                    break
                round_attempts, round_accepted = [], 0
                for demand in targets:
                    existing = by_demand[demand['demand_id']]
                    excluded_families = set().union(*(
                        branch_source_families(witness) for witness in existing)) if existing else set()
                    allowed, seen_bases = [], set()
                    for candidate_record in grounding_aliases.values():
                        if candidate_record['demand_id'] != demand['demand_id']:
                            continue
                        base_id = (candidate_record.get('base_record_id')
                                   or candidate_record.get('record_id'))
                        family = source_family(base_id)
                        if (not family or family in excluded_families
                                or base_id in seen_bases):
                            continue
                        seen_bases.add(base_id)
                        allowed.append(candidate_record)
                    allowed.sort(key=lambda item: (
                        -float(item['score']), str(item['base_record_id'])))
                    allowed = allowed[:args.independent_witness_candidate_limit]
                    attempt = {
                        'round': round_index,
                        'demand_id': demand['demand_id'],
                        'excluded_source_families': sorted(excluded_families),
                        'candidate_count': len(allowed),
                        'candidate_source_families': sorted({
                            source_family(item.get('base_record_id') or item.get('record_id'))
                            for item in allowed}),
                    }
                    if not allowed:
                        attempt['status'] = 'no_unseen_candidate_family'
                        round_attempts.append(attempt)
                        continue
                    local_aliases, local_candidates = {}, []
                    for alias_index, candidate_record in enumerate(allowed):
                        alias = f'r{alias_index}'
                        local_aliases[alias] = {**candidate_record, 'record_id': alias}
                        compact = compact_record(candidate_record, objects, alias)
                        compact['source_family'] = source_family(
                            candidate_record.get('base_record_id')
                            or candidate_record.get('record_id'))
                        local_candidates.append(compact)
                    local_payload = {
                        'demands': [demand],
                        'relations': relations,
                        'candidates': local_candidates,
                        'excluded_source_families': sorted(excluded_families),
                        'output_contract': 'JSON only',
                    }
                    if operation_plan:
                        local_payload['query_operations'] = operation_plan
                    if args.query_contract:
                        local_payload['original_question'] = query
                    local_result = structured_call_with_profile(
                        reader, processor, INDEPENDENT_WITNESS_PROMPT, local_payload,
                        max_tokens=4096, budget=args.input_budget, json_schema=WITNESS_SCHEMA,
                        enabled=args.pre_reader_operator_profile,
                        lm_head_dispatch=lm_head_dispatch)
                    search_reader_seconds += local_result.get('reader_seconds', 0.0)
                    search_output_tokens += len(local_result.get('answer_token_ids', []))
                    local_grounded, local_rejections = validate_witnesses(
                        expand_compact_sources(local_result.get('parsed'), local_aliases, objects),
                        [demand], relations, local_aliases, objects)
                    accepted = 0
                    for witness in local_grounded:
                        materialized = materialize_shadow(witness, local_aliases)
                        if (materialized is not None
                                and independent_of_existing(materialized, existing)):
                            multi_witness_shadow_witnesses.append(materialized)
                            existing.append(materialized)
                            accepted += 1
                    round_accepted += accepted
                    attempt.update({
                        'status': 'accepted' if accepted else 'no_independent_witness',
                        'returned_witness_count': len(local_grounded),
                        'accepted_independent_witness_count': accepted,
                        'rejections': local_rejections,
                        'remaining_unseen_family_count': len({
                            source_family(item.get('base_record_id') or item.get('record_id'))
                            for item in allowed}),
                    })
                    search_rejections.extend({'round': round_index,
                                              'demand_id': demand['demand_id'], **item}
                                             for item in local_rejections)
                    round_attempts.append(attempt)
                search_rounds.append({
                    'round': round_index,
                    'target_demand_count': len(targets),
                    'accepted_independent_witness_count': round_accepted,
                    'attempts': round_attempts,
                })
                if round_accepted == 0:
                    break

            multi_witness_shadow = {
                'status': 'independent_search_complete',
                'witness_count': len(multi_witness_shadow_witnesses),
                'initial_witness_count_before_limit': initial_shadow_witness_count,
                'initial_witness_per_demand_limit': args.initial_witness_per_demand_limit,
                'independent_demand_count': sum(
                    has_independent_branch_pair([
                        item for item in multi_witness_shadow_witnesses
                        if item['demand_id'] == demand['demand_id']])
                    for demand in demands),
                'rejections': shadow_rejections,
                'search_rejections': search_rejections,
                'search_rounds': search_rounds,
                'candidate_count': len(grounding_candidates),
                'witnesses': multi_witness_shadow_witnesses,
                'seconds': time.perf_counter() - multi_witness_started,
                'search_reader_seconds': search_reader_seconds,
                'search_output_tokens': search_output_tokens,
                'reader_timing': {
                    'input_tokens': shadow_result.get('structured_input_tokens'),
                    'prefill_seconds': shadow_result.get('structured_reader_prefill_seconds'),
                    'decode_seconds': shadow_result.get('structured_reader_decode_seconds'),
                    'first_token_seconds': shadow_result.get('structured_first_token_seconds'),
                    'output_tokens': len(shadow_result.get('answer_token_ids', [])),
                },
                'search_boundary': ('answer path unchanged; independent means disjoint source families, '
                                    'not semantic entailment'),
                'answer_path_unchanged': True,
            }
        if args.query_contract and query_contract['answer_mode'] == 'contextual':
            grounding_result = {'parsed': {'witnesses': []}, 'answer': '',
                                'reader_seconds': 0., 'stop_reason': 'not_applicable_contextual',
                                'answer_token_ids': []}
        w12_after_grounding = w12_snapshot()
        grounding_seconds = time.perf_counter() - grounding_started
        if proof_query_shadow is not None:
            proof_query_shadow['host_coverage'] = audit_host_grounding_coverage(
                proof_query_shadow['result'], grounded_witnesses)
            proof_query_shadow['safe_to_replace_host_grounding'] = (
                proof_query_shadow['host_coverage']['safe_to_replace_host_grounding'])
        if evidence_state_prefilter_shadow is not None:
            frontier_audit = audit_evidence_prefilter(
                evidence_state_prefilter_shadow, witness_records, grounded_witnesses)
            evidence_state_prefilter_shadow = enforce_host_witness_coverage(
                evidence_state_prefilter_shadow, frontier_audit, witness_records)
        plan_started = time.perf_counter()
        empty_messages = [{'role': 'system', 'content': READER_SYSTEM}, {'role': 'user', 'content': [
            {'type': 'text', 'text': reader_context(query, manifest['effective_timestamp'], '', operation_plan)}]}]
        fixed_inputs = prepare_inputs(processor, empty_messages)
        fixed_prompt_tokens = int(fixed_inputs.input_ids.shape[1])

        def exact_render(object_ids):
            identity = tuple(sorted(object_ids))
            if identity in render_cache:
                return render_cache[identity]
            selected = materialize(objects, identity)
            loaded = []
            for ref in selected['images']:
                image_identity = ref['image_sha256']
                if image_identity not in image_cache:
                    image_cache[image_identity] = load_image(args.image_archive, ref)
                loaded.append(image_cache[image_identity][0])
            content = [{'type': 'text', 'text': reader_context(
                query, manifest['effective_timestamp'], selected['text'], operation_plan)}]
            for ref, image in zip(selected['images'], loaded):
                content.append({'type': 'text', 'text': image_source_header(ref)})
                content.append({'type': 'image', 'image': image})
            inputs = prepare_inputs(processor, [{'role': 'system', 'content': READER_SYSTEM},
                                                {'role': 'user', 'content': content}], loaded)
            assessment = {'feasible': int(inputs.input_ids.shape[1]) <= args.input_budget,
                          'cost': int(inputs.input_ids.shape[1]), 'text': selected['text'],
                          'text_tokens': int(len(processor.tokenizer.encode(selected['text'], add_special_tokens=False))),
                          'visual_tokens': sum(image_costs[item['image_sha256']] for item in selected['images']),
                          'image_count': len(selected['images']), 'exact_input_tokens': int(inputs.input_ids.shape[1])}
            render_cache[identity] = assessment
            return assessment

        def evaluate(object_ids):
            selected = [objects[object_id] for object_id in object_ids]
            text_objects = sorted((item for item in selected if item['kind'] == 'text'),
                                  key=lambda item: item['ordinal'])
            image_objects = [item for item in selected if item['kind'] == 'image']
            text = '\n'.join(item['text'] for item in text_objects)
            text_tokens = len(processor.tokenizer.encode(text, add_special_tokens=False))
            visual_tokens = sum(image_costs[item['source_sha256']] for item in image_objects)
            rough = {'feasible': text_tokens <= args.total_text_tokens
                    and len(image_objects) <= args.max_images
                    and fixed_prompt_tokens + text_tokens + visual_tokens <= args.input_budget,
                    'cost': text_tokens + visual_tokens, 'text': text,
                    'text_tokens': text_tokens, 'visual_tokens': visual_tokens,
                    'image_count': len(image_objects)}
            if not rough['feasible']:
                return rough
            exact = exact_render(object_ids)
            exact['feasible'] = exact['feasible'] and text_tokens <= args.total_text_tokens and len(image_objects) <= args.max_images
            return exact

        plan = solve_witness_join(demands, grounded_witnesses, relations, evaluate,
                                  time_limit=args.planner_time_limit, max_states=100000)
        proof_source = 'grounded_witnesses'
        evidence_records = []
        if not plan['feasible'] or parse_status != 'ok':
            strict_failure = ({'reason': 'invalid_query_graph', 'query_parse_status': parse_status,
                               'untrusted_fallback_solver_result': plan}
                              if parse_status != 'ok' else plan)
            package_candidates = {}
            for candidate in witness_records:
                previous = package_candidates.get(candidate['base_record_id'])
                if previous is None or candidate['score'] > previous['score']:
                    package_candidates[candidate['base_record_id']] = {
                        **records[candidate['query_rank']], 'score': candidate['score'],
                        'object_ids': list(dict.fromkeys(candidate['object_ids']
                            + candidate.get('retrieval_context_object_ids', [])))}
            package = select_retrieval_package(list(package_candidates.values()), evaluate)
            if not package['feasible']:
                raise RuntimeError(f'no budget-feasible evidence package for {question_id}')
            plan = {**package, 'selected_witnesses': [], 'strict_proof_failure': True,
                    'strict_proof_missing_demands': strict_failure.get('missing_demands', []),
                    'failed_strict_search': strict_failure,
                    'query_graph_invalid': parse_status != 'ok',
                    'proof_status': 'no_complete_source_located_proof',
                    'selection_mode': 'nonproof_retrieval_package', 'semantic_proof': False}
            evidence_records = package['selected_records']
            proof_source = 'explicit_nonproof_retrieval_package'
        else:
            evidence_records = plan.get('selected_witnesses', [])
            plan['proof_status'] = 'source_located_not_semantically_verified'
            plan['selection_mode'] = 'source_located_proof_join'
            plan['semantic_proof'] = False
        selected_witnesses = plan.get('selected_witnesses', [])
        plan['selected_records'] = evidence_records
        plan['demands'], plan['relations'] = demands, relations
        if args.query_contract:
            plan['query_contract'] = query_contract
            plan['query_operations'] = query_contract['operations']
            plan['reader_operation_plan'] = operation_plan
            plan['operations_require_source_witnesses'] = False
        plan['required_demand_ids'] = [demand['demand_id'] for demand in demands]
        plan['proof_hypergraph'] = {
            'and_edges': [{'head': 'query', 'tail': [demand['demand_id'] for demand in demands],
                           'kind': 'all_required_demands'}]
                         + [{'head': witness['witness_id'], 'tail': witness['object_ids'],
                             'source_anchors': witness['anchors']} for witness in grounded_witnesses],
            'or_edges': [{'head': demand['demand_id'], 'tail': [witness['witness_id']
                         for witness in grounded_witnesses if witness['demand_id'] == demand['demand_id']]}
                         for demand in demands],
            'required_conjunction': plan['required_demand_ids'], 'query_relations': relations,
            'witnesses': grounded_witnesses,
            'boundary': 'machine-checked source location and query bindings only; no semantic entailment certificate'}
        if args.query_contract:
            plan['proof_hypergraph']['operation_edges'] = query_contract['operations']
            plan['proof_hypergraph']['contextual_not_positive_conjunction'] = query_contract['answer_mode'] == 'contextual'
        plan['selected_record_ids'] = [item['base_record_id'] if 'base_record_id' in item
                                       else item['record_id'] for item in evidence_records]
        plan['accepted_atom_ids'] = [objects[object_id]['atom_id'] for object_id in plan['accepted_object_ids']
                                     if objects[object_id]['kind'] == 'text']
        plan['text'] = plan['assessment']['text']
        plan['text_tokens_estimate'] = plan['assessment']['text_tokens']
        plan['visual_tokens_estimate'] = plan['assessment']['visual_tokens']
        plan['planner_seconds'] = time.perf_counter() - plan_started
        plan['query_parse'] = {'status': parse_status, 'raw': parse_result.get('answer', ''),
                               'parsed': parse_result.get('parsed'), 'seconds': parse_result.get('reader_seconds'),
                               'stop_reason': parse_result['stop_reason'],
                               'output_token_count': len(parse_result['answer_token_ids']),
                               'timing': {
                                   'structured_input_tokens': parse_result.get('structured_input_tokens'),
                                   'prepare_seconds': parse_result.get('structured_prepare_seconds'),
                                   'grammar_compile_seconds': parse_result.get('structured_grammar_compile_seconds'),
                                   'reader_prefill_seconds': parse_result.get('structured_reader_prefill_seconds'),
                                   'reader_decode_seconds': parse_result.get('structured_reader_decode_seconds'),
                                   'first_token_seconds': parse_result.get('structured_first_token_seconds'),
                'postprocess_seconds': parse_result.get('structured_postprocess_seconds'),
               }}
        plan['grounding'] = {'status': 'complete' if grounded_witnesses else 'no_valid_witness',
            'source': proof_source, 'raw': grounding_result.get('answer', ''),
            'parsed': grounding_result.get('parsed'), 'rejections': grounding_rejections,
            'candidate_count': len(grounding_candidates), 'witness_count': len(grounded_witnesses),
            'alias_to_record_id': {alias: record['base_record_id'] for alias, record in grounding_aliases.items()},
            'stop_reason': grounding_result['stop_reason'],
            'output_token_count': len(grounding_result['answer_token_ids']),
            'omitted_whole_record_ids_due_to_context': omitted,
            'seconds': grounding_result.get('reader_seconds'),
            'timing': {
                'structured_input_tokens': grounding_result.get('structured_input_tokens'),
                'prepare_seconds': grounding_result.get('structured_prepare_seconds'),
                'grammar_compile_seconds': grounding_result.get('structured_grammar_compile_seconds'),
                'reader_prefill_seconds': grounding_result.get('structured_reader_prefill_seconds'),
                'reader_decode_seconds': grounding_result.get('structured_reader_decode_seconds'),
                'first_token_seconds': grounding_result.get('structured_first_token_seconds'),
                'postprocess_seconds': grounding_result.get('structured_postprocess_seconds'),
                'budget_prepare_seconds': grounding_budget_prepare_seconds,
                'validation_and_join_seconds': grounding_validation_seconds,
                'total_reader_seconds_across_attempts': grounding_reader_seconds,
            },
            'frontier_seed_per_demand': args.candidate_frontier_seed_per_demand,
            'frontier_selected_level': (selected_frontier_level['level']
                                        if selected_frontier_level is not None else None),
            'frontier_attempts': grounding_attempts}
        if proof_query_shadow is not None:
            plan['proof_query_shadow'] = proof_query_shadow
        if evidence_state_prefilter_shadow is not None:
            plan['evidence_state_prefilter_shadow'] = evidence_state_prefilter_shadow
        if multi_witness_shadow is not None:
            needed_object_ids = set(plan['accepted_object_ids'])
            for witness in multi_witness_shadow_witnesses:
                needed_object_ids.update(witness['object_ids'])
            for candidate_record in grounding_source_records:
                needed_object_ids.update(candidate_record['object_ids'])
            object_costs = {}
            for identity in needed_object_ids:
                item = objects[identity]
                if item['kind'] == 'text':
                    object_costs[identity] = float(len(processor.tokenizer.encode(
                        item['text'], add_special_tokens=False)))
                else:
                    object_costs[identity] = float(image_costs[item['source_sha256']])
            multi_witness_audit = audit_multi_witness(
                demands, multi_witness_shadow_witnesses,
                candidates=grounding_source_records, relations=relations,
                selected_object_ids=plan['accepted_object_ids'],
                object_costs=object_costs, max_candidate_alternatives=5,
                time_limit=args.planner_time_limit)
            plan['multi_witness_shadow'] = {
                **multi_witness_shadow, **multi_witness_audit,
                'answer_path_unchanged': True,
            }
        if witness_descriptor_shadow is not None:
            plan['witness_descriptor_shadow'] = witness_descriptor_shadow
        plan['retrieval'] = {'records': len(records), 'witness_candidates': len(witness_records),
                             'candidate_pool_count': len(candidate_pool_records),
                             'candidate_pool_width': candidate_width,
                             'demand_count': len(demands), 'query_vector_seconds': parse_vector_seconds,
                             'timing': {
                                 'query_parser_total_seconds': parse_result.get('reader_seconds'),
                                 'query_parser_input_tokens': parse_result.get('structured_input_tokens'),
                'query_parser_first_token_seconds': parse_result.get('structured_first_token_seconds'),
                'query_parser_decode_phase_profile': parse_result.get('decode_phase_profile'),
                'query_vector_encode_seconds': query_vector_seconds,
                                 'dense_search_seconds': dense_search_seconds,
                                 'reranker_seconds': reranker_seconds,
                                 'frontier_compile_seconds': frontier_compile_seconds,
                                 'combined_parse_retrieval_seconds': parse_vector_seconds,
                             },
                             'linear_operator_trace': parse_result.get('structured_linear_trace'),
                             'operator_profile': parse_result.get('structured_operator_profile'),
                             'frozen_reranker_seconds': reranker_seconds,
                             'score_semantics': 'Qwen3-Reranker-0.6B frozen yes-minus-no logit on retained text/locator candidates'}
        if args.query_contract:
            plan['retrieval']['views'] = demand_texts
            plan['retrieval']['original_question_preserved'] = all(
                json.loads(view)['original_question'] == query for view in demand_texts)
        source_hints = {}
        for item in evidence_records:
            source_record = next((record for record in records
                                  if record['record_id'] == item.get('record_id')), None)
            if source_record and source_record.get('image_source'):
                source_hints[source_record['image_object_id']] = source_record['image_source']
            source_hints.update(item.get('source_hints', {}))
        materialized = materialize(objects, plan['accepted_object_ids'], source_hints)
        loaded = []
        for ref in materialized['images']:
            identity = ref['image_sha256']
            if identity not in image_cache:
                image_cache[identity] = load_image(args.image_archive, ref)
            if image_cache[identity][1] != identity:
                raise ValueError('selected source image bytes differ from pinned content identity')
            loaded.append(image_cache[identity][0])
        context = reader_context(query, manifest['effective_timestamp'], materialized['text'], operation_plan)
        content = [{'type': 'text', 'text': context}]
        for ref, image in zip(materialized['images'], loaded):
            content.append({'type': 'text', 'text': image_source_header(ref)})
            content.append({'type': 'image', 'image': image})
        image_bytes = {ref['image_sha256']: image.width * image.height * 3
                       for ref, image in zip(materialized['images'], loaded)}
        messages = [{'role': 'system', 'content': READER_SYSTEM}, {'role': 'user', 'content': content}]
        inputs = prepare_inputs(processor, messages, loaded)
        if inputs.input_ids.shape[1] > args.input_budget:
            raise ValueError(f'final processor input exceeds budget for {question_id}')
        plan['final_render'] = {'input_tokens': int(inputs.input_ids.shape[1]),
            'image_count': len(loaded), 'image_sha256s': [ref['image_sha256'] for ref in materialized['images']],
            'text_tokens': int(len(processor.tokenizer.encode(materialized['text'], add_special_tokens=False))),
            'processor_input_sha256': hashlib.sha256(inputs.input_ids.detach().cpu().numpy().tobytes()).hexdigest()}
        if (inputs.input_ids.shape[1] > args.input_budget
                or inputs.input_ids.shape[1] != plan['assessment']['exact_input_tokens']):
            raise RuntimeError('solver/render contract mismatch: final processor input exceeded budget')
        if args.capture_reader_inputs:
            args.capture_reader_inputs.mkdir(parents=True, exist_ok=True)
            capture_path = args.capture_reader_inputs / f'{args.user_id}_{question_id}.pt'
            if capture_path.exists():
                raise ValueError(f'refusing to overwrite captured Reader input: {capture_path}')
            torch.save({name: value.detach().cpu().contiguous()
                        for name, value in inputs.items() if isinstance(value, torch.Tensor)},
                       capture_path)
            row = {
                'person_id': args.user_id,
                'question_id': question_id,
                'request_index': position,
                'input_path': str(capture_path),
                'input_tokens': int(inputs.input_ids.shape[1]),
                'image_count': len(loaded),
                'input_ids_sha256': hashlib.sha256(
                    inputs.input_ids.detach().cpu().numpy().tobytes()).hexdigest(),
                'pre_reader_seconds': time.perf_counter() - request_started,
                'persistent_replay': ({
                    'enabled': replay_queries is not None,
                    'visible_session_count': len(visible_session_ids) if visible_session_ids is not None else None,
                    'visible_record_count': visible_record_count,
                    'full_record_count': len(base_records),
                    'runtime_labels_read': False,
                } if replay_queries is not None else {
                    'enabled': False,
                    'runtime_labels_read': False,
                }),
                'evaluation_label_files_read': [],
            }
            rows.append(row)
            completed_ids.add(question_id)
            write_json(partial_path, {'complete': False,
                'person_id': args.user_id, 'question_ids': question_ids, 'rows': rows,
                'shard_index': args.shard_index, 'shard_count': args.shard_count,
                'capacity_capture_only': True,
                'cohort_sha256': file_digest(args.cohort),
                'evaluation_label_files_read': []})
            print(json.dumps({'question_id': question_id, 'position': position + 1,
                              'captured_input_tokens': row['input_tokens'],
                              'images': row['image_count'],
                              'pre_reader_seconds': row['pre_reader_seconds']}, ensure_ascii=False),
                  flush=True)
            continue
        lifecycle = compile_lifecycle(objects, plan['accepted_object_ids'],
            reader_input_tokens=int(inputs.input_ids.shape[1]), image_bytes=image_bytes)
        backend_plan = choose_backend(
            reader_dtype='bf16', object_frontier_bytes=lifecycle['peak_frontier_bytes'],
            input_tokens=int(inputs.input_ids.shape[1]), lifecycle=lifecycle,
            question_id=question_id, grounding_route=grounding_route)
        execution_events = [{'phase': 'query_parse_and_retrieval', 'seconds': parse_vector_seconds,
                             'object_ids': [], 'consumer': 'proof_planner',
                             'timing': plan['retrieval']['timing']},
                            {'phase': 'source_grounding', 'seconds': grounding_seconds,
                             'object_ids': [], 'consumer': 'proof_planner',
                             'timing': plan['grounding']['timing'],
                             'backend': grounding_route['chosen_backend'],
                             'frontier_state_count': grounding_frontier['state_count'],
                             'frontier_packed_bytes': grounding_frontier['packed_bytes']},
                            {'phase': 'proof_materialize', 'seconds': time.perf_counter() - plan_started,
                             'object_ids': plan['accepted_object_ids'], 'consumer': 'reader_prefill',
                             'backend': backend_plan['phase_routes'][0]['backend'],
                             'bytes': sum(len(objects[object_id].get('text', '').encode())
                                          if objects[object_id]['kind'] == 'text' else image_bytes[objects[object_id]['source_sha256']]
                                          for object_id in plan['accepted_object_ids'])},
                            {'phase': 'reader_prefill', 'object_ids': plan['accepted_object_ids'],
                             'consumer': 'decode', 'backend': 'host', 'input_allocations': []},]
        generation_started = time.perf_counter()
        generated = greedy(reader, processor, inputs, max_tokens=args.max_new_tokens)
        w12_after_reader = w12_snapshot()
        w12_stage_dispatch = None
        if w12_dispatch is not None:
            w12_stage_dispatch = {
                'query_parser': w12_stage_delta(
                    w12_request_started, w12_after_parser),
                'source_grounding': w12_stage_delta(
                    w12_after_parser, w12_after_grounding),
                'reader_generation': w12_stage_delta(
                    w12_after_grounding, w12_after_reader),
            }
        execution_events[3]['seconds'] = generated['prefill_seconds']
        execution_events[3]['input_allocations'] = generated['prefill_input_allocations']
        execution_events.extend([{'phase': 'reader_decode', 'seconds': generated['decode_seconds'],
                                  'object_ids': plan['accepted_object_ids'], 'consumer': 'answer', 'backend': 'host'},
                                 {'phase': 'release', 'seconds': 0.0,
                                  'object_ids': plan['accepted_object_ids'], 'reader_inputs': generated['released_before_decode'],
                                  'consumer': 'none'}])
        if generated['input_tokens'] != lifecycle['reader_input_tokens']:
            raise RuntimeError('reader input token count changed after backend planning')
        released_images = len(loaded)
        del loaded, image_cache, inputs
        row = {'person_id': args.user_id, 'question_id': question_id,
            'question_sha256': hashlib.sha256(query.encode()).hexdigest(), 'answer': generated['answer'],
            'answer_sha256': hashlib.sha256(generated['answer'].encode()).hexdigest(),
            'answer_token_ids': generated['answer_token_ids'], 'stop_reason': generated['stop_reason'],
            'service_seconds': time.perf_counter() - request_started,
            'ttft_seconds': generation_started - request_started + generated['first_token_from_reader_start_seconds'],
            'prefill_seconds': generated['prefill_seconds'],
            'decode_seconds': generated['decode_seconds'], 'input_tokens': generated['input_tokens'],
            'w12_stage_dispatch': w12_stage_dispatch,
            'images': materialized['images'], 'accepted_atom_ids': materialized['accepted_atom_ids'],
            'materialized_object_ids': materialized['materialized_object_ids'],
            'proof_plan': plan, 'execution_events': execution_events,
            'grounding_frontier': grounding_frontier,
            'grounding_frontier_levels': frontier_levels,
            'grounding_backend_plan': grounding_route,
            'image_rasters_released_after_prefill': released_images,
            'lifecycle_plan': lifecycle, 'backend_plan': backend_plan,
            'evidence_policy': ('source_fact_operation_contract_v1' if args.query_contract else
                                'source_located_witness_join_with_explicit_nonproof_package_v2'),
            'persistent_replay': ({
                'enabled': replay_queries is not None,
                'visible_index_mode': visible_index_mode,
                'visible_session_count': len(visible_session_ids) if visible_session_ids is not None else None,
                'visible_record_count': visible_record_count,
                'full_record_count': len(base_records),
                'visible_vector_bytes': visible_vector_bytes,
                'full_vector_bytes': int(base_vectors.nbytes),
                'visible_index_filter_seconds': visible_index_seconds,
                'runtime_labels_read': False,
            } if replay_queries is not None else {
                'enabled': False, 'visible_index_mode': visible_index_mode,
                'runtime_labels_read': False,
            }),
            'evaluation_label_files_read': [], 'request_index': position}
        rows.append(row)
        completed_ids.add(question_id)
        write_json(partial_path, {'complete': False,
            'person_id': args.user_id, 'question_ids': question_ids, 'rows': rows,
            'shard_index': args.shard_index, 'shard_count': args.shard_count,
            'pre_reader_operator_profile': args.pre_reader_operator_profile,
            'reader_quantization': args.reader_quantization,
            'w12_functional_route': args.w12_functional_route,
            'w12_numeric_mode': args.w12_numeric_mode,
            'w12_bank_count': args.w12_bank_count,
            'w12_tile_rows': args.w12_tile_rows,
            'w12_host_contract_rows': args.w12_host_contract_rows,
            'w12a16_group_size': args.w12a16_group_size,
            'w12_release_source': args.w12_release_source,
            'cohort_sha256': file_digest(args.cohort), 'evaluation_label_files_read': []})
        print(json.dumps({'question_id': question_id, 'position': position + 1,
                          'seconds': row['service_seconds'], 'records': len(plan['selected_records']),
                          'images': released_images, 'input_tokens': generated['input_tokens']}), flush=True)
    w12_stage_summary = aggregate_w12_stage_dispatch(rows)
    w12_dispatch_summary = (w12_dispatch.summary()
                            if w12_dispatch is not None else None)
    if w12_dispatch is not None:
        w12_dispatch.__exit__(None, None, None)
    if lm_head_dispatch is not None:
        lm_head_dispatch.__exit__(None, None, None)
    if args.maintenance_progress:
        write_json(args.maintenance_progress, {
            'person_id': args.user_id,
            'complete': True,
            'question_ids': question_ids,
            'runtime_labels_read': False,
        })
    result = {'complete': True,
        'schema': ('proof-mainline-reader-input-capture-v1' if args.capture_reader_inputs
                   else 'proof-mainline-run-v1'),
        'capacity_capture_only': bool(args.capture_reader_inputs),
        'capture_directory': str(args.capture_reader_inputs) if args.capture_reader_inputs else None,
        'person_id': args.user_id,
        'question_ids': question_ids, 'prepared_audit_sha256': file_digest(args.prepared / 'audit.json'),
        'cohort_sha256': file_digest(args.cohort), 'index_sha256': file_digest(args.index / 'index.json'),
        'reader_model': str(args.reader), 'embedder_model': str(args.embedder),
        'reader_processor_input_budget': args.input_budget, 'total_text_tokens': args.total_text_tokens,
        'max_images': args.max_images, 'max_new_tokens': args.max_new_tokens,
        'reader_quantization': args.reader_quantization,
        'reader_weight_bits': (12 if args.reader_quantization == 'w12a16'
                               else None),
        'reader_w12a16_group_size': (args.w12a16_group_size
                                     if args.reader_quantization == 'w12a16'
                                     else None),
        'reader_quantization_materialization': reader_quantization_materialization,
        'w12_functional_route': args.w12_functional_route,
        'w12_numeric_mode': args.w12_numeric_mode,
        'w12_bank_count': args.w12_bank_count,
        'w12_tile_rows': args.w12_tile_rows,
        'w12_host_contract_rows': args.w12_host_contract_rows,
        'w12a16_group_size': args.w12a16_group_size,
        'w12_release_source': args.w12_release_source,
        'evaluation_label_files_read': [], 'validation_accessed': False, 'holdout_accessed': False,
        'model_load_and_index_seconds': startup_seconds,
        'shard_index': args.shard_index, 'shard_count': args.shard_count,
        'elapsed_seconds': time.perf_counter() - started,
        'measurement_boundary': ('label-free final Reader processor tensors captured after normal retrieval, grounding, and materialization; answer generation skipped'
                                 if args.capture_reader_inputs else
            (
            'diagnostic profile run for query parser and source-grounding operator attribution; '
            'profiler overhead is included in service wall times, which are not performance-comparable'
            if args.pre_reader_operator_profile else
            'server wall-clock online query parse/retrieval/planning/raw-image preparation/prefill/decode; model/index startup included in run metadata only')),
        'query_contract_enabled': args.query_contract,
        'persistent_replay': {
            'enabled': replay_queries is not None,
            'visible_index_mode': args.visible_index_mode if replay_queries is not None else None,
            'runtime_labels_read': False,
            'replay_path': str(args.persistent_replay) if args.persistent_replay else None,
        },
        'pre_reader_operator_profile': {
            'enabled': args.pre_reader_operator_profile,
            'performance_comparable': False if args.pre_reader_operator_profile else None,
            'scope': ('query parser and source-grounding structured model calls; final Reader generation excluded'
                      if args.pre_reader_operator_profile else 'disabled'),
            'profiled_decode_forward_calls_per_model_call': 4,
        },
        'lm_head_functional_dispatch': (lm_head_dispatch.summary()
                                        if lm_head_dispatch is not None else {
                                            'mode': 'off', 'physical_pim_executed': False,
                                            'performance_comparable': None}),
        'reader_w12_bank_dispatch': (w12_dispatch_summary if w12_dispatch_summary
                                     is not None else {
                                         'mode': 'off', 'physical_pim_executed': False,
                                         'performance_comparable': None}),
        'reader_w12_bank_materialization': w12_materialization,
        'w12_stage_dispatch': w12_stage_summary,
        'proof_query_shadow_enabled': args.proof_query_shadow,
        'evidence_state_prefilter_shadow_enabled': args.evidence_state_prefilter_shadow,
        'multi_witness_shadow_enabled': args.multi_witness_shadow,
        'witness_descriptor_shadow_enabled': args.witness_descriptor_shadow,
        'witness_descriptor_index_sha256': descriptor_index_sha256,
        'main_experiment_closed': False, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == '__main__':
    main()
