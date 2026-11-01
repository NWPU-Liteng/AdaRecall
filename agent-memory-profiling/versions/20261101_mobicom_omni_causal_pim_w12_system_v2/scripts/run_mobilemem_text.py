"""MobileMem development execution; labels are deliberately absent from this process."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend import BankPimConfig, BankPimModel
from pim_backend.agent_memory_pim import plan_materialized_scan, replay_reader_residual
from causal_materialization import solve_materialization_plan
from mobilemem_runtime import hybrid_candidates, last_valid_positions, select_development, trajectory_bundles
from prefix_state import PrefixStateSnapshot
from source_closed_scorer import SourceClosedScorer
from reranker_prefix_reuse import SharedPrefixScorer
from state_reuse import account_prefix_reuse, PrefixStateStore
from reader_single_pass import read_single_answer
from model_identity import require_model_roles
from shared_int8_store import CONTRACT as INT8_CONTRACT, SharedInt8Store, physical_bank_topk, quantize

READER = Path('/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B')
EMBEDDER = Path('/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B')
RERANKER = PROJECT / 'models/rerankers/Qwen3-Reranker-0.6B'
SYSTEM = ('你是端侧长期记忆助手。记忆来自用户、助手和手机应用，可能包含旧状态或助手建议。'
          '只依据截至查询时间可见的原始证据作答，区分用户事实与助手建议；没有证据时明确说明证据不足。'
          '回答保持简洁，问题涉及日期、顺序、金额或人物时保留必要细节；只给答案，不要展开推理。')
VERIFY = '请核验上述回答是否被原始记忆支持。若原始记忆支持问题所问事实，即使问题措辞包含示例、前提或同义改写，也直接给出简洁答案；不要因未逐字出现问题措辞而否定已有事实。只有在所问事实确实缺失或证据冲突时才输出“证据不足”。只输出核验后的最终答案，不要解释、不要列举证据、不要复述问题，最多三句。'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def model_hashes(path):
    return {str(item.name): file_hash(item) for item in sorted(path.glob('*'))
            if item.is_file() and (item.suffix in {'.json', '.safetensors', '.model', '.txt', '.jinja', '.tiktoken', '.vocab'})}


def atom_text(atom):
    return (f"[SOURCE {atom['event_id']} chars={atom['start']}:{atom['end']}] "
            f"time={atom['timestamp']} source={atom['source']} role={atom['role']}\n{atom['content']}\n")


def atomize(graph, tokenizer, window=384):
    atoms, by_event = [], {}
    for event in graph['nodes']:
        content = event['content']
        encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
        previous_end = 0
        indices = []
        for position in range(0, len(offsets), window):
            end = len(content) if position + window >= len(offsets) else offsets[position + window][0]
            if end <= previous_end:
                continue
            atom = {key: event[key] for key in ('event_id', 'person_id', 'session_id', 'timestamp', 'source', 'role', 'state_index')}
            atom.update(start=previous_end, end=end, content=content[previous_end:end], ordinal=len(atoms))
            atom['identity'] = hashlib.sha256(atom_text(atom).encode()).hexdigest()
            atom['cost'] = len(tokenizer.encode(atom_text(atom), add_special_tokens=False))
            indices.append(len(atoms))
            atoms.append(atom)
            previous_end = end
        by_event[event['event_id']] = indices
    return atoms, by_event


class FeatureCache:
    def __init__(self, model_path, device, capacity_bytes, contract):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side='left')
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True, dtype=torch.bfloat16,
                                             attn_implementation='sdpa').to(device).eval()
        self.device, self.capacity_bytes, self.contract = device, capacity_bytes, contract
        self.cache = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.encoded_tokens = 0

    def encode(self, texts):
        outputs = []
        for start in range(0, len(texts), 8):
            batch = texts[start:start + 8]
            inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors='pt').to(self.device)
            if inputs['input_ids'].shape[1] > 2048:
                raise ValueError('embedding input exceeds frozen limit; no silent truncation')
            self.encoded_tokens += int(inputs['attention_mask'].sum())
            with torch.inference_mode():
                hidden_states = self.model(**inputs, use_cache=False).last_hidden_state
                last_index = last_valid_positions(inputs['attention_mask'])
                batch_index = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                hidden = hidden_states[batch_index, last_index].float()
                outputs.append(torch.nn.functional.normalize(hidden, dim=-1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, 1024), dtype=np.float32)

    def materialize(self, atoms):
        keys = [hashlib.sha256((self.contract + atom['identity']).encode()).hexdigest() for atom in atoms]
        missing = [index for index, key in enumerate(keys) if key not in self.cache]
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        new = self.encode([atom_text(atoms[index]) for index in missing])
        computed = {keys[index]: vector for index, vector in zip(missing, new)}
        available = {key: self.cache[key] for key in keys if key in self.cache}
        available.update(computed)
        result = []
        for key in keys:
            if key in self.cache:
                vector = self.cache.pop(key)
                self.cache[key] = vector
            else:
                vector = available[key]
                if vector.nbytes <= self.capacity_bytes:
                    while self.bytes + vector.nbytes > self.capacity_bytes:
                        _, removed = self.cache.popitem(last=False)
                        self.bytes -= removed.nbytes
                        self.evictions += 1
                    self.cache[key] = vector
                    self.bytes += vector.nbytes
            result.append(vector)
        return np.stack(result), len(missing)

    def metrics(self):
        return {'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions,
                'resident_bytes': self.bytes, 'capacity_bytes': self.capacity_bytes,
                'encoded_tokens': self.encoded_tokens}


def decode(model, logits, cache, tokenizer, limit, device):
    eos = getattr(model.config, 'eos_token_id', None)
    if eos is None and hasattr(model.config, 'text_config'):
        eos = getattr(model.config.text_config, 'eos_token_id', None)
    eos = tokenizer.eos_token_id if eos is None else eos
    generated, stop = [], set(eos if isinstance(eos, list) else [eos])
    next_id = logits[:, -1, :].argmax(-1, keepdim=True)
    with torch.inference_mode():
        for position in range(limit):
            generated.append(next_id)
            if int(next_id.item()) in stop or position + 1 == limit:
                break
            output = model(input_ids=next_id, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
            next_id = output.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize(device)
    tokens = torch.cat(generated, dim=1)
    return tokenizer.decode(tokens[0], skip_special_tokens=True), int(tokens.shape[1])


def read_answer(model, processor, question, selected_atoms, args):
    function_started = time.perf_counter()
    memory = ''.join(atom_text(atom) for atom in selected_atoms)
    user = f"查询时间：{question['effective_timestamp']}\n原始记忆证据：\n{memory}\n问题：{question['question']}"
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
    first_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[first_prompt], return_tensors='pt').to(args.device)
    probe_messages = messages + [{'role': 'assistant', 'content': 'PREFIX_LAYOUT_PROBE'},
                                 {'role': 'user', 'content': VERIFY}]
    probe_prompt = processor.apply_chat_template(probe_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    probe_tokens = processor.tokenizer.encode(probe_prompt, add_special_tokens=False)
    first_tokens = inputs.input_ids[0].tolist()
    prefix_length = 0
    while prefix_length < min(len(first_tokens), len(probe_tokens)) and first_tokens[prefix_length] == probe_tokens[prefix_length]:
        prefix_length += 1
    if prefix_length < 1:
        raise ValueError('no stable chat-template prefix')
    prefix_length = min(prefix_length, len(first_tokens) - 1)
    memory_tokens = len(processor.tokenizer.encode(memory, add_special_tokens=False))
    if memory_tokens > args.budget_tokens or inputs.input_ids.shape[1] > args.budget_tokens + 2048:
        raise ValueError('serialized memory/prompt exceeds frozen budget')
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        prefix_inputs = {key: value[:, :prefix_length] for key, value in inputs.items()
                         if key in {'input_ids', 'attention_mask', 'mm_token_type_ids'}}
        prefix_output = model(**prefix_inputs, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(args.device)
    prefill_seconds = time.perf_counter() - started
    snapshot = None
    snapshot_started = time.perf_counter()
    try:
        snapshot = PrefixStateSnapshot(prefix_output.past_key_values, prefix_inputs['input_ids'],
                                       getattr(model.model, 'rope_deltas', None),
                                       capacity_bytes=args.prefix_capacity_mib * 1024**2)
        snapshot_meta = {'status': 'saved_before_decode', 'snapshot_bytes': snapshot.bytes,
                         'prefix_tokens': snapshot.tokens, 'prefix_fingerprint': snapshot.prefix_fingerprint}
    except ValueError as error:
        snapshot_meta = {'status': 'capacity_recompute', 'detail': str(error)}
    torch.cuda.synchronize(args.device)
    snapshot_meta['snapshot_seconds'] = time.perf_counter() - snapshot_started
    with torch.inference_mode():
        prefill = model(input_ids=inputs.input_ids[:, prefix_length:],
                        past_key_values=prefix_output.past_key_values, use_cache=True, logits_to_keep=1)
    del prefix_output
    torch.cuda.synchronize(args.device)
    prefill_seconds = time.perf_counter() - started - snapshot_meta['snapshot_seconds']
    first_token_seconds = time.perf_counter() - function_started
    decode_started = time.perf_counter()
    first_text, first_count = decode(model, prefill.logits, prefill.past_key_values,
                                     processor.tokenizer, args.gen_tokens, args.device)
    first_decode_seconds = time.perf_counter() - decode_started
    del prefill
    messages += [{'role': 'assistant', 'content': first_text}, {'role': 'user', 'content': VERIFY}]
    proof_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    proof_inputs = processor(text=[proof_prompt], return_tensors='pt').to(args.device)
    proof_started = time.perf_counter()
    reused, restore_seconds = 0, 0.0
    with torch.inference_mode():
        if snapshot is not None and snapshot.compatible(proof_inputs.input_ids) and proof_inputs.input_ids.shape[1] > snapshot.tokens:
            restore_started = time.perf_counter()
            cache, rope = snapshot.restore(args.device)
            model.model.rope_deltas = rope
            torch.cuda.synchronize(args.device)
            restore_seconds = time.perf_counter() - restore_started
            reused = snapshot.tokens
            proof = model(input_ids=proof_inputs.input_ids[:, reused:], past_key_values=cache,
                          use_cache=True, logits_to_keep=1)
            snapshot_meta['status'] = 'exact_token_prefix_restored'
        else:
            proof = model(**proof_inputs, use_cache=True, logits_to_keep=1)
            snapshot_meta['status'] = 'incompatible_or_capacity_recompute'
    torch.cuda.synchronize(args.device)
    proof_prefill_seconds = time.perf_counter() - proof_started
    final_answer_first_token_seconds = time.perf_counter() - function_started
    validation = {'status': 'not_requested', 'seconds': 0.0}
    if args.validate_prefix and reused:
        validation_started = time.perf_counter()
        saved_rope = getattr(model.model, 'rope_deltas', None)
        with torch.inference_mode():
            reference = model(**proof_inputs, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(args.device)
            reference_prefill_seconds = time.perf_counter() - validation_started
            cached_logits = proof.logits[:, -1, :].float()
            reference_logits = reference.logits[:, -1, :].float()
            difference = (cached_logits - reference_logits).abs()
            same_greedy = bool(torch.equal(cached_logits.argmax(-1), reference_logits.argmax(-1)))
            maximum_difference = float(difference.max())
            reference_scale = float(reference_logits.abs().max())
        torch.cuda.synchronize(args.device)
        validation = {'status': 'pass' if same_greedy and maximum_difference <= 0.5 else 'fail',
                      'greedy_next_token_equal': same_greedy, 'max_absolute_logit_difference': maximum_difference,
                      'max_reference_absolute_logit': reference_scale, 'absolute_tolerance': 0.5,
                      'reference_full_prefill_seconds': reference_prefill_seconds,
                      'scope': 'proof-prefill next-token logits; not full-sequence equivalence',
                      'seconds': time.perf_counter() - validation_started}
        if validation['status'] != 'pass':
            del proof
            proof = reference
            reused = 0
            snapshot_meta['status'] = 'numeric_contract_full_recompute'
            validation['reference_consumed_as_fallback'] = True
            validation['overhead_seconds'] = validation['seconds'] - reference_prefill_seconds
            proof_prefill_seconds += reference_prefill_seconds
        else:
            del reference
            model.model.rope_deltas = saved_rope
            validation['reference_consumed_as_fallback'] = False
            validation['overhead_seconds'] = validation['seconds']
        final_answer_first_token_seconds = time.perf_counter() - function_started
    proof_decode_started = time.perf_counter()
    proof_text, proof_count = decode(model, proof.logits, proof.past_key_values,
                                     processor.tokenizer, args.proof_tokens, args.device)
    snapshot_meta['reused_tokens'] = reused
    snapshot_meta['restore_seconds'] = restore_seconds
    stages = {
        'first': {'prompt_tokens': int(inputs.input_ids.shape[1]), 'generated_tokens': first_count,
                  'causal_kv_reused_tokens': 0, 'residual_prefill_tokens': int(inputs.input_ids.shape[1]),
                  'prefill_seconds': prefill_seconds, 'decode_seconds': first_decode_seconds,
                  'ttft_from_reader_start_seconds': first_token_seconds,
                  'generation_limit_reached': first_count == args.gen_tokens},
        'proof': {'prompt_tokens': int(proof_inputs.input_ids.shape[1]), 'generated_tokens': proof_count,
                  'causal_kv_reused_tokens': reused, 'residual_prefill_tokens': int(proof_inputs.input_ids.shape[1]) - reused,
                  'prefill_seconds': proof_prefill_seconds, 'decode_seconds': time.perf_counter() - proof_decode_started,
                  'ttft_from_reader_start_seconds': final_answer_first_token_seconds,
                  'generation_limit_reached': proof_count == args.proof_tokens},
    }
    snapshot_meta['reuse_accounting'] = account_prefix_reuse(snapshot_meta, validation, stages).as_dict()
    return {'answer': first_text.strip(), 'first_answer': first_text, 'proof_answer': proof_text.strip(),
            'verification_mode': 'draft_answer_preserved_proof_metadata',
            'submitted_answer_source': 'draft',
            'execution_stages': stages, 'prefix_state_reuse': snapshot_meta,
            'prefix_validation': validation,
            'serialized_memory_tokens': memory_tokens, 'reader_seconds': time.perf_counter() - function_started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--sample-count', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--budget-tokens', type=int, default=6000)
    parser.add_argument('--gen-tokens', type=int, default=256)
    parser.add_argument('--proof-tokens', type=int, default=128)
    parser.add_argument('--prefix-capacity-mib', type=int, default=2048)
    parser.add_argument('--feature-capacity-mib', type=int, default=128)
    parser.add_argument('--int8-capacity-mib', type=int, default=128)
    parser.add_argument('--candidate-count', type=int, default=16)
    parser.add_argument('--dense-event-count', type=int, default=64)
    parser.add_argument('--max-search-nodes', type=int, default=10000)
    parser.add_argument('--planner-objective', choices=('causal_materialization',), default='causal_materialization')
    parser.add_argument('--query-support', choices=('off',), default='off')
    parser.add_argument('--reranker-state', choices=('full', 'shared'), default='shared')
    parser.add_argument('--validate-prefix', action='store_true')
    parser.add_argument('--prefix-audit-every', type=int, default=1)
    parser.add_argument('--reader-protocol', choices=('single',), default='single')
    parser.add_argument('--cross-query-prefix', action='store_true')
    args = parser.parse_args()
    if args.prefix_audit_every < 1 or args.dense_event_count < 1:
        raise ValueError('positive audit interval and dense event budget required')
    if args.cross_query_prefix and not args.validate_prefix:
        raise ValueError('cross-query prefix requires the configured numerical audit schedule')
    if args.output.exists() or args.limit < 1:
        raise SystemExit('fresh output and positive development limit required')
    if not (EMBEDDER / 'config_sentence_transformers.json').is_file():
        raise ValueError('embedding model identity is not the frozen sentence-transformer model')
    torch.set_num_threads(2)
    audit = json.loads((args.prepared / 'audit.json').read_text())
    for name, expected in audit['artifacts_sha256'].items():
        if file_hash(args.prepared / name) != expected:
            raise ValueError('prepared source mutation')
    questions = [json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    wanted = select_development(questions, limit=args.limit, sample_count=args.sample_count,
                                shard_count=args.shard_count, shard_index=args.shard_index)
    people = {question['person_id'] for question in wanted}
    if len(people) != 1:
        raise ValueError('run exactly one development user; sealed test is not authorized')
    graphs = json.loads((args.prepared / 'mobilemem_graph.json').read_text())
    graph = next(graph for graph in graphs if graph['person_id'] in people)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    startup_started = time.perf_counter()
    provenance = {'prepared_audit_sha256': file_hash(args.prepared / 'audit.json'),
                  'code_sha256': {path.name: file_hash(path) for path in (Path(__file__), ROOT / 'scripts/mobilemem_runtime.py', ROOT / 'scripts/causal_materialization.py', ROOT / 'scripts/prefix_state.py', ROOT / 'scripts/state_reuse.py', ROOT / 'scripts/trajectory_hypergraph.py', ROOT / 'scripts/reader_single_pass.py', ROOT / 'scripts/residual_profile.py', ROOT / 'scripts/system_audit.py', ROOT / 'scripts/model_identity.py', ROOT / 'scripts/source_closed_scorer.py', ROOT / 'scripts/reranker_prefix_reuse.py', ROOT / 'scripts/intact_bundle_scorer.py', ROOT / 'scripts/joint_bundle_scorer.py', ROOT / 'pim_backend/lpddr_bank_pim.py', ROOT / 'pim_backend/agent_memory_pim.py')},
                  'reader_weights': model_hashes(READER), 'embedding_weights': model_hashes(EMBEDDER),
                  'reranker_weights': model_hashes(RERANKER),
                  'embedding_model_path': str(EMBEDDER),
                  'embedding_contract': json.loads((EMBEDDER / 'config_sentence_transformers.json').read_text()),
                  'embedding_pooling_contract': json.loads((EMBEDDER / '1_Pooling/config.json').read_text()),
                  'embedding_pooling_sha256': file_hash(EMBEDDER / '1_Pooling/config.json'),
                  'profile_sha256': file_hash(ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json')}
    provenance['model_role_validation'] = require_model_roles(provenance)
    provenance['code_sha256']['shared_int8_store.py'] = file_hash(ROOT / 'scripts/shared_int8_store.py')
    import transformers
    provenance['runtime'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                             'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
                             'gpu': torch.cuda.get_device_name(args.device),
                             'embedding_pooling': 'last valid token for either padding direction',
                             'reader_dtype': 'bfloat16', 'reranker_dtype': 'float32', 'torch_threads': 2}
    processor = AutoProcessor.from_pretrained(str(READER), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(str(READER), local_files_only=True,
              dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    features = FeatureCache(EMBEDDER, args.device, args.feature_capacity_mib * 1024**2,
                            json.dumps(provenance['embedding_weights'], sort_keys=True))
    scorer_class = SharedPrefixScorer if args.reranker_state == 'shared' else SourceClosedScorer
    reranker = scorer_class(RERANKER, device=args.device, batch_size=4)
    atoms, by_event = atomize(graph, processor.tokenizer)
    int8_contract = json.dumps({'weights': provenance['embedding_weights'],
                               'pooling': provenance['embedding_pooling_contract'],
                               'quantization': 'clip(rint(fp32 * 127), -127, 127).astype(int8)',
                               'atom_text_version': 1}, sort_keys=True)
    int8_store = SharedInt8Store(atoms, int8_contract, args.int8_capacity_mib * 1024**2)
    profile = json.loads((ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json').read_text())
    backend = BankPimModel(BankPimConfig(**profile['config']))
    args.output.mkdir(parents=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration['int8_store_contract'] = INT8_CONTRACT
    registration = {'selected_question_ids': [question['question_id'] for question in wanted],
                    'configuration': configuration, 'provenance': provenance,
                    'startup_seconds': time.perf_counter() - startup_started,
                    'boundary': 'development only; causal observable closure and shared materialization; PIM uncalibrated simulation'}
    (args.output / 'registration.json').write_text(json.dumps(registration, indent=2))
    rows = []
    pim_resident = set()
    prefix_contract = json.dumps({'weights': provenance['reader_weights'], 'dtype': 'bfloat16',
                                 'attention': 'sdpa', 'single_prompt_version': 2,
                                 'reader_execution_contract': 'unsplit_cold_reference_v1',
                                 'person_id': graph['person_id'], 'runtime': provenance['runtime']}, sort_keys=True)
    prefix_store = (PrefixStateStore(prefix_contract, capacity_bytes=args.prefix_capacity_mib * 1024**2)
                    if args.cross_query_prefix else None)
    for request_index, question in enumerate(wanted):
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        active_stage = 'feature_materialization'
        row = {'question_id': question['question_id'], 'person_id': question['person_id'],
               'effective_timestamp': question['effective_timestamp'], 'split': 'development'}
        before = features.metrics()
        try:
            visible = [atom for atom in atoms if atom['timestamp'] <= question['effective_timestamp']]
            if not visible:
                raise ValueError('no temporally visible atoms')
            feature_started = time.perf_counter()
            vectors, misses = features.materialize(visible)
            torch.cuda.synchronize(args.device)
            materialization_seconds = time.perf_counter() - feature_started
            query_started = time.perf_counter()
            query_vector = features.encode(['Instruct: Given a personal memory question, retrieve supporting historical messages.\nQuery: ' + question['question']])[0]
            torch.cuda.synchronize(args.device)
            query_encoding_seconds = time.perf_counter() - query_started
            int8_started = time.perf_counter()
            int_vectors, int8_trace = int8_store.materialize(visible, vectors, question['effective_timestamp'], int8_contract)
            int_query = quantize(query_vector)
            int8_materialization_seconds = time.perf_counter() - int8_started
            top_k = min(128, len(visible))
            pim_started = time.perf_counter()
            selected, int_scores, bank_reads = physical_bank_topk(int_vectors, int_query, int8_trace['scan_ordinals'], top_k)
            pim_emulation_seconds = time.perf_counter() - pim_started
            reference_started = time.perf_counter()
            reference_vectors = np.clip(np.rint(vectors * 127), -127, 127).astype(np.int8)
            if not np.array_equal(reference_vectors, int_vectors):
                raise ValueError('immutable primary differs from independent FP32 quantization reference')
            reference_scores = reference_vectors.astype(np.int32) @ int_query.astype(np.int32)
            reference = np.lexsort((np.arange(len(visible)), -reference_scores))[:top_k]
            if not np.array_equal(selected, reference) or not np.array_equal(int_scores, reference_scores[reference]):
                raise ValueError('bank scan differs from frozen int32 reference')
            reference_validation_seconds = time.perf_counter() - reference_started
            int8_trace.update(primary_values_reference_exact=True, physical_bank_int32_exact=True,
                              bank_read_transactions=bank_reads, result_payload_bytes=top_k * 8,
                              result_write_bytes=((top_k * 8 + 31) // 32) * 32,
                              query_read_bytes=int_query.nbytes, result_read_bytes=0,
                              verifier_quantized_bytes=reference_vectors.nbytes,
                              verifier_int32_matrix_bytes=reference_vectors.size * 4,
                              materialization_seconds=int8_materialization_seconds,
                              functional_scan_seconds=pim_emulation_seconds)
            pim_model_started = time.perf_counter()
            pim_trace = plan_materialized_scan(backend, [atom['identity'] for atom in visible],
                                               vectors.shape[1], top_k, pim_resident)
            pim_model_seconds = time.perf_counter() - pim_model_started
            pim_trace['scan_reference_exact'] = True
            pim_trace['placement_boundary'] = 'legacy separate-copy round-robin cost envelope; not timing for address-derived host functional scan'
            hybrid_started = time.perf_counter()
            cosine_scores = vectors @ query_vector
            dense_order = sorted(selected.tolist(), key=lambda index: (-float(cosine_scores[index]), index))
            retrieved, retrieval_meta = hybrid_candidates(
                visible, question['question'], dense_order, count=args.dense_event_count)
            hybrid_seconds = time.perf_counter() - hybrid_started
            active_stage = 'fp32_reranker'
            scores, scoring = reranker.score(question['question'], [atom_text(visible[index]) for index in retrieved])
            primary = sorted(range(len(retrieved)), key=lambda index: (-scores[index], index))[:args.candidate_count]
            bundles, trajectory_trace = trajectory_bundles(
                graph, visible, [retrieved[index] for index in primary], question['question'])
            materializable = sorted(set().union(*bundles))
            local_index = {index: local for local, index in enumerate(materializable)}
            local_bundles = [tuple(local_index[index] for index in bundle) for bundle in bundles]
            active_stage = 'union_search'
            plan = solve_materialization_plan(local_bundles, [visible[index]['cost'] for index in materializable],
                                              budget=args.budget_tokens, max_nodes=args.max_search_nodes,
                                              evidence_scores=[scores[index] for index in primary])
            plan['atom_identities'] = [visible[index]['identity'] for index in materializable]
            plan['atom_costs'] = [visible[index]['cost'] for index in materializable]
            chosen_indices = [materializable[index] for index in plan['materialized']]
            chosen_atoms = sorted((visible[index] for index in chosen_indices), key=lambda atom: atom['ordinal'])
            int8_trace['selected_entries'] = [int8_store.entries[atom['identity']] for atom in chosen_atoms]
            
            row.update(evidence_plan=plan,
                       trajectory_hypergraph=trajectory_trace,
                       selected_atoms=[{key: atom[key] for key in ('event_id', 'start', 'end', 'identity', 'timestamp')} for atom in chosen_atoms],
                       candidate_count=len(retrieved), visible_atoms=len(visible),
                       feature_materialization_seconds=materialization_seconds,
                       int8_primary=int8_trace,
                       feature_stage_seconds=materialization_seconds + query_encoding_seconds,
                       query_encoding_seconds=query_encoding_seconds,
                       query_planning_seconds=time.perf_counter() - query_started,
                       rerank_seconds=scoring['seconds'],
                       rerank_stage_seconds=scoring['seconds'],
                       reranker_state_reuse=scoring.get('state_reuse'),
                       union_search_seconds=plan['seconds'],
                       hybrid_candidate_seconds=hybrid_seconds,
                       retrieval_audit={'candidate_events': [visible[index]['event_id'] for index in retrieved],
                                        'primary_events': [visible[retrieved[index]]['event_id'] for index in primary],
                                        'dense_events': [visible[index]['event_id'] for index in retrieval_meta['dense_top_indices']],
                                        'reranker_scores': scores, 'reranker_input_sha256': scoring['inputs_sha256'],
                                        'candidate_source': retrieval_meta['candidate_source'],
                                        'candidate_granularity': retrieval_meta['candidate_granularity'],
                                        'dense_top_events': retrieval_meta['dense_top_events']},
                       pim_stage_seconds=pim_emulation_seconds + pim_model_seconds,
                       reference_validation_seconds=reference_validation_seconds,
                       feature_reuse={key: features.metrics()[key] - before[key] for key in ('hits', 'misses', 'evictions', 'encoded_tokens')},
                       feature_cache=features.metrics(),
                       pim_retrieval=pim_trace)
            active_stage = 'reader'
            reader_service_offset = time.perf_counter() - started
            if args.reader_protocol == 'single':
                request_args = argparse.Namespace(**{**vars(args), 'request_index': request_index})
                row.update(read_single_answer(model, processor, question, chosen_atoms, request_args, prefix_store,
                                               prefix_contract, SYSTEM, atom_text))
                row['prefix_state_reuse']['reuse_accounting'] = account_prefix_reuse(
                    row['prefix_state_reuse'], row['prefix_validation'], row['execution_stages']).as_dict()
                row['pim_reader_residual'] = replay_reader_residual(backend, model.config.text_config, row)
            else:
                row.update(read_answer(model, processor, question, chosen_atoms, args))
            row['reader_stage_seconds'] = row['reader_seconds']
            row['snapshot_seconds'] = row['prefix_state_reuse'].get('snapshot_seconds', 0.0)
            row['restore_seconds'] = row['prefix_state_reuse'].get('restore_seconds', 0.0)
            served_stage = 'first' if args.reader_protocol == 'single' else 'proof'
            prompt_tokens = row['execution_stages'][served_stage]['prompt_tokens']
            reused_tokens = row['prefix_state_reuse'].get('reused_tokens', 0)
            row['prefix_state_reuse']['reuse_ratio'] = reused_tokens / max(1, prompt_tokens)
            row['prefix_state_reuse']['residual_tokens'] = prompt_tokens - reused_tokens
            row['first_answer_ttft_service_seconds'] = reader_service_offset + row['execution_stages']['first']['ttft_from_reader_start_seconds']
            row['final_answer_ttft_service_seconds'] = reader_service_offset + row['execution_stages'][served_stage]['ttft_from_reader_start_seconds']
            row['status'] = 'ok'
        except Exception as error:
            row.update(status='error', failure_stage=active_stage, error_type=type(error).__name__, detail=str(error))
            torch.cuda.empty_cache()
        torch.cuda.synchronize(args.device)
        row['reader_service_seconds'] = time.perf_counter() - started
        row['service_excluding_numeric_validation_seconds'] = row['reader_service_seconds'] - row.get('prefix_validation', {}).get('overhead_seconds', 0.0)
        row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
        rows.append(row)
        (args.output / 'shard_0.json').write_text(json.dumps({'complete': False, 'configuration': configuration,
                    'provenance': provenance, 'rows': rows}, ensure_ascii=False))
        print(json.dumps({'question_id': row['question_id'], 'status': row['status'],
                          'service_seconds': row['reader_service_seconds'], 'detail': row.get('detail'),
                          'reused_tokens': row.get('prefix_state_reuse', {}).get('reused_tokens')}, ensure_ascii=False), flush=True)
    changed = [name for name, expected in provenance['code_sha256'].items()
               if file_hash((ROOT / 'pim_backend' if name in {'lpddr_bank_pim.py', 'agent_memory_pim.py'} else ROOT / 'scripts') / name) != expected]
    if changed:
        raise ValueError(f'code changed during frozen run: {changed}')
    export_started = time.perf_counter()
    primary_snapshot = int8_store.save(args.output / 'shared_int8_primary.bin')
    primary_snapshot['diagnostic_export_seconds'] = time.perf_counter() - export_started
    primary_snapshot['boundary'] = 'post-service diagnostic export, not producer or PIM residency cost'
    payload = {'complete': True, 'configuration': configuration, 'provenance': provenance, 'rows': rows,
               'selected_question_ids': registration['selected_question_ids'],
               'quality_measurement': None, 'official_success_rate': None,
               'boundary': registration['boundary'], 'int8_primary_snapshot': primary_snapshot}
    (args.output / 'shard_0.json').write_text(json.dumps(payload, ensure_ascii=False))
    return int(any(row['status'] != 'ok' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
