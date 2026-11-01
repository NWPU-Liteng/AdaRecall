"""Replay frozen pilot retrieval scans through a functional INT8 PIM path and timing model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from pim_backend.lpddr_bank_pim import BankPimConfig, BankPimModel, bank_int8_topk


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def quantize_symmetric_int8(values: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(values, dtype=np.float32)
    if source.size == 0 or not np.isfinite(source).all():
        raise ValueError('finite nonempty values required for INT8 quantization')
    maximum = float(np.max(np.abs(source)))
    scale = maximum / 127.0 if maximum else 1.0
    quantized = np.rint(source / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def topk_overlap(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_ids = set(map(int, reference))
    candidate_ids = set(map(int, candidate))
    return len(reference_ids & candidate_ids) / max(1, len(reference_ids))


def load_requests(run_root: Path, cohort_path: Path) -> tuple[dict[str, list[dict]], dict]:
    cohort = json.loads(cohort_path.read_text())
    if cohort.get('schema') != 'mobilemem-omni-frozen-pilot-cohort-v1':
        raise ValueError('frozen label-free pilot cohort required')
    expected = cohort.get('question_ids_by_user', {})
    if sum(map(len, expected.values())) != cohort.get('questions'):
        raise ValueError('frozen pilot denominator mismatch')
    by_user = {}
    for user_id, identities in expected.items():
        path = run_root / f'user_{user_id}_shard_0.json'
        payload = json.loads(path.read_text())
        if not payload.get('complete') or payload.get('person_id') != user_id:
            raise ValueError(f'complete matching reader run required: {path}')
        rows = payload.get('rows', [])
        if [row.get('question_id') for row in rows] != identities:
            raise ValueError(f'reader output differs from frozen cohort: {path}')
        by_user[user_id] = rows
    return by_user, cohort


def encode_queries(texts: list[str], tokenizer, model, device: str, batch_size: int) -> np.ndarray:
    import torch

    encoded_rows = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], padding=True,
                            truncation=False, return_tensors='pt').to(device)
        if encoded.input_ids.shape[1] > 2048:
            raise ValueError('retrieval view exceeds the frozen embedder context')
        with torch.inference_mode():
            output = model(**encoded, use_cache=False).last_hidden_state
        positions = (encoded.attention_mask.shape[1] - 1
                     - encoded.attention_mask.flip(dims=[1]).argmax(dim=1))
        batch = torch.arange(output.shape[0], device=output.device)
        vectors = torch.nn.functional.normalize(output[batch, positions].float(), dim=-1)
        encoded_rows.append(vectors.cpu().numpy())
    return np.concatenate(encoded_rows, axis=0)


def profile(args: argparse.Namespace) -> dict:
    if args.output.exists() or args.pim_index_root.exists():
        raise ValueError('fresh PIM profile output and quantized-index directory required')
    if args.top_k < 1 or args.cpu_repeats < 1 or args.batch_size < 1:
        raise ValueError('top-k, CPU repeats and batch size must be positive')
    by_user, cohort = load_requests(args.run_root, args.cohort)
    args.pim_index_root.mkdir(parents=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(str(args.embedder), local_files_only=True,
                                              padding_side='left')
    embedder = AutoModel.from_pretrained(str(args.embedder), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()

    config = BankPimConfig()
    model = BankPimModel(config)
    detail_rows = []
    per_user = {}
    total_pim_resident_bytes = 0
    total_index_load_cycles = 0
    total_host_model_cycles = 0
    total_pim_model_cycles = 0
    overlap_numerator = overlap_denominator = 0
    cpu_scan_seconds = []
    quantized_kernel_seconds = []
    query_quantization_seconds = []

    for user_id, rows in by_user.items():
        index_dir = args.index_root / f'proof_index_user_{user_id}'
        meta_path = index_dir / 'index.json'
        meta = json.loads(meta_path.read_text())
        vectors_path = index_dir / 'vectors.npy'
        if (not meta.get('complete') or str(meta.get('person_id')) != user_id
                or digest(vectors_path) != meta.get('vectors_sha256')):
            raise ValueError(f'index identity or vector hash mismatch for user {user_id}')
        vectors = np.load(vectors_path, mmap_mode='r')
        if vectors.ndim != 2 or len(vectors) != int(meta['records']):
            raise ValueError(f'vector/index shape mismatch for user {user_id}')
        if vectors.dtype != np.float32:
            raise ValueError(f'FP32 host reference expected, got {vectors.dtype} for user {user_id}')

        queries = []
        for row in rows:
            views = row.get('proof_plan', {}).get('retrieval', {}).get('views')
            if not isinstance(views, list) or not views:
                raise ValueError(f'missing original retrieval views: {row["question_id"]}')
            if len(views) != row['proof_plan']['retrieval'].get('demand_count'):
                raise ValueError(f'retrieval view/demand mismatch: {row["question_id"]}')
            for demand_index, text in enumerate(views):
                queries.append((row['question_id'], demand_index, text))
        query_vectors = encode_queries([entry[2] for entry in queries], tokenizer,
                                       embedder, args.device, args.batch_size)
        if query_vectors.shape[1] != vectors.shape[1]:
            raise ValueError('query and resident embedding dimensions differ')

        quantize_started = time.perf_counter()
        resident_int8, resident_scale = quantize_symmetric_int8(vectors)
        quantize_seconds = time.perf_counter() - quantize_started
        user_pim_dir = args.pim_index_root / f'user_{user_id}'
        user_pim_dir.mkdir()
        quantized_path = user_pim_dir / 'vectors_int8.npy'
        with quantized_path.open('xb') as handle:
            np.save(handle, resident_int8, allow_pickle=False)
        index_manifest = {
            'schema': 'mobilemem-omni-pim-resident-int8-v1',
            'person_id': user_id,
            'source_index_sha256': digest(meta_path),
            'source_vectors_sha256': digest(vectors_path),
            'vectors_int8_sha256': digest(quantized_path),
            'vector_count': int(vectors.shape[0]),
            'dimension': int(vectors.shape[1]),
            'operand': 'symmetric global-scale INT8; round-to-nearest; clip [-127,127]',
            'resident_scale': resident_scale,
            'resident_bytes': int(resident_int8.nbytes),
            'quantization_wall_seconds': quantize_seconds,
            'evaluation_labels_read': False,
        }
        with (user_pim_dir / 'index.json').open('x') as handle:
            json.dump(index_manifest, handle, ensure_ascii=False, indent=2)

        vector_count, dimension = vectors.shape
        top_k = min(args.top_k, vector_count)
        host_model = model.resident_score_scan(vector_count, dimension, top_k=top_k,
                                               route='host', element_bytes=4)
        pim_model = model.resident_score_scan(vector_count, dimension, top_k=top_k,
                                              route='pim', element_bytes=1)
        if not pim_model['admission']['admitted'] or pim_model['route'] != 'pim':
            raise ValueError(f'INT8 PIM index failed configured capacity for user {user_id}')
        index_load = model.copy_cost(int(resident_int8.nbytes))
        user_overlap_num = user_overlap_den = 0
        user_cpu_seconds, user_kernel_seconds = [], []

        for query_index, (question_id, demand_index, text) in enumerate(queries):
            query = query_vectors[query_index]
            cpu_elapsed = []
            cpu_top = None
            for _ in range(args.cpu_repeats):
                started = time.perf_counter()
                scores = vectors @ query
                cpu_top = np.argsort(-scores)[:top_k]
                cpu_elapsed.append(time.perf_counter() - started)
            cpu_seconds = statistics.median(cpu_elapsed)

            query_started = time.perf_counter()
            query_int8, query_scale = quantize_symmetric_int8(query)
            query_quant_seconds = time.perf_counter() - query_started
            kernel_started = time.perf_counter()
            pim_top, _ = bank_int8_topk(resident_int8, query_int8,
                                        banks=config.bank_count, top_k=top_k)
            kernel_seconds = time.perf_counter() - kernel_started
            overlap = topk_overlap(cpu_top, pim_top)
            overlap_count = len(set(map(int, cpu_top)) & set(map(int, pim_top)))
            user_overlap_num += overlap_count
            user_overlap_den += len(cpu_top)
            overlap_numerator += overlap_count
            overlap_denominator += len(cpu_top)
            user_cpu_seconds.append(cpu_seconds)
            user_kernel_seconds.append(kernel_seconds)
            cpu_scan_seconds.append(cpu_seconds)
            quantized_kernel_seconds.append(kernel_seconds)
            query_quantization_seconds.append(query_quant_seconds)
            detail_rows.append({
                'person_id': user_id,
                'question_id': question_id,
                'demand_index': demand_index,
                'retrieval_view_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'vector_count': int(vector_count),
                'dimension': int(dimension),
                'top_k': top_k,
                'fp32_host_score_and_topk_seconds_median': cpu_seconds,
                'int8_functional_emulator_seconds': kernel_seconds,
                'query_int8_quantization_seconds': query_quant_seconds,
                'query_scale': query_scale,
                'topk_overlap': overlap,
                'host_model_cycles_fp32': host_model['cycles'],
                'pim_model_cycles_int8': pim_model['cycles'],
                'pim_model_refresh_cycles': pim_model['refresh_cycles'],
                'pim_model_external_bytes': pim_model['external_bytes'],
                'pim_model_bank_internal_bytes': pim_model['bank_internal_bytes'],
                'pim_model_row_activations': pim_model['row_activations'],
                'pim_model_host_commands': pim_model['host_commands'],
            })

        query_count = len(queries)
        user_resident_bytes = int(resident_int8.nbytes)
        user_pim_bytes = user_resident_bytes
        user_index_load_cycles = int(index_load['cycles'])
        total_pim_resident_bytes += user_pim_bytes
        total_index_load_cycles += user_index_load_cycles
        total_host_model_cycles += host_model['cycles'] * query_count
        total_pim_model_cycles += pim_model['cycles'] * query_count
        per_user[user_id] = {
            'requests': len(rows),
            'query_vectors': query_count,
            'vector_count': int(vector_count),
            'dimension': int(dimension),
            'host_fp32_index_bytes': int(vectors.nbytes),
            'pim_int8_index_bytes': user_resident_bytes,
            'index_quantization_wall_seconds': quantize_seconds,
            'pim_index_load_model_cycles_once': user_index_load_cycles,
            'host_model_cycles_warm': host_model['cycles'] * query_count,
            'pim_model_cycles_warm': pim_model['cycles'] * query_count,
            'pim_topk_overlap': user_overlap_num / user_overlap_den if user_overlap_den else None,
            'cpu_scan_seconds_median': statistics.median(user_cpu_seconds),
            'int8_functional_emulator_seconds_median': statistics.median(user_kernel_seconds),
            'capacity_admission': pim_model['admission'],
        }
        del resident_int8, query_vectors

    demand_count = len(detail_rows)
    warm_ratio = total_host_model_cycles / total_pim_model_cycles if total_pim_model_cycles else None
    cold_pim_cycles = total_pim_model_cycles + total_index_load_cycles
    cold_ratio = total_host_model_cycles / cold_pim_cycles if cold_pim_cycles else None
    clock_hz = config.clock_hz
    host_model_seconds = total_host_model_cycles / clock_hz
    pim_warm_model_seconds = total_pim_model_cycles / clock_hz
    pim_cold_model_seconds = cold_pim_cycles / clock_hz
    payload = {
        'schema': 'mobilemem-omni-main-pim-profile-v1',
        'cohort_sha256': digest(args.cohort),
        'reader_run_sha256': {user: digest(args.run_root / f'user_{user}_shard_0.json')
                              for user in by_user},
        'index_source_sha256': {user: digest(args.index_root / f'proof_index_user_{user}' / 'index.json')
                                for user in by_user},
        'model': 'fixed uncalibrated LPDDR5-6400 bank timing model',
        'model_profile': str(ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json'),
        'model_profile_sha256': digest(ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json'),
        'model_config': asdict(config),
        'operation': 'resident vector dot-product scan + bank-local Top-K + host merge',
        'numeric_paths': {
            'host_reference': 'original FP32 index vectors and frozen embedder query vectors; exact reference for the replay',
            'pim_functional': 'global-scale symmetric INT8 vectors and per-query symmetric INT8; int32 dot accumulation in software bank emulator',
            'numerical_rank_validation': 'PIM returned Top-K set compared with original FP32 Host Top-K; reranker and answer are not replayed',
        },
        'reader_runs': len(by_user),
        'requests': sum(map(len, by_user.values())),
        'query_vector_scans': demand_count,
        'top_k': args.top_k,
        'pim_index_total_bytes': total_pim_resident_bytes,
        'pim_pool_capacity_bytes': config.pim_capacity_bytes,
        'pim_all_user_indices_admitted': total_pim_resident_bytes <= config.pim_capacity_bytes,
        'host_fp32_modeled_warm_cycles': total_host_model_cycles,
        'pim_int8_modeled_warm_cycles': total_pim_model_cycles,
        'pim_int8_modeled_cold_cycles_including_one_index_load_per_user': cold_pim_cycles,
        'host_fp32_modeled_warm_seconds': host_model_seconds,
        'pim_int8_modeled_warm_seconds': pim_warm_model_seconds,
        'pim_int8_modeled_cold_seconds_including_index_load': pim_cold_model_seconds,
        'modeled_warm_scan_speedup': warm_ratio,
        'modeled_cold_scan_speedup': cold_ratio,
        'fp32_host_scan_wall_seconds_median': statistics.median(cpu_scan_seconds),
        'fp32_host_scan_wall_seconds_sum_of_medians': sum(cpu_scan_seconds),
        'int8_software_emulator_wall_seconds_median': statistics.median(quantized_kernel_seconds),
        'int8_query_quantization_wall_seconds_median': statistics.median(query_quantization_seconds),
        'mean_topk_overlap_vs_fp32_host': overlap_numerator / overlap_denominator if overlap_denominator else None,
        'p95_topk_overlap_vs_fp32_host': float(np.quantile(
            [row['topk_overlap'] for row in detail_rows], 0.95)) if detail_rows else None,
        'per_user': per_user,
        'rows': detail_rows,
        'claim_boundary': 'modeled retrieval scan only; fixed uncalibrated profile; software INT8 function validates ranking approximation; no AttAcc LPDDR-PIM instruction execution, no shared Reader contention, no reranker/answer replay, and no phone/end-to-end speedup claim',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    report = [
        '# 200 题主实验：LPDDR-PIM 向量检索模拟', '',
        f"- 重放 {payload['requests']} 个请求、{demand_count} 个实际 query-vector 扫描；PIM 向量索引 {total_pim_resident_bytes / 1024**2:.2f} MiB，模型容量准入：{payload['pim_all_user_indices_admitted']}。",
        f"- INT8 PIM Top-{args.top_k} 与原始 FP32 Host 候选平均重合率：{payload['mean_topk_overlap_vs_fp32_host']:.2%}。该项不等价于 Judge 成功率保持。",
        f"- 固定 LPDDR5 模型的 warm 扫描估算：Host {host_model_seconds:.6f}s，PIM {pim_warm_model_seconds:.6f}s，模拟扫描比值 {warm_ratio:.2f}x。",
        f"- 加入每用户一次索引装载后，PIM 冷启动模型时间 {pim_cold_model_seconds:.6f}s，模拟比值 {cold_ratio:.2f}x。",
        f"- 实测服务器 FP32 矩阵扫描+Top-K 中位数：{statistics.median(cpu_scan_seconds) * 1000:.3f} ms/查询向量；INT8 NumPy 软件 emulator 中位数：{statistics.median(quantized_kernel_seconds) * 1000:.3f} ms/查询向量（这是软件模拟开销，不是 PIM 硬件时延）。",
        f"- {warm_ratio:.2f}x warm / {cold_ratio:.2f}x cold 是固定参数下的模型比值；模型 Host 为 {host_model_seconds / demand_count * 1000:.3f} ms/查询，服务器 CPU 实测中位数为 {statistics.median(cpu_scan_seconds) * 1000:.3f} ms/查询，二者未校准一致，不能把模型比值当作实际性能预测。",
        '', '以上只覆盖检索向量扫描，不覆盖 parser、reranker、超图见证、Reader 或端到端请求；LPDDR5 参数未做器件校准，不能作为手机实测或端到端加速结论。',
    ]
    report_path = args.output.with_suffix('.md')
    report_path.write_text('\n'.join(report) + '\n')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--index-root', type=Path, required=True)
    parser.add_argument('--embedder', type=Path, required=True)
    parser.add_argument('--pim-index-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--top-k', type=int, default=48)
    parser.add_argument('--cpu-repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    payload = profile(args)
    print(json.dumps({key: payload[key] for key in (
        'requests', 'query_vector_scans', 'pim_index_total_bytes',
        'mean_topk_overlap_vs_fp32_host', 'host_fp32_modeled_warm_seconds',
        'pim_int8_modeled_warm_seconds', 'pim_int8_modeled_cold_seconds_including_index_load',
        'modeled_warm_scan_speedup', 'modeled_cold_scan_speedup',
        'fp32_host_scan_wall_seconds_median', 'claim_boundary')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
