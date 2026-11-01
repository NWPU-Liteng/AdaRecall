"""Two-query, label-free raw-image Reader smoke for MobileMem Omni."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.decode_gemv import bank_tiled_gemv
from pim_backend.functional_decode import (DeferredInt8Validation, Int8LinearContract,
    materialize_int8_primary, materialize_w8a16_primary,
    materialize_w12a16_primary)
from pim_backend.pimnast_functional import PimnastReaderDispatch
from pim_backend.w12_functional_dispatch import (W12FunctionalBankLinear,
                                                 W12FunctionalDispatch)
from omni_evidence_planner import bound_hyperedges, build_local_hyperedges, select_hyperedges
from omni_atom_token_cost import AtomUnionTokenCosts
from omni_embedding_store import EdgeEmbeddingStore
from omni_embedding_producer import resolve_producers
from omni_patch_projection import configure_patch_projection
from omni_service_measurement import measure_vision, service_stage_ledger
from omni_visual_feature_store import VisualFeatureStore
from omni_image_selection import select_dialogue_images
from omni_provenance_optimizer import (FrozenRelevanceModel, build_provenance_hyperedges,
                                        optimize_provenance)
from omni_evidence_planner import images_for_atoms
from omni_multimodal_relevance import FrozenMultimodalRelevance
from omni_joint_index import JointEncoder, JointIndex, TrainedMultimodalReranker, model_files
from omni_candidate_union import candidate_union
from omni_evidence_compiler import compile_evidence
DEFAULT_READER = Path("/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B")
DEFAULT_EMBEDDER = Path("/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B")
SYSTEM = (
    "你是端侧长期多模态记忆助手。只依据提供的、在本次查询协议中可见的原始对话和图片回答。"
    "区分用户陈述与助手建议；没有充分证据时明确说明证据不足。回答简洁，不要展示推理过程。"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_vector(texts, tokenizer, model, device):
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors="pt").to(device)
    if encoded.input_ids.shape[1] > 2048:
        raise ValueError("session/query embedding input exceeds 2048 tokens")
    with torch.inference_mode():
        output = model(**encoded, use_cache=False).last_hidden_state
    positions = encoded.attention_mask.shape[1] - 1 - encoded.attention_mask.flip(dims=[1]).argmax(dim=1)
    rows = torch.arange(output.shape[0], device=output.device)
    vector = output[rows, positions].float()
    return torch.nn.functional.normalize(vector, dim=-1)


def image_members(image_archive: Path) -> set[str]:
    with zipfile.ZipFile(image_archive) as archive:
        return {name for name in archive.namelist()
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))}


def load_image(image_archive: Path, ref: dict, members: set[str]) -> tuple[Image.Image, str]:
    if ref["status"] == "exact":
        member = ref["normalized"]
    elif ref["status"] == "unique_user_basename":
        member = ref["resolved"]
    else:
        raise ValueError(f"non-resolved visible-history image: {ref}")
    if member not in members:
        raise ValueError(f"image is not in the pinned archive manifest: {member}")
    with zipfile.ZipFile(image_archive) as archive:
        image_bytes = archive.read(member)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), sha256_bytes(image_bytes)


def pack_session_turns(turns, tokenizer, budget):
    if budget <= 0:
        raise ValueError("positive full-turn evidence budget required")
    selected, accepted, rejected = [], [], []
    for atom in turns:
        if atom.get("content_type") != "text" or not atom.get("content"):
            continue
        rendered = f"[{atom['role']}][text] {atom['content']}"
        candidate = "\n".join(selected + [rendered])
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
            selected.append(rendered)
            accepted.append(atom["atom_id"])
        else:
            rejected.append(atom["atom_id"])
    return "\n".join(selected), accepted, rejected


def processor_image_token_costs(reader_processor, reader, refs, image_archive, members):
    costs = {}
    unique = {}
    for ref in refs:
        identity = ref.get('resolved', ref['normalized'])
        unique.setdefault(identity, ref)
    for identity, ref in unique.items():
        image, _ = load_image(image_archive, ref, members)
        encoded = reader_processor.image_processor(images=[image], return_tensors='pt')
        grid = encoded.get('image_grid_thw')
        if grid is None:
            raise ValueError('Qwen processor did not expose image_grid_thw')
        merge = int(reader.model.visual.spatial_merge_size)
        patch_count = int(grid.detach().cpu().prod(dim=-1).sum().item())
        if patch_count <= 0 or patch_count % (merge * merge):
            raise ValueError('image grid is incompatible with Qwen spatial merge')
        costs[identity] = patch_count // (merge * merge)
    return costs


class LinearTrace:
    def __init__(self, model, validate_bank_gemv=False):
        self.records = {}
        self.functional_samples = {}
        self.validate_bank_gemv = validate_bank_gemv
        self.phase = "prefill"
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, Int8LinearContract,
                                   W12FunctionalBankLinear)):
                self.handles.append(module.register_forward_hook(self.make_hook(name)))

    def make_hook(self, name):
        def hook(module, inputs, output):
            if not inputs:
                return
            rows = int(inputs[0].numel() // inputs[0].shape[-1])
            weight_bytes = (module.weight_bytes
                            if isinstance(module, W12FunctionalBankLinear)
                            else module.weight.numel() * module.weight.element_size())
            element_bytes = (module.element_bytes
                             if isinstance(module, W12FunctionalBankLinear)
                             else module.weight.element_size())
            key = (self.phase, name, rows)
            item = self.records.setdefault(key, {
                "phase": self.phase, "name": name, "rows": rows,
                "in_features": module.in_features, "out_features": module.out_features,
                "weight_bytes": weight_bytes,
                "element_bytes": element_bytes, "calls": 0,
            })
            item["calls"] += 1
            targets = {"model.language_model.layers.0.mlp.gate_proj",
                       "model.language_model.layers.0.mlp.down_proj", "lm_head"}
            if (self.validate_bank_gemv and self.phase == "decode" and rows == 1
                    and name in targets and name not in self.functional_samples):
                import numpy as np
                sample_rows = min(128, module.out_features)
                weights = module.weight[:sample_rows].detach().float().cpu().numpy()
                activation = inputs[0].detach().reshape(-1).float().cpu().numpy()
                bias = module.bias[:sample_rows].detach().float().cpu().numpy() if module.bias is not None else None
                bank_output = bank_tiled_gemv(weights, activation, bias=bias, operand_dtype="bf16")
                reference = weights @ activation
                if bias is not None:
                    reference += bias
                np.testing.assert_allclose(bank_output, reference, rtol=1e-4, atol=3e-4)
                bf16_output = output.detach().reshape(-1)[:sample_rows].float().cpu().numpy()
                self.functional_samples[name] = {
                    "rows": sample_rows, "inputs": module.in_features,
                    "activation_sha256": sha256_bytes(activation.tobytes()),
                    "weight_slice_sha256": sha256_bytes(weights.tobytes()),
                    "bf16_operands_fp32_reference_max_abs_error": float(np.max(np.abs(bank_output - reference))),
                    "bf16_gpu_max_abs_difference": float(np.max(np.abs(bank_output - bf16_output))),
                    "allclose": True, "rtol": 1e-4, "atol": 3e-4,
                    "boundary": "actual decode BF16 activation/weights, first 128 output rows only; custom FP32 accumulator assumption",
                }
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self):
        return {"operators": list(self.records.values()),
                "functional_samples": self.functional_samples,
                "coverage": "executed nn.Linear calls only; attention/recurrent/vision kernels may be omitted"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--image-archive", type=Path, required=True)
    parser.add_argument("--reader", type=Path, default=DEFAULT_READER)
    parser.add_argument("--embedder", type=Path, default=DEFAULT_EMBEDDER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-id", default="0")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--allowed-splits", nargs='+',
                        choices=('development', 'validation', 'holdout'),
                        default=['development'])
    parser.add_argument("--screening-run", action='store_true')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-sessions", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--session-text-tokens", type=int, default=900)
    parser.add_argument("--evidence-policy", choices=("chronological", "local_hyperedge", "provenance_milp", "evidence_compiler"), default="chronological")
    parser.add_argument("--reranker", type=Path, default=Path("/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0___6B"))
    parser.add_argument('--relevance-modality', choices=('text', 'multimodal', 'joint_index'), default='text')
    parser.add_argument('--multimodal-reranker', type=Path, default=Path(
        '/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct'))
    parser.add_argument('--relevance-batch-size', type=int, default=1)
    parser.add_argument('--joint-index', type=Path)
    parser.add_argument('--joint-recall-count', type=int, default=16)
    parser.add_argument('--compiler-retrieval', choices=('exchange', 'evidence_units'), default='exchange')
    parser.add_argument('--joint-text-recall-count', type=int, default=16)
    parser.add_argument('--joint-embedding-model', type=Path, default=ROOT.parents[1] / 'models/Qwen3-VL-Embedding-2B')
    parser.add_argument('--joint-reranker-model', type=Path, default=ROOT.parents[1] / 'models/Qwen3-VL-Reranker-2B')
    parser.add_argument('--joint-index-image-max-pixels', type=int, default=65536)
    parser.add_argument('--profile-reranker', action='store_true')
    parser.add_argument('--joint-patch-projection', choices=('conv', 'linear'), default='conv')
    parser.add_argument('--reader-patch-projection', choices=('conv', 'linear'), default='conv')
    parser.add_argument('--reader-quantization', choices=('bf16', 'w8a8', 'w8a16', 'w12a16'), default='bf16')
    parser.add_argument('--w8a16-group-size', type=int, choices=(8, 16, 32, 64, 128, 256), default=128)
    parser.add_argument('--w8a16-residual-bits', type=int, choices=(0, 2, 4), default=0)
    parser.add_argument('--w8a16-fallback-scope', choices=('none', 'lm_head', 'output_sensitive'), default='none')
    parser.add_argument('--w12a16-group-size', type=int, choices=(8, 16, 32, 64, 128, 256), default=128)
    parser.add_argument('--w12-functional-route', choices=('none', 'shadow', 'replace'), default='none')
    parser.add_argument('--w12-numeric-mode', choices=('fp32_bank', 'bf16_reference'), default='fp32_bank')
    parser.add_argument('--w12-bank-count', type=int, default=256)
    parser.add_argument('--w12-tile-rows', type=int, default=16)
    parser.add_argument('--w12-release-source', action='store_true')
    parser.add_argument('--w12-allow-decode-fallback', action='store_true')
    parser.add_argument('--defer-int8-finite-checks', action='store_true')
    parser.add_argument('--pimnast-functional-projection', type=Path)
    parser.add_argument("--planner-time-limit", type=float, default=5.0)
    parser.add_argument("--input-budget", type=int, default=8192)
    parser.add_argument("--total-text-tokens", type=int, default=3600)
    parser.add_argument("--edge-cache-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--disable-edge-reuse", action="store_true")
    parser.add_argument("--visual-cache-bytes", type=int, default=0)
    parser.add_argument("--max-hyperedges", type=int, default=64)
    parser.add_argument("--image-edges-per-session", type=int, default=0)
    parser.add_argument('--query-local-token-cost-reuse', action='store_true')
    parser.add_argument("--image-novelty-weight", type=float, default=0.0)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--verify-generation", action="store_true")
    parser.add_argument("--diagnostic-input-dump-dir", type=Path,
                        help="explicit diagnostic-only CPU dump of exact Reader inputs")
    parser.add_argument("--validate-bank-gemv", action="store_true")
    parser.add_argument("--cohort-registration-sha256", default="")
    parser.add_argument("--hardware-identity", default="")
    parser.add_argument("--concurrency-contract", default="")
    parser.add_argument("--startup-policy", default="model_load_outside_request_per_worker")
    args = parser.parse_args()
    if args.visual_cache_bytes < 0 or (args.visual_cache_bytes and not args.verify_generation):
        raise SystemExit('visual cache validation requires nonnegative capacity and --verify-generation')
    if args.reader_quantization == 'w8a8' and args.validate_bank_gemv:
        raise SystemExit('BF16 bank-GEMV probe cannot validate integer primary arithmetic')
    if args.reader_quantization != 'w8a16' and args.w8a16_group_size != 128:
        raise SystemExit('W8A16 group size only applies to the W8A16 arm')
    if args.reader_quantization != 'w8a16' and args.w8a16_fallback_scope != 'none':
        raise SystemExit('W8A16 fallback only applies to the W8A16 arm')
    if args.reader_quantization != 'w8a16' and args.w8a16_residual_bits != 0:
        raise SystemExit('W8A16 residual correction only applies to the W8A16 arm')
    if args.reader_quantization != 'w12a16' and args.w12a16_group_size != 128:
        raise SystemExit('W12A16 group size only applies to the W12A16 arm')
    if args.w12_functional_route != 'none' and args.reader_quantization != 'w12a16':
        raise SystemExit('W12 functional bank route requires the W12A16 arm')
    if (args.w12_functional_route != 'none'
            and (args.w12_bank_count <= 0 or args.w12_tile_rows <= 0)):
        raise SystemExit('positive W12 bank count and row tile required')
    if (args.w12_functional_route == 'none'
            and (args.w12_release_source or args.w12_allow_decode_fallback)):
        raise SystemExit('W12 bank options require a functional route')
    if (args.w12_functional_route != 'none'
            and (args.pimnast_functional_projection is not None
                 or args.defer_int8_finite_checks)):
        raise SystemExit('W12 bank route cannot be combined with the W8 PIM functional route')
    if args.pimnast_functional_projection and args.reader_quantization != 'w8a8':
        raise SystemExit('PIMnast integer functional backend requires shared W8A8 primary')
    if args.defer_int8_finite_checks and args.reader_quantization != 'w8a8':
        raise SystemExit('deferred finite-check aggregation requires shared W8A8 primary')
    if (args.output.exists() or args.limit < 1 or args.max_images < 0
            or args.max_hyperedges < 1 or args.image_edges_per_session < 0
            or not math.isfinite(args.image_novelty_weight) or args.image_novelty_weight < 0
            or not math.isfinite(args.planner_time_limit) or args.planner_time_limit <= 0
            or args.input_budget <= 0 or args.relevance_batch_size < 1
            or args.joint_recall_count < 1 or args.joint_text_recall_count < 1
            or args.joint_index_image_max_pixels < 1
            or (args.relevance_modality != 'text' and args.evidence_policy not in {'provenance_milp', 'evidence_compiler'})
            or (args.evidence_policy == 'evidence_compiler' and args.relevance_modality != 'joint_index')
            or (args.relevance_modality == 'joint_index' and args.joint_index is None)):
        raise SystemExit("fresh output, positive development request count, and raw-image input are required")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    for name in ("omni_graph.json", "manifests.jsonl", "split.json", "audit.json"):
        if not (args.prepared / name).is_file():
            raise ValueError(f"missing prepared input: {name}")
    audit = json.loads((args.prepared / "audit.json").read_text())
    for name in ('omni_graph.json', 'manifests.jsonl', 'split.json'):
        if sha256_file(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('serving source hash changed')
    if audit.get("quality_evaluated") or audit.get("status") not in {
        "pass", "qualified_with_reference_gaps"
    }:
        raise ValueError("prepared data did not pass schema/source audit")
    serving_names = set(audit["serving_files"])
    forbidden_names = set(audit["evaluation_only_files"])
    if serving_names & forbidden_names:
        raise ValueError("serving/evaluation input manifests overlap")
    manifests = [json.loads(line) for line in (args.prepared / "manifests.jsonl").read_text().splitlines() if line]
    allowed_splits = set(args.allowed_splits)
    available = [item for item in manifests if item['person_id'] == str(args.user_id)
                 and item['split'] in allowed_splits]
    if args.cohort is not None:
        cohort = json.loads(args.cohort.read_text())
        if cohort['prepared_audit_sha256'] != sha256_file(args.prepared / 'audit.json'):
            raise ValueError('cohort was registered on another prepared release')
        if set(cohort.get('serving_splits') or ['development']) != allowed_splits:
            raise ValueError('serving splits differ from registered cohort')
        identities = cohort['question_ids_by_user'][str(args.user_id)]
        if len(identities) != args.limit or len(set(identities)) != len(identities):
            raise ValueError('cohort count or unique identities mismatch')
        selected = [item for item in available if item['question_id'] in set(identities)]
        if [item['question_id'] for item in selected] != identities:
            raise ValueError('cohort order differs from registered original question ordering')
    else:
        selected = available[:args.limit]
    if len(selected) != args.limit or any(item["visibility_protocol"] != "history_end" for item in selected):
        raise ValueError("frozen history-end questions from the registered split are required")
    graphs = json.loads((args.prepared / "omni_graph.json").read_text())
    graph = next((item for item in graphs if item["person_id"] == str(args.user_id)), None)
    if graph is None:
        raise ValueError("registered user's observable history is absent")
    sessions = graph["sessions"]
    source_sha256 = sha256_file(Path(__file__))
    source_files_sha256 = {name: sha256_file(ROOT / name) for name in (
        'scripts/run_omni_raw_image_smoke.py', 'scripts/omni_evidence_planner.py',
        'scripts/omni_embedding_store.py', 'scripts/omni_embedding_producer.py',
        'scripts/omni_atom_token_cost.py',
        'scripts/omni_service_measurement.py',
        'scripts/omni_patch_projection.py',
        'scripts/omni_visual_feature_store.py',
        'scripts/omni_image_selection.py',
        'pim_backend/decode_gemv.py', 'pim_backend/lpddr_bank_pim.py', 'pim_backend/functional_decode.py',
        'pim_backend/pimnast_functional.py', 'pim_backend/w12_bank_layout.py',
        'pim_backend/w12_functional_dispatch.py',
        'scripts/omni_provenance_optimizer.py', 'scripts/omni_multimodal_relevance.py',
        'scripts/omni_joint_index.py', 'scripts/omni_candidate_union.py',
        'scripts/omni_evidence_compiler.py', 'scripts/omni_evidence_views.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_snapshot = args.output.with_suffix('.sources')
    source_snapshot.mkdir(exist_ok=False)
    for name, expected_hash in source_files_sha256.items():
        snapshot_path = source_snapshot / name
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes((ROOT / name).read_bytes())
        if sha256_file(snapshot_path) != expected_hash:
            raise ValueError('source changed while freezing execution snapshot')
    registration = args.output.with_suffix('.registration.json')
    with registration.open('x') as handle:
        json.dump({'question_ids': [item['question_id'] for item in selected],
                   'arguments': {name: str(value) if isinstance(value, Path) else value
                                 for name, value in vars(args).items()},
                   'script_sha256': source_sha256,
                   'source_files_sha256': source_files_sha256,
                   'source_snapshot_directory': str(source_snapshot),
                   'prepared_audit_sha256': sha256_file(args.prepared / 'audit.json'),
                   'cohort_sha256': sha256_file(args.cohort) if args.cohort is not None else None,
                   'status': 'registered_before_model_execution'}, handle, ensure_ascii=False, indent=2)
    atoms_by_session = {}
    atoms_by_id = {}
    for atom in graph["atoms"]:
        atoms_by_session.setdefault(atom["session_id"], []).append(atom)
        atoms_by_id[atom["atom_id"]] = atom

    identity_started = time.perf_counter()
    reader_identity_files = {}
    reader_identity_stats = {}
    identity_paths = [args.reader / 'config.json', *sorted(args.reader.glob('*.safetensors'))]
    if len(identity_paths) < 2:
        raise ValueError('local immutable checkpoint weights are required')
    for path in identity_paths:
        stat = path.stat()
        reader_identity_stats[path.name] = (stat.st_size, stat.st_mtime_ns)
        reader_identity_files[path.name] = sha256_file(path)
    reader_identity = sha256_bytes(json.dumps(reader_identity_files, sort_keys=True).encode())
    reader_execution_identity = sha256_bytes(json.dumps({
        'reader_identity': reader_identity, 'patch_projection': args.reader_patch_projection,
        'attention': 'sdpa', 'dtype': 'torch.bfloat16', 'quantization': args.reader_quantization,
        'w8a16_group_size': args.w8a16_group_size if args.reader_quantization == 'w8a16' else None,
        'w8a16_residual_bits': (args.w8a16_residual_bits
                                if args.reader_quantization == 'w8a16' else None),
        'w8a16_fallback_scope': (args.w8a16_fallback_scope
                                 if args.reader_quantization == 'w8a16' else None),
        'w12a16_group_size': (args.w12a16_group_size
                              if args.reader_quantization == 'w12a16' else None),
        'w12_functional_route': args.w12_functional_route,
        'w12_numeric_mode': args.w12_numeric_mode,
        'w12_bank_count': (args.w12_bank_count
                           if args.w12_functional_route != 'none' else None),
        'w12_tile_rows': (args.w12_tile_rows
                          if args.w12_functional_route != 'none' else None),
        'reader_weight_bits': 12 if args.reader_quantization == 'w12a16' else 8 if args.reader_quantization == 'w8a16' else None,
    }, sort_keys=True).encode())
    embedder_path = args.joint_embedding_model if args.evidence_policy == 'evidence_compiler' else args.embedder
    embedder_paths = sorted(path for path in embedder_path.iterdir()
                            if path.is_file() and path.suffix in {'.json', '.safetensors', '.txt'})
    embedder_identity_files = {path.name: sha256_file(path) for path in embedder_paths}
    embedder_identity_stats = {path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                               for path in embedder_paths}
    relevance_path = (args.multimodal_reranker if args.relevance_modality == 'multimodal'
                      else args.joint_reranker_model if args.relevance_modality == 'joint_index' else args.reranker)
    reranker_paths = sorted(path for path in relevance_path.iterdir()
                            if path.is_file() and path.suffix in {'.json', '.safetensors', '.txt'}) if args.evidence_policy == 'provenance_milp' else []
    reranker_identity_files = {path.name: sha256_file(path) for path in reranker_paths}
    reranker_identity_stats = {path.name: (path.stat().st_size, path.stat().st_mtime_ns)
                               for path in reranker_paths}
    text_relevance_identity = (model_files(args.reranker)
                              if args.relevance_modality == 'joint_index' and args.evidence_policy == 'provenance_milp'
                              else None)
    trained_relevance_identity = (model_files(relevance_path)
                                 if args.relevance_modality == 'joint_index' and args.evidence_policy == 'provenance_milp'
                                 else None)
    embedder_identity = sha256_bytes(json.dumps({
        'files': embedder_identity_files, 'dtype': 'torch.bfloat16', 'attention': 'sdpa',
        'pooling': 'last_nonpad_token_then_fp32_l2_normalize',
        'token_vector_source_sha256': sha256_bytes(inspect.getsource(token_vector).encode()),
    }, sort_keys=True).encode())
    identity_seconds = time.perf_counter() - identity_started
    started = time.perf_counter()
    import transformers
    from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(embedder_path), local_files_only=True, padding_side="left")
    embedder = (AutoModel.from_pretrained(
        str(args.embedder), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device).eval()
        if args.evidence_policy != 'evidence_compiler' else None)
    reader_processor = AutoProcessor.from_pretrained(str(args.reader), local_files_only=True)
    reader_processor.image_processor.size["longest_edge"] = args.image_max_pixels
    reader = AutoModelForImageTextToText.from_pretrained(
        str(args.reader), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to(device).eval()
    configure_patch_projection(reader.model.visual, args.reader_patch_projection)
    reader_integer_primary = None
    reader_weight_only_primary = None
    w12_dispatch = None
    if args.reader_quantization == 'w8a8':
        torch.cuda.synchronize(device)
        conversion_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        before_footprint = reader.get_memory_footprint()
        with torch.inference_mode():
            reader_integer_primary = materialize_int8_primary(reader)
        torch.cuda.synchronize(device)
        reader_integer_primary.update(conversion_seconds=time.perf_counter() - conversion_started,
            model_footprint_before_bytes=before_footprint,
            model_footprint_after_bytes=reader.get_memory_footprint(),
            conversion_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
        torch.cuda.empty_cache()
    elif args.reader_quantization == 'w8a16':
        torch.cuda.synchronize(device)
        conversion_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        before_footprint = reader.get_memory_footprint()
        with torch.inference_mode():
            reader_weight_only_primary = materialize_w8a16_primary(
                reader, group_size=args.w8a16_group_size,
                fallback_scope=args.w8a16_fallback_scope,
                residual_bits=args.w8a16_residual_bits)
        torch.cuda.synchronize(device)
        reader_weight_only_primary.update(conversion_seconds=time.perf_counter() - conversion_started,
            model_footprint_before_bytes=before_footprint,
            model_footprint_after_bytes=reader.get_memory_footprint(),
            conversion_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
        torch.cuda.empty_cache()
    elif args.reader_quantization == 'w12a16':
        torch.cuda.synchronize(device)
        conversion_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        before_footprint = reader.get_memory_footprint()
        with torch.inference_mode():
            if args.w12_functional_route == 'none':
                reader_weight_only_primary = materialize_w12a16_primary(
                    reader, group_size=args.w12a16_group_size)
            else:
                w12_dispatch = W12FunctionalDispatch(
                    reader, mode=args.w12_functional_route,
                    bank_count=args.w12_bank_count,
                    tile_rows=args.w12_tile_rows,
                    group_size=args.w12a16_group_size,
                    numeric_mode=args.w12_numeric_mode,
                    release_source=args.w12_release_source,
                    fail_on_decode_fallback=not args.w12_allow_decode_fallback,
                ).__enter__()
                reader_weight_only_primary = w12_dispatch.materialization_summary()
        torch.cuda.synchronize(device)
        reader_weight_only_primary.update(conversion_seconds=time.perf_counter() - conversion_started,
            model_footprint_before_bytes=before_footprint,
            model_footprint_after_bytes=reader.get_memory_footprint(),
            conversion_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
        torch.cuda.empty_cache()
    finite_dispatch = DeferredInt8Validation(reader).__enter__() if args.defer_int8_finite_checks else None
    pim_dispatch = None
    pim_startup = None
    if args.pimnast_functional_projection:
        from pim_backend.pimnast_functional import frozen_route
        projection_hash = sha256_file(args.pimnast_functional_projection)
        candidates, capacity = frozen_route(
            json.loads(args.pimnast_functional_projection.read_text()), str(args.user_id))
        torch.cuda.synchronize(device)
        placement_started = time.perf_counter()
        before_placement = torch.cuda.memory_allocated(device)
        with torch.inference_mode():
            pim_dispatch = PimnastReaderDispatch(reader, candidates).__enter__()
        torch.cuda.synchronize(device)
        resident = pim_dispatch.summary()['resident_layout_bytes']
        if resident > capacity:
            raise ValueError('PIMnast functional layout exceeds frozen PIM pool')
        pim_startup = {'projection_path': str(args.pimnast_functional_projection),
                       'projection_sha256': projection_hash,
                       'layout_seconds_server': time.perf_counter() - placement_started,
                       'layout_resident_bytes': resident, 'declared_weight_pool_bytes': capacity,
                       'functional_extra_cuda_bytes': torch.cuda.memory_allocated(device) - before_placement,
                       'physical_primary_aliasing_verified': False,
                       'boundary': 'functional backend duplicates Host primary weights for bank-layout checking; not a deployment alias, hardware layout time or total PIM-buffer capacity qualification'}
    reranker = None
    joint_index, joint_encoder = None, None
    text_retrieval_reranker = None
    joint_encoder_identity = None
    if args.evidence_policy in {'provenance_milp', 'evidence_compiler'}:
        if args.relevance_modality == 'multimodal':
            reranker = FrozenMultimodalRelevance(
                relevance_path, device, args.image_max_pixels,
                batch_size=args.relevance_batch_size, max_input_tokens=args.input_budget)
        elif args.relevance_modality == 'joint_index':
            joint_encoder_identity = model_files(args.joint_embedding_model)
            joint_index = JointIndex(args.joint_index, sha256_file(args.prepared / 'audit.json'),
                args.user_id, joint_encoder_identity, args.joint_index_image_max_pixels,
                sha256_file(args.image_archive))
            joint_encoder = JointEncoder(args.joint_embedding_model, device,
                                         args.joint_index_image_max_pixels, args.input_budget)
            if args.evidence_policy == 'evidence_compiler' and joint_index.edge_scope != 'all_complete_exchange':
                raise ValueError('evidence compiler requires full-history multimodal and text exchange index')
            if args.evidence_policy == 'evidence_compiler':
                if args.compiler_retrieval == 'evidence_units' and joint_index.view_vectors is None:
                    raise ValueError('evidence-unit retrieval requires independently indexed text/image views')
                from build_omni_compiler_index import processor_identity
                if (joint_index.metadata['reader_processor_files_sha256'] != processor_identity(args.reader)
                        or joint_index.metadata['reader_image_max_pixels'] != args.image_max_pixels):
                    raise ValueError('offline image costs refer to another Reader processor')
            if args.evidence_policy == 'provenance_milp':
                reranker = TrainedMultimodalReranker(relevance_path, device, args.image_max_pixels,
                    batch_size=args.relevance_batch_size, max_input_tokens=args.input_budget,
                    profile=args.profile_reranker, patch_projection=args.joint_patch_projection)
                text_retrieval_reranker = FrozenRelevanceModel(args.reranker, device)
        else:
            reranker = FrozenRelevanceModel(relevance_path, device)
    visual_cache = None
    if args.visual_cache_bytes:
        processor_identity = sha256_bytes(json.dumps({
            'processor': reader_processor.to_dict(),
            'image_processor': reader_processor.image_processor.to_dict(),
            'image_max_pixels': args.image_max_pixels,
        }, sort_keys=True, default=str).encode())
        visual_cache = VisualFeatureStore(
            args.visual_cache_bytes, reader_execution_identity, processor_identity,
            output_dtype=reader.model.visual.dtype,
            spatial_merge_size=reader.model.visual.spatial_merge_size)
    model_load_seconds = time.perf_counter() - started
    model_loaded_cuda_allocated_bytes = torch.cuda.memory_allocated(device)
    model_loaded_cuda_reserved_bytes = torch.cuda.memory_reserved(device)

    session_texts = []
    for session in sessions:
        turns = atoms_by_session.get(session["session_id"], [])
        session_texts.append("\n".join(
            f"{atom['role']}: {atom['content']}" for atom in turns
            if atom.get("content_type") == "text" and atom.get("content")))
    chunk_texts, chunk_sessions = [], []
    for session_index, text in enumerate(session_texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for begin in range(0, max(1, len(token_ids)), 384):
            chunk_texts.append(tokenizer.decode(token_ids[begin:begin + 384]))
            chunk_sessions.append(session_index)
    session_vectors = []
    construction_started = time.perf_counter()
    if args.evidence_policy != 'evidence_compiler':
        for begin in range(0, len(chunk_texts), 8):
            session_vectors.append(token_vector(chunk_texts[begin:begin + 8], tokenizer, embedder, device))
        session_vectors = torch.cat(session_vectors, dim=0)
    torch.cuda.synchronize(device)
    construction_seconds = time.perf_counter() - construction_started
    archive_members = image_members(args.image_archive)
    rows = []
    hyperedge_embedding_cache = EdgeEmbeddingStore(args.edge_cache_bytes)
    trace = LinearTrace(reader, validate_bank_gemv=args.validate_bank_gemv)
    for request_index, manifest in enumerate(selected):
        query = manifest["question"]
        torch.cuda.reset_peak_memory_stats(device)
        request_started = time.perf_counter()
        pim_before = dict(pim_dispatch.calls) if pim_dispatch is not None else None
        joint_edges, joint_scores = None, None
        joint_vectors = None
        if args.evidence_policy == 'evidence_compiler':
            query_vectors, query_producer = joint_encoder.encode([{'text': query, 'images': []}])
            query_vector = query_vectors[0].detach().cpu().numpy()
            if args.compiler_retrieval == 'evidence_units':
                joint_edges, joint_scores, joint_vectors, joint_records = joint_index.search_evidence_units(
                    query_vector, args.joint_recall_count)
            else:
                joint_edges, joint_scores = joint_index.search(query_vector, args.joint_recall_count)
                joint_vectors = np.stack([joint_index.vectors[joint_index.edge_positions[edge['edge_id']]]
                                          for edge in joint_edges])
                joint_records = [{'edge_id': edge['edge_id'], 'branch': 'shared_multimodal_space',
                                  'score': score, 'score_semantics': 'joint_embedding_cosine'}
                                 for edge, score in zip(joint_edges, joint_scores)]
            candidate_session_ids = {edge['session_id'] for edge in joint_edges}
            chosen_indices = [index for index, session in enumerate(sessions)
                              if session['session_id'] in candidate_session_ids]
        elif joint_index is not None:
            query_vectors, query_producer = joint_encoder.encode([{'text': query, 'images': []}])
            visual_edges, visual_scores = joint_index.search(
                query_vectors[0].cpu().numpy(), args.joint_recall_count)
            query_vector = token_vector([query], tokenizer, embedder, device)[0]
            chunk_scores = torch.mv(session_vectors, query_vector).tolist()
            scores = torch.full((len(sessions),), -float('inf'), device=device)
            for chunk_index, session_index in enumerate(chunk_sessions):
                scores[session_index] = max(float(scores[session_index]), chunk_scores[chunk_index])
            k = min(args.candidate_sessions, len(sessions))
            chosen_indices = torch.topk(scores, k=k, largest=True, sorted=True).indices.tolist()
            ranked_ids = [sessions[index]['session_id'] for index in chosen_indices]
            text_edges = [edge for edge in build_provenance_hyperedges(ranked_ids, atoms_by_session)
                          if len(tokenizer.encode(edge['embedding_text'], add_special_tokens=False)) <= 2048
                          and all(ref.get('status') in ('exact', 'unique_user_basename')
                                  for identity in edge['atom_ids']
                                  for ref in atoms_by_id[identity].get('image_refs', []))]
            text_relevance = text_retrieval_reranker.score(
                query, [edge['embedding_text'] for edge in text_edges])
            joint_edges, joint_records = candidate_union(
                visual_edges, visual_scores, text_edges, text_relevance, args.joint_text_recall_count)
            candidate_session_ids = {edge['session_id'] for edge in joint_edges}
            chosen_indices = [index for index, session in enumerate(sessions)
                              if session['session_id'] in candidate_session_ids]
        else:
            query_vector = token_vector([query], tokenizer, embedder, device)[0]
            chunk_scores = torch.mv(session_vectors, query_vector).tolist()
            scores = torch.full((len(sessions),), -float('inf'), device=device)
            for chunk_index, session_index in enumerate(chunk_sessions):
                scores[session_index] = max(float(scores[session_index]), chunk_scores[chunk_index])
            k = min(args.candidate_sessions, len(sessions))
            chosen_indices = torch.topk(scores, k=k, largest=True, sorted=True).indices.tolist()
        chosen_sessions = [sessions[index] for index in chosen_indices]
        ranked_ids = [sessions[index]['session_id'] for index in chosen_indices]
        context_parts, evidence_selection, planned = [], [], None
        torch.cuda.synchronize(device)
        query_retrieval_seconds = time.perf_counter() - request_started
        edge_embedding_seconds, evidence_planning_seconds = 0.0, 0.0
        if args.evidence_policy in {"provenance_milp", "evidence_compiler"}:
            planning_started = time.perf_counter()
            all_edges = joint_edges if joint_edges is not None else build_provenance_hyperedges(ranked_ids, atoms_by_session)
            rejected_edges = {}
            candidate_edges = []
            for edge in all_edges:
                if (args.evidence_policy != 'evidence_compiler'
                        and len(tokenizer.encode(edge['embedding_text'], add_special_tokens=False)) > 2048):
                    rejected_edges[edge['edge_id']] = 'complete_exchange_exceeds_model_input_budget'
                elif any(ref.get('status') not in {'exact', 'unique_user_basename'}
                         for identity in edge['atom_ids'] for ref in atoms_by_id[identity].get('image_refs', [])):
                    rejected_edges[edge['edge_id']] = 'missing_or_ambiguous_observable_image_not_legally_loadable'
                else:
                    candidate_edges.append(edge)
            if not candidate_edges:
                raise ValueError('no complete provenance exchange fits reranker input')
            multimodal_records, multimodal_timing = None, None
            scoring_started = time.perf_counter()
            if args.evidence_policy == 'evidence_compiler':
                relevance = (joint_vectors @ query_vectors[0].detach().cpu().numpy()).tolist()
            elif args.relevance_modality == 'joint_index':
                relevance, multimodal_records, multimodal_timing = reranker.score(
                    query, candidate_edges, atoms_by_id,
                    lambda ref: load_image(args.image_archive, ref, archive_members))
            elif args.relevance_modality == 'multimodal':
                relevance, multimodal_records = reranker.score(
                    query, candidate_edges, atoms_by_id,
                    lambda ref: load_image(args.image_archive, ref, archive_members))
            else:
                relevance = reranker.score(query, [edge['embedding_text'] for edge in candidate_edges])
            relevance_scoring_seconds = time.perf_counter() - scoring_started
            image_refs = []
            for edge in candidate_edges:
                image_refs.extend(images_for_atoms([atoms_by_id[identity] for identity in edge['atom_ids']]))
            image_cost_started = time.perf_counter()
            image_costs = (joint_index.metadata['reader_image_token_costs']
                           if args.evidence_policy == 'evidence_compiler' else
                           processor_image_token_costs(
                               reader_processor, reader, image_refs, args.image_archive, archive_members))
            image_cost_seconds = time.perf_counter() - image_cost_started
            empty_messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': [
                {'type': 'text', 'text': '原始对话记忆：\n可观察完整交互证据：\n'
                 + f"\n\n查询时间协议：完整历史末端（{manifest['effective_timestamp']}）"
                 + f'\n问题：{query}\n请依据文字和随后附上的原始图片作答。'}]}]
            template = reader_processor.apply_chat_template(
                empty_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            label_costs = [len(reader_processor.tokenizer.encode(
                f"以下原图来自会话 {ref['source_session_id']} 的对话轮 {ref['source_atom_id']}："
                + '<|vision_start|><|vision_end|>', add_special_tokens=False)) for ref in image_refs]
            prompt_overhead = len(reader_processor.tokenizer.encode(template, add_special_tokens=False))
            prompt_overhead += sum(sorted(label_costs, reverse=True)[:args.max_images])
            if args.evidence_policy == 'evidence_compiler':
                candidate_vector_rows = np.stack([
                    joint_vectors[index] for index, edge in enumerate(joint_edges)
                    if edge['edge_id'] in {candidate['edge_id'] for candidate in candidate_edges}])
                planned = compile_evidence(
                    candidate_edges, candidate_vector_rows, query_vectors[0].detach().cpu().numpy(),
                    atoms_by_id, reader_processor.tokenizer, image_costs,
                    text_budget=args.total_text_tokens, max_images=args.max_images,
                    input_budget=args.input_budget, fixed_prompt_tokens=prompt_overhead,
                    time_limit=args.planner_time_limit)
                if args.compiler_retrieval == 'evidence_units':
                    planned['policy'] = 'evidence_compiler_independent_views_v2'
                    planned['retrieval_unit'] = 'text_or_single_image_max_over_provenance_views'
                    planned['winning_evidence_views'] = joint_records
            else:
                planned = optimize_provenance(
                    candidate_edges, relevance, atoms_by_id, reader_processor.tokenizer, image_costs,
                    text_budget=args.total_text_tokens, max_images=args.max_images,
                    input_budget=args.input_budget, fixed_prompt_tokens=prompt_overhead,
                    time_limit=args.planner_time_limit)
            evidence_planning_seconds = time.perf_counter() - planning_started
            planned['candidate_hyperedge_count'] = len(candidate_edges)
            planned['relevance_scoring_seconds'] = relevance_scoring_seconds
            planned['image_cost_seconds'] = image_cost_seconds
            if multimodal_timing is not None:
                planned['multimodal_scoring_timing'] = multimodal_timing
            planned['rejected_candidate_hyperedges'] = rejected_edges
            planned['candidate_hyperedge_limit'] = len(candidate_edges)
            if args.evidence_policy != 'evidence_compiler':
                planned['candidate_replay'] = [
                    {'edge_id': edge['edge_id'], 'atom_ids': edge['atom_ids'],
                     'relevance_log_odds': float(score)}
                    for edge, score in zip(candidate_edges, relevance)]
            planned['reranker'] = str(relevance_path)
            planned['relevance_modality'] = args.relevance_modality
            planned['multimodal_relevance_records'] = multimodal_records
            if joint_index is not None:
                planned['joint_index_lookup'] = {'index_json_sha256': sha256_file(args.joint_index / 'index.json'),
                    'candidate_ids': [edge['edge_id'] for edge in joint_edges],
                    'retrieval_branch_records': joint_records, 'query_producer': query_producer[0],
                    'score_semantics': ('shared cosine semantic coverage; no reranking'
                                        if args.evidence_policy == 'evidence_compiler' else
                                        'cosine only for recall; trained binary relevance logits for MILP')}
            planned['prompt_overhead_tokens'] = prompt_overhead
            if args.evidence_policy != 'evidence_compiler':
                planned['planner'] = 'frozen pointwise ' + args.relevance_modality + ' logits + exact provenance MILP'
            context_parts = ['可观察完整交互证据：\n' + planned['text']]
            accepted = set(planned['accepted_atom_ids'])
            for session in chosen_sessions:
                session_atoms = atoms_by_session[session['session_id']]
                evidence_selection.append({'session_id': session['session_id'],
                    'accepted_atom_ids': [atom['atom_id'] for atom in session_atoms if atom['atom_id'] in accepted],
                    'unselected_atom_ids': [atom['atom_id'] for atom in session_atoms if atom['atom_id'] not in accepted]})
            chosen_image_refs = planned['images']
        elif args.evidence_policy == "local_hyperedge":
            edge_embedding_started = time.perf_counter()
            all_edges = build_local_hyperedges(ranked_ids, atoms_by_session, radius=2)
            local_edges = [edge for edge in all_edges if len(tokenizer.encode(
                edge["embedding_text"], add_special_tokens=True)) <= 2048]
            embedding_rejected_ids = sorted({edge['edge_id'] for edge in all_edges}
                                            - {edge['edge_id'] for edge in local_edges})
            def encode_edges(texts):
                return list(token_vector(texts, tokenizer, embedder, device).detach())

            cached_vectors, producer_records = resolve_producers(
                local_edges, atoms_by_id, None if args.disable_edge_reuse else hyperedge_embedding_cache, encode_edges,
                embedder_identity, batch_size=32,
                output_digest=lambda values: VisualFeatureStore.tensor_digest(torch.stack(values)),
                request_index=request_index)
            if not local_edges:
                raise ValueError("no complete local hyperedge fits the embedding input limit")
            edge_vectors = torch.stack([cached_vectors[edge['edge_id']] for edge in local_edges])
            edge_scores = torch.mv(edge_vectors, query_vector).tolist()
            local_edges, edge_scores, candidate_pruned_ids = bound_hyperedges(
                local_edges, edge_scores, atoms_by_id, args.max_hyperedges,
                image_edges_per_session=args.image_edges_per_session)
            torch.cuda.synchronize(device)
            edge_embedding_seconds = time.perf_counter() - edge_embedding_started
            planning_started = time.perf_counter()
            atom_token_costs = (AtomUnionTokenCosts(reader_processor.tokenizer,
                {identity: atoms_by_id[identity] for edge in local_edges for identity in edge['atom_ids']})
                if args.query_local_token_cost_reuse else None)
            planned = select_hyperedges(
                local_edges, edge_scores, atoms_by_id, reader_processor.tokenizer,
                text_budget=args.total_text_tokens, max_images=args.max_images, require_image=True,
                image_novelty_weight=args.image_novelty_weight, atom_token_costs=atom_token_costs)
            if atom_token_costs is not None:
                planned['query_local_atom_token_cost_reuse'] = atom_token_costs.summary()
                del atom_token_costs
            evidence_planning_seconds = time.perf_counter() - planning_started
            planned["embedding_over_budget_edge_ids"] = embedding_rejected_ids
            planned['candidate_pruned_edge_ids'] = candidate_pruned_ids
            planned['candidate_replay'] = [{'edge_id': edge['edge_id'], 'atom_ids': edge['atom_ids'],
                                           'similarity': score} for edge, score in zip(local_edges, edge_scores)]
            planned['embedding_producers'] = producer_records
            planned['embedding_producer_costs'] = {
                'lookup_seconds': sum(record['lookup_seconds'] for record in producer_records),
                'encode_seconds': sum(record['encode_seconds'] for record in producer_records),
                'write_seconds': sum(record['write_seconds'] for record in producer_records),
                'reused_batches': sum(record['reused'] for record in producer_records),
                'miss_batches': sum(not record['reused'] for record in producer_records),
                'reused_vectors': sum(record['reused_vectors'] for record in producer_records),
                'computed_vectors': sum(record['computed_vectors'] for record in producer_records),
                'reuse_scope': 'cross_request_only; exact ordered producer batch; model identity bound',
            }
            planned["candidate_hyperedge_count"] = len(local_edges)
            planned["candidate_hyperedge_limit"] = args.max_hyperedges
            planned['image_candidate_quota_per_session'] = args.image_edges_per_session
            planned['image_novelty_weight'] = args.image_novelty_weight
            planned["hyperedge_embedding_cache_hits"] = sum(record['reused_vectors'] for record in producer_records)
            planned["hyperedge_embedding_cache_misses"] = sum(record['computed_vectors'] for record in producer_records)
            planned['embedding_store'] = hyperedge_embedding_cache.summary()
            context_parts = ["局部对话超边：\n" + planned["text"]]
            for session in chosen_sessions:
                session_atoms = atoms_by_session[session["session_id"]]
                accepted = set(planned["accepted_atom_ids"])
                evidence_selection.append({"session_id": session["session_id"],
                    "accepted_atom_ids": [atom["atom_id"] for atom in session_atoms if atom["atom_id"] in accepted],
                    "unselected_atom_ids": [atom["atom_id"] for atom in session_atoms if atom["atom_id"] not in accepted]})
            chosen_image_refs = planned["images"]
        else:
            for session in chosen_sessions:
                packed, accepted, rejected = pack_session_turns(
                    atoms_by_session.get(session["session_id"], []),
                    reader_processor.tokenizer, args.session_text_tokens)
                evidence_selection.append({"session_id": session["session_id"],
                                           "accepted_atom_ids": accepted,
                                           "budget_rejected_atom_ids": rejected})
                if packed:
                    context_parts.append(f"会话 {session['session_id']}：\n" + packed)
            chosen_image_refs = select_dialogue_images(ranked_ids, atoms_by_session, args.max_images)
        torch.cuda.synchronize(device)
        evidence_seconds = time.perf_counter() - request_started
        preparation_started = time.perf_counter()
        loaded_images = [load_image(args.image_archive, ref, archive_members) for ref in chosen_image_refs]
        if args.evidence_policy == 'local_hyperedge':
            del cached_vectors, edge_vectors
        user_content = [{"type": "text", "text": "原始对话记忆：\n" + "\n\n".join(context_parts)
                         + f"\n\n查询时间协议：完整历史末端（{manifest['effective_timestamp']}）"
                         + f"\n问题：{query}\n请依据文字和随后附上的原始图片作答。"}]
        for ref, (image, _) in zip(chosen_image_refs, loaded_images):
            user_content.append({"type": "text", "text":
                f"以下原图来自会话 {ref['source_session_id']} 的对话轮 {ref['source_atom_id']}："})
            user_content.append({"type": "image", "image": image})
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_content}]
        prompt_text = reader_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        processor_kwargs = {'text': [prompt_text], 'return_tensors': 'pt'}
        if loaded_images:
            processor_kwargs['images'] = [image for image, _ in loaded_images]
        inputs = reader_processor(**processor_kwargs).to(device)
        if inputs.input_ids.shape[1] > args.input_budget:
            raise ValueError("processor prompt exceeds registered input budget; do not silently truncate")
        processor_inputs_sha256 = {name: VisualFeatureStore.tensor_digest(value)
                                   for name, value in inputs.items() if isinstance(value, torch.Tensor)}
        if args.diagnostic_input_dump_dir is not None:
            args.diagnostic_input_dump_dir.mkdir(parents=True, exist_ok=True)
            tensor_inputs = {name: value.detach().cpu()
                             for name, value in inputs.items()
                             if isinstance(value, torch.Tensor)}
            torch.save(tensor_inputs, args.diagnostic_input_dump_dir
                       / f"{manifest['question_id']}.reader_inputs.pt")
            (args.diagnostic_input_dump_dir
             / f"{manifest['question_id']}.reader_inputs.json").write_text(json.dumps({
                 'question_id': manifest['question_id'], 'person_id': manifest['person_id'],
                 'tensor_inputs_sha256': processor_inputs_sha256,
                 'tensor_shapes': {name: list(value.shape) for name, value in tensor_inputs.items()},
                 'tensor_dtypes': {name: str(value.dtype) for name, value in tensor_inputs.items()},
                 'boundary': ('diagnostic-only exact post-retrieval processor inputs; '
                              'no evaluation labels; not a serving output'),
             }, ensure_ascii=False, indent=2) + '\n')
        torch.cuda.synchronize(device)
        input_preparation_seconds = time.perf_counter() - preparation_started
        trace.phase = "prefill"
        visual_before = visual_cache.snapshot() if visual_cache is not None else None
        prefill_started = time.perf_counter()
        visual_context = (visual_cache.activate(reader.model, [digest for _, digest in loaded_images])
                          if visual_cache is not None and loaded_images else nullcontext())
        with visual_context, measure_vision(reader.model, lambda: torch.cuda.synchronize(device)) as vision_timing, torch.inference_mode():
            output = reader(**inputs, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(device)
        prefill_seconds = time.perf_counter() - prefill_started
        cache = output.past_key_values
        tokens = []
        ttft_seconds = None
        eos_id = getattr(reader.config.text_config, "eos_token_id", reader_processor.tokenizer.eos_token_id)
        stop_ids = set(eos_id if isinstance(eos_id, list) else [eos_id])
        trace.phase = "decode"
        decode_started = time.perf_counter()
        stop_reason = 'max_new_tokens'
        with torch.inference_mode():
            for position in range(args.max_new_tokens):
                token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                if position == 0:
                    torch.cuda.synchronize(device)
                    ttft_seconds = time.perf_counter() - request_started
                if tokens[-1] in stop_ids:
                    stop_reason = "eos"
                    break
                if position + 1 < args.max_new_tokens:
                    output = reader(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
        torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - decode_started
        ttlt_seconds = time.perf_counter() - request_started
        conversion_started = time.perf_counter()
        answer = reader_processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        answer_conversion_seconds = time.perf_counter() - conversion_started
        service_seconds = time.perf_counter() - request_started
        service_peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated(device)
        service_peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved(device)
        validation_seconds, generation_equivalent = 0.0, None
        visual_features_bitwise_equivalent = None
        reference_vision_seconds = []
        if args.verify_generation:
            del output, cache
            torch.cuda.empty_cache()
            trace.phase = "reference_validation"
            validation_started = time.perf_counter()
            original_get = reader.model.get_image_features
            reference_feature_hashes = []

            def reference_get(pixel_values, image_grid_thw=None, **kwargs):
                torch.cuda.synchronize(device)
                vision_started = time.perf_counter()
                result = original_get(pixel_values, image_grid_thw, **kwargs)
                torch.cuda.synchronize(device)
                reference_vision_seconds.append(time.perf_counter() - vision_started)
                reference_feature_hashes.append(tuple(VisualFeatureStore.tensor_digest(value)
                                                     for value in result.pooler_output))
                return result

            if visual_cache is not None:
                reader.model.get_image_features = reference_get
            try:
                with torch.inference_mode():
                    reference_tokens = reader.generate(
                        **inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            finally:
                if visual_cache is not None:
                    del reader.model.get_image_features
            if visual_cache is not None:
                visual_features_bitwise_equivalent = (reference_feature_hashes == [visual_cache.last_feature_hashes])
                if not visual_features_bitwise_equivalent or visual_cache.bypasses:
                    args.output.with_suffix('.failure.json').write_text(json.dumps({
                        'question_id': manifest['question_id'], 'completed_requests': len(rows),
                        'cache_snapshot': visual_cache.snapshot(),
                        'service_feature_hashes': visual_cache.last_feature_hashes,
                        'reference_feature_hashes': reference_feature_hashes,
                        'token_ids_equivalent': reference_tokens[0, inputs.input_ids.shape[1]:].tolist() == tokens,
                        'reason': 'visual feature identity or supported-interface gate failed',
                        'complete': False}, indent=2))
                    raise ValueError('reused vision features differ bitwise from uncached original-image reference')
            reference_ids = reference_tokens[0, inputs.input_ids.shape[1]:].tolist()
            generation_equivalent = reference_ids == tokens
            if not generation_equivalent:
                raise ValueError("manual cached decode differs from independent full-prompt generate")
            torch.cuda.synchronize(device)
            validation_seconds = time.perf_counter() - validation_started
            del reference_tokens
        selected_sessions_sorted = sorted(chosen_sessions, key=lambda item: item["ordinal"])
        rows.append({
            "question_id": manifest["question_id"], "person_id": manifest["person_id"],
            "visibility_protocol": manifest["visibility_protocol"],
            "effective_timestamp": manifest["effective_timestamp"],
            "question_sha256": sha256_bytes(query.encode()), "answer": answer,
            "answer_sha256": sha256_bytes(answer.encode()), "answer_token_ids": tokens,
            "stop_reason": stop_reason,
            "input_tokens": int(inputs.input_ids.shape[1]),
            "visual_patches": int(inputs.pixel_values.shape[0]) if 'pixel_values' in inputs else 0,
            "images": [{"archive_member": ref.get("resolved", ref["normalized"]), "sha256": image_hash,
                        "source_session_id": ref["source_session_id"],
                        "source_atom_id": ref["source_atom_id"],
                        "selection_source": ref["selection_source"]}
                       for ref, (_, image_hash) in zip(chosen_image_refs, loaded_images)],
            "retrieved_session_ids": [session["session_id"] for session in selected_sessions_sorted],
            "evidence_selection": evidence_selection,
            "hypergraph_evidence_plan": planned,
            "embedding_producers": planned.get('embedding_producers') if planned is not None else None,
            "processor_inputs_sha256": processor_inputs_sha256,
            'pim_functional_calls': ({name: count - pim_before[name]
                for name, count in pim_dispatch.calls.items()} if pim_dispatch is not None else None),
            "service_stage_ledger": service_stage_ledger(
                service_seconds, evidence_seconds=evidence_seconds,
                input_preparation_seconds=input_preparation_seconds, prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds, answer_conversion_seconds=answer_conversion_seconds,
                vision_in_prefill_seconds=vision_timing['vision_in_prefill_seconds']),
            "retrieval_seconds": service_seconds - prefill_seconds - decode_seconds,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            'ttft_seconds': ttft_seconds,
            'ttlt_seconds': ttlt_seconds,
            'token_latency_boundary': 'request start through first/last greedy token id available on Host; includes evidence, preparation and model work; excludes transport/rendering; last token may be EOS',
            "service_seconds": service_seconds,
            "query_retrieval_seconds": query_retrieval_seconds,
            "edge_embedding_seconds": edge_embedding_seconds,
            "evidence_planning_seconds": evidence_planning_seconds,
            "service_peak_cuda_allocated_bytes": service_peak_cuda_allocated_bytes,
            "service_peak_cuda_reserved_bytes": service_peak_cuda_reserved_bytes,
            "generation_equivalent": generation_equivalent,
            "generation_validation_seconds": validation_seconds,
            "visual_features_bitwise_equivalent": visual_features_bitwise_equivalent,
            "reference_vision_seconds": reference_vision_seconds,
            "visual_cache_before": visual_before,
            "visual_cache_after": visual_cache.snapshot() if visual_cache is not None else None,
        })
        args.output.with_suffix('.partial.json').write_text(json.dumps({
            'complete': False, 'rows': rows, 'boundary': 'smoke in progress, not quality result'},
            ensure_ascii=False, indent=2))
        print(json.dumps({"question_id": manifest["question_id"], "input_tokens": rows[-1]["input_tokens"],
                          "visual_patches": rows[-1]["visual_patches"], "prefill_seconds": prefill_seconds,
                          "decode_seconds": decode_seconds, "stop_reason": stop_reason}, ensure_ascii=False),
              flush=True)
        if not args.verify_generation:
            del output, cache
        del inputs, messages, loaded_images
        torch.cuda.empty_cache()
    trace.close()
    pim_summary = pim_dispatch.summary() if pim_dispatch is not None else None
    w12_summary = w12_dispatch.summary() if w12_dispatch is not None else None
    finite_summary = finite_dispatch.summary() if finite_dispatch is not None else None
    if finite_dispatch is not None:
        finite_dispatch.__exit__(None, None, None)
    if pim_dispatch is not None:
        pim_dispatch.__exit__(None, None, None)
        if sha256_file(args.pimnast_functional_projection) != projection_hash:
            raise ValueError('frozen PIMnast route changed during service')
    if w12_dispatch is not None:
        w12_dispatch.__exit__(None, None, None)
    for name, expected in reader_identity_stats.items():
        stat = (args.reader / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError('pinned Reader checkpoint changed during visual reuse validation')
    for name, expected in embedder_identity_stats.items():
        stat = (embedder_path / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError('immutable embedder producer changed during service')
    for name, expected in reranker_identity_stats.items():
        stat = (relevance_path / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise ValueError('immutable relevance model changed during service')
    if sha256_file(Path(__file__)) != source_sha256:
        raise ValueError('runner source changed during execution')
    if any(sha256_file(ROOT / name) != expected for name, expected in source_files_sha256.items()):
        raise ValueError('imported runtime source changed during execution')
    if joint_encoder_identity is not None and model_files(args.joint_embedding_model) != joint_encoder_identity:
        raise ValueError('joint encoder checkpoint changed during execution')
    if text_relevance_identity is not None and model_files(args.reranker) != text_relevance_identity:
        raise ValueError('text recall reranker changed during execution')
    if trained_relevance_identity is not None and model_files(relevance_path) != trained_relevance_identity:
        raise ValueError('multimodal reranker changed during execution')
    if args.validate_bank_gemv and len(trace.functional_samples) != 3:
        raise ValueError('required actual-activation GEMV probes did not all execute')
    split_label = ",".join(args.allowed_splits)
    result = {
        "schema_version": "omni-raw-image-reader-smoke-v4", "complete": True,
        "performance_comparable": (not args.validate_bank_gemv and not args.screening_run
                                   and pim_dispatch is None
                                   and args.w12_functional_route == 'none'),
        "screening_run": args.screening_run,
        "numerical_probe_overhead_in_service": args.validate_bank_gemv,
        "verify_generation": args.verify_generation,
        "diagnostic_input_dump_dir": (str(args.diagnostic_input_dump_dir)
                                      if args.diagnostic_input_dump_dir is not None else None),
        "protocol": f"{len(selected)} frozen {split_label} questions; history_end visibility; label-free dense session retrieval",
        "allowed_splits": args.allowed_splits,
        "device": str(device), "reader": str(args.reader), "embedder": str(embedder_path),
        "reader_config_sha256": sha256_file(args.reader / "config.json"),
        "embedder_config_sha256": sha256_file(embedder_path / "config.json"),
        "reader_weight_bytes": sum(path.stat().st_size for path in args.reader.glob("*.safetensors")),
        "model_load_seconds": model_load_seconds,
        "model_loaded_cuda_allocated_bytes": model_loaded_cuda_allocated_bytes,
        "model_loaded_cuda_reserved_bytes": model_loaded_cuda_reserved_bytes,
        'construction_seconds': construction_seconds,
        'embedding_windows': len(chunk_texts),
        'script_sha256': source_sha256,
        'source_files_sha256': source_files_sha256,
        "system_prompt_sha256": sha256_bytes(SYSTEM.encode()),
        "candidate_sessions": args.candidate_sessions, "max_images": args.max_images,
        'query_local_token_cost_reuse_enabled': args.query_local_token_cost_reuse,
        "image_selection_policy": ("indivisible provenance exchange union; processor-grid budget; no QA image refs"
                                   if args.evidence_policy in {"provenance_milp", "evidence_compiler"} else
                                   "indivisible local dialogue hyperedge union; no QA image refs"
                                   if args.evidence_policy == "local_hyperedge" else
                                   "dialogue-observed images only; round robin over dense-ranked sessions; no QA image refs"),
        "session_text_tokens": args.session_text_tokens,
        "evidence_policy_name": args.evidence_policy, "total_text_tokens": args.total_text_tokens,
        'relevance_modality': args.relevance_modality,
        'compiler_retrieval': args.compiler_retrieval,
        'joint_index_metadata': ({'index_json_sha256': sha256_file(args.joint_index / 'index.json'),
            'vector_bytes': joint_index.vectors.nbytes, 'producer_hits': joint_index.hits,
            'evidence_view_count': len(joint_index.views),
            'evidence_view_vector_bytes': (joint_index.view_vectors.nbytes
                                           if joint_index.view_vectors is not None else 0),
            'retrieval_unit': joint_index.metadata.get('retrieval_unit'),
            'construction_seconds': joint_index.metadata['construction_seconds'],
            'index_image_max_pixels': joint_index.metadata['image_max_pixels'],
            'cold_construction_in_request_service': False,
            'producer_boundary': 'prebuilt immutable development history index; not Reader KV reuse',
            'encoder_model_files_sha256': joint_encoder_identity} if joint_index is not None else None),
        'text_retrieval_reranker_identity': text_relevance_identity,
        'trained_multimodal_reranker_identity': trained_relevance_identity,
        'joint_patch_projection': args.joint_patch_projection,
        'relevance_model': str(relevance_path) if reranker is not None else None,
        "embedding_state_reuse": hyperedge_embedding_cache.summary(),
        "embedding_state_reuse_enabled": not args.disable_edge_reuse and args.evidence_policy == 'local_hyperedge',
        "visual_state_reuse": visual_cache.snapshot() if visual_cache is not None else None,
        "visual_state_identity_files_sha256": reader_identity_files,
        "visual_state_identity_seconds": identity_seconds,
        "visual_state_reuse_boundary": "ordered whole-image batch, exact pixels/grid/model/processor; CPU bounded store; reference bypasses store; no Reader KV/recurrent reuse",
        "embedding_state_reuse_boundary": "run-local immutable model/processor; edge identity plus exact text hash; construction charged on miss; not Reader KV/recurrent reuse",
        "image_max_pixels": args.image_max_pixels,
        "evidence_policy": ("offline same-space multimodal exchange index; query-weighted semantic coverage compiler; no online reranker"
                            if args.evidence_policy == 'evidence_compiler' else
                            "frozen " + args.relevance_modality + " relevance + exact provenance MILP with processor-grid image costs"
                            if args.evidence_policy == "provenance_milp" else
                            "query-weighted local dialogue hyperedge union, global token/image budgets; smoke only"
                            if args.evidence_policy == "local_hyperedge" else
                            "chronological complete-turn first-fit per retrieved session; explicit rejected IDs; smoke only"),
        "max_new_tokens": args.max_new_tokens,
        "image_novelty_weight": args.image_novelty_weight,
        "archive_sha256": sha256_file(args.image_archive),
        "prepared_audit_sha256": sha256_file(args.prepared / "audit.json"),
        "reader_identity": reader_identity,
        "reader_patch_projection": args.reader_patch_projection,
        'reader_quantization': args.reader_quantization,
        'reader_w8a16_fallback_scope': args.w8a16_fallback_scope,
        'reader_w8a16_residual_bits': args.w8a16_residual_bits,
        'reader_w12a16_group_size': (args.w12a16_group_size
                                     if args.reader_quantization == 'w12a16' else None),
        'reader_weight_bits': 12 if args.reader_quantization == 'w12a16' else 8 if args.reader_quantization == 'w8a16' else None,
        'reader_integer_primary': reader_integer_primary,
        'reader_weight_only_primary': reader_weight_only_primary,
        'reader_w12_bank_dispatch': w12_summary,
        'reader_finite_validation': finite_summary,
        'reader_functional_backend': ('w12_bit_sliced_bank_functional'
                                      if w12_dispatch is not None else
                                      'pimnast_bank_layout_reference' if pim_dispatch is not None else
                                      'host_w12a16_dequantized' if args.reader_quantization == 'w12a16' else
                                      'host_w8a16_dequantized' if reader_weight_only_primary is not None else
                                      'host_primary'),
        'pim_functional_startup': pim_startup,
        'pim_functional_dispatch': pim_summary,
        'physical_pim_execution': False,
        "reader_execution_identity": reader_execution_identity,
        "reranker_identity_files_sha256": reranker_identity_files,
        "embedding_execution_identity": embedder_identity,
        "embedding_source_identity_scope": "token_vector function source; full serving snapshot recorded separately",
        "embedder_identity_files_sha256": embedder_identity_files,
        "runtime_environment": {'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'device_name': torch.cuda.get_device_name(device)},
        "embedding_producer_policy": "session-local canonical ordered batches of 32; partial producer hits recompute the whole exact batch; producer outputs are content-addressed and model-identity bound",
        "measurement_contract": {
            "scope": "online serving: retrieval, raw-image preparation, prefill, decode and answer conversion; model/index startup excluded",
            "hardware_identity": args.hardware_identity or str(device),
            "concurrency": args.concurrency_contract or "single_worker",
            "startup_policy": args.startup_policy,
            "cohort_registration_sha256": args.cohort_registration_sha256,
            "input_archive_sha256": sha256_file(args.image_archive),
            "reader_identity": reader_identity,
            "reader_execution_identity": reader_execution_identity,
        },
        "timing_origin": "measured_server_wall_clock_seconds",
        "diagnostic_graph_trace_in_service": False,
        "serving_input_files": ["omni_graph.json", "manifests.jsonl", "split.json"],
        "evaluation_label_files_read": [], "rows": rows, "operator_trace": trace.summary(),
        "boundary": "smoke only; raw image path executes on GPU; not quality or end-to-end PIM evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "questions": len(rows), "output": str(args.output),
                      "model_load_seconds": model_load_seconds}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
