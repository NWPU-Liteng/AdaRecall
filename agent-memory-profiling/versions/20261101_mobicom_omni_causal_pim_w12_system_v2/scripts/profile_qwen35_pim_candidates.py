"""Static Qwen3.5 weight-flow profile for low-batch LPDDR-PIM screening.

This tool reads model configuration and safetensors headers only. It does not
claim runtime operator timing, device residency, or PIM speedup. Runtime
traces must replace these estimates before a system-level conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I4": 0.5,
    "U4": 0.5,
}


def load_tensor_headers(model_dir: Path) -> list[dict]:
    tensors = []
    names = set()
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors files under {model_dir}")
    for path in files:
        with path.open("rb") as handle:
            header_size_raw = handle.read(8)
            if len(header_size_raw) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", header_size_raw)[0]
            if header_size < 2 or header_size > path.stat().st_size - 8:
                raise ValueError(f"invalid safetensors header length: {path}")
            header_raw = handle.read(header_size)
            header = json.loads(header_raw)
        payload_bytes = path.stat().st_size - 8 - header_size
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError(f"duplicate tensor across shards: {name}")
            names.add(name)
            dtype = str(spec["dtype"])
            shape = [int(value) for value in spec["shape"]]
            if any(value < 0 for value in shape):
                raise ValueError(f"negative tensor dimension: {name}")
            elements = 1
            for value in shape:
                elements *= value
            offsets = spec["data_offsets"]
            if (len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= payload_bytes):
                raise ValueError(f"tensor payload outside shard: {name}")
            if dtype in DTYPE_BYTES and offsets[1] - offsets[0] != elements * DTYPE_BYTES[dtype]:
                raise ValueError(f"tensor shape and byte count disagree: {name}")
            tensors.append({
                "file": path.name,
                "header_sha256": hashlib.sha256(header_raw).hexdigest(),
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "elements": elements,
                "bytes": int(offsets[1] - offsets[0]),
                "dtype_bytes": DTYPE_BYTES.get(dtype),
            })
    return tensors


def group_for(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "auxiliary_mtp"
    if ".mlp." in name:
        return "language_mlp"
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "full_attention"
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding_or_head"
    return "other"


def layer_id(name: str) -> int | None:
    marker = ".layers."
    if marker not in name:
        return None
    suffix = name.split(marker, 1)[1]
    token = suffix.split(".", 1)[0]
    return int(token) if token.isdigit() else None


def gemv_candidate(tensor: dict, activation_bytes: int, weight_bytes: int) -> dict | None:
    shape = tensor["shape"]
    if len(shape) != 2:
        return None
    out_features, in_features = shape
    macs = out_features * in_features
    bytes_read = weight_bytes + in_features * activation_bytes
    bytes_written = out_features * activation_bytes
    return {
        "name": tensor["name"],
        "layer": layer_id(tensor["name"]),
        "group": group_for(tensor["name"]),
        "shape": shape,
        "source_dtype": tensor["dtype"],
        "source_bytes": tensor["bytes"],
        "modeled_weight_bytes": weight_bytes,
        "activation_bytes": in_features * activation_bytes,
        "output_bytes": bytes_written,
        "macs": macs,
        "arithmetic_intensity_macs_per_byte": macs / max(1, bytes_read + bytes_written),
        "screening_boundary": "static weight-flow estimate; no runtime timing or PIM claim",
    }


def profile(model_dir: Path, *, activation_bytes: int, pim_weight_bytes: int) -> dict:
    if activation_bytes not in (1, 2, 4) or pim_weight_bytes not in (1, 2, 4):
        raise ValueError("activation and modeled weight bytes must be 1, 2 or 4")
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    tensors = load_tensor_headers(model_dir)
    groups = defaultdict(lambda: {"bytes": 0, "tensors": 0})
    for tensor in tensors:
        group = group_for(tensor["name"])
        groups[group]["bytes"] += tensor["bytes"]
        groups[group]["tensors"] += 1
    candidates = []
    tied_embeddings = text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
    for tensor in tensors:
        name = tensor["name"]
        runtime_role = ("output_projection" if name.endswith("lm_head.weight")
                        or tied_embeddings and name.endswith("embed_tokens.weight")
                        else "language_linear")
        if (group_for(name) not in {"language_mlp", "linear_attention", "full_attention"}
                and runtime_role != "output_projection"):
            continue
        candidate = gemv_candidate(
            tensor,
            activation_bytes=activation_bytes,
            weight_bytes=int(tensor["elements"] * pim_weight_bytes),
        )
        if candidate is not None:
            candidate["runtime_role"] = runtime_role
            candidates.append(candidate)
    candidates.sort(key=lambda item: (-item["modeled_weight_bytes"], item["name"]))
    layer_types = list(text_config.get("layer_types", []))
    return {
        "schema_version": "qwen35-pim-candidate-profile-v1",
        "model_dir": str(model_dir),
        "config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
        "shard_header_sha256": {tensor["file"]: tensor["header_sha256"] for tensor in tensors},
        "model_architecture": config.get("architectures"),
        "text_config": {
            key: text_config.get(key)
            for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "head_dim", "layer_types",
                        "linear_key_head_dim", "linear_value_head_dim",
                        "linear_num_key_heads", "linear_num_value_heads")
            if key in text_config
        },
        "vision_config": config.get("vision_config", {}),
        "weight_summary": dict(groups),
        "total_tensor_bytes": sum(tensor["bytes"] for tensor in tensors),
        "layer_type_counts": dict(Counter(layer_types)),
        "candidate_count": len(candidates),
        "candidate_layers": candidates,
        "screening_contract": {
            "activation_bytes": activation_bytes,
            "pim_weight_bytes": pim_weight_bytes,
            "meaning": "rank language GEMV weight-flow candidates, including tied output head, before runtime trace; auxiliary MTP excluded",
            "representation_boundary": "modeled weight width is a storage scenario, not implemented quantization; scale/metadata excluded",
            "arithmetic_intensity_boundary": "MAC per logical weight/input/output byte for one row; not captured DRAM traffic or FLOP per byte",
            "checkpoint_boundary": "config and shard headers hashed; weight payload hashes not computed",
            "excluded": ["device residency", "bank conflicts", "command/fence", "refresh",
                         "layout conversion", "partial-sum reduction", "quality impact",
                         "end-to-end latency"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-bytes", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--pim-weight-bytes", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = profile(args.model_dir, activation_bytes=args.activation_bytes,
                     pim_weight_bytes=args.pim_weight_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "model": result["model_architecture"],
        "layers": result["layer_type_counts"],
        "candidate_count": result["candidate_count"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
