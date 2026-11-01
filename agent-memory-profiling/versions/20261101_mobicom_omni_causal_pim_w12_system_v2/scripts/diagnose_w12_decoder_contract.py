#!/usr/bin/env python3
"""Audit every Reader decoder matrix against the bank-tile-aligned W12 contract.

The script loads the frozen Reader checkpoint and materializes one matrix at a
time.  It compares the production Host W12A16 row-chunk contract with both the
single-row bank decoder and the multi-row Host fallback used by the functional
W12 backend.  Exact equality is required; this is a numerical contract check,
not a latency measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from omni_proof_runtime import load_reader
from pim_backend.functional_decode import Int12WeightOnlyContract
from pim_backend.w12_bank_layout import W12BankLayout


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_digest(shapes):
    value = json.dumps(sorted(shapes), separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def selected_modules(reader):
    return [
        (name, module)
        for name, module in reader.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (name.startswith("model.language_model.") or name == "lm_head")
    ]


def compare(actual, expected, *, phase, rows):
    if actual.shape != expected.shape:
        raise ValueError(f"{phase} output shape differs: {actual.shape} != {expected.shape}")
    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    exact = torch.equal(actual, expected)
    return {
        "phase": phase,
        "rows": rows,
        "output_elements": int(actual.numel()),
        "bitwise_equal": exact,
        "equal_elements": int((actual == expected).sum()),
        "max_absolute_error": float(difference.max().item()),
        "relative_l2_error": float(
            (difference.norm() / expected_float.norm().clamp_min(1e-30)).item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bank-count", type=int, default=256)
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--contract-rows", type=int, default=4096)
    parser.add_argument("--prefill-rows", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh diagnostic output required")
    if (args.group_size <= 0 or args.bank_count <= 0 or args.tile_rows <= 0
            or args.contract_rows <= 0 or not args.prefill_rows
            or any(value <= 0 for value in args.prefill_rows)):
        raise ValueError("positive matrix and row dimensions required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    _processor, reader = load_reader(args.reader)
    modules = selected_modules(reader)
    projection = json.loads(args.projection.read_text())
    inventory = projection["decode_token_accounting"]["decoder_bank_residency"]["matrices"]
    expected_shapes = sorted(
        (int(row["out_features"]), int(row["in_features"])) for row in inventory)
    actual_shapes = sorted(
        (int(module.weight.shape[0]), int(module.weight.shape[1]))
        for _name, module in modules)
    if Counter(actual_shapes) != Counter(expected_shapes):
        raise ValueError("loaded Reader matrix inventory differs from the frozen projection")
    if any(module.weight.dtype != torch.bfloat16 for _name, module in modules):
        raise ValueError("frozen Reader language weights must be BF16")

    records = []
    matrix_started = time.perf_counter()
    for index, (name, module) in enumerate(modules):
        weight = module.weight.detach()
        bias = module.bias.detach() if module.bias is not None else None
        contract = Int12WeightOnlyContract(
            weight, bias, group_size=args.group_size,
            row_chunk=args.contract_rows).eval()
        layout = W12BankLayout(
            weight, group_size=args.group_size, bank_count=args.bank_count,
            tile_rows=args.tile_rows, bias=bias)
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(args.seed + index)
        decode_activation = torch.randn(
            weight.shape[1], dtype=torch.bfloat16, device=weight.device,
            generator=generator)
        host_decode = contract(decode_activation.unsqueeze(0)).squeeze(0)
        bank_decode = layout.gemv(
            decode_activation, numeric_mode="bf16_reference")
        checks = [compare(bank_decode, host_decode, phase="decode", rows=1)]
        for rows in args.prefill_rows:
            activation = torch.randn(
                rows, weight.shape[1], dtype=torch.bfloat16,
                device=weight.device, generator=generator)
            host_output = contract(activation)
            fallback_output = layout.linear(activation)
            checks.append(compare(
                fallback_output, host_output, phase="prefill_fallback", rows=rows))
        records.append({
            "name": name,
            "out_features": int(weight.shape[0]),
            "in_features": int(weight.shape[1]),
            "has_bias": bias is not None,
            "bank_active_banks": int(layout.active_banks),
            "bank_rounds": int(layout.rounds),
            "bank_padded_rows": int(layout.padded_rows),
            "checks": checks,
            "all_bitwise_equal": all(item["bitwise_equal"] for item in checks),
        })
        del contract, layout, generator, decode_activation, host_decode, bank_decode
        del checks, weight, bias, module
        if index % 16 == 15:
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    decode_checks = [item for record in records for item in record["checks"]
                     if item["phase"] == "decode"]
    fallback_checks = [item for record in records for item in record["checks"]
                       if item["phase"] == "prefill_fallback"]
    shape_digest_value = shape_digest(actual_shapes)
    result = {
        "schema_version": "w12-decoder-contract-diagnostic-v1",
        "complete": True,
        "reader_model": str(args.reader),
        "reader_config_sha256": digest(args.reader / "config.json"),
        "projection": str(args.projection),
        "projection_sha256": digest(args.projection),
        "inventory": {
            "matrices": len(records),
            "projection_matrices": len(inventory),
            "shape_multiset_sha256": shape_digest_value,
            "shape_counts": [
                {"out_features": output, "in_features": input, "count": count}
                for (output, input), count in sorted(Counter(actual_shapes).items())
            ],
        },
        "contract": {
            "reader_quantization": "w12a16",
            "weight_bits": 12,
            "group_size": args.group_size,
            "contract_row_chunk": args.contract_rows,
            "bank_count": args.bank_count,
            "tile_rows": args.tile_rows,
            "numeric_mode": "bf16_reference",
        },
        "summary": {
            "matrices_added": len(records),
            "decode_checks": len(decode_checks),
            "decode_bitwise_equal": sum(item["bitwise_equal"] for item in decode_checks),
            "prefill_fallback_checks": len(fallback_checks),
            "prefill_fallback_bitwise_equal": sum(
                item["bitwise_equal"] for item in fallback_checks),
            "max_absolute_error": max(
                item["max_absolute_error"]
                for record in records for item in record["checks"]),
            "all_checks_bitwise_equal": all(
                record["all_bitwise_equal"] for record in records),
        },
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "matrix_loop_seconds": time.perf_counter() - matrix_started,
        "boundary": [
            "functional Torch execution on an RTX 3090, not silicon PIM",
            "wall-clock fields are diagnostic only and are not PIM latency results",
            "activations are deterministic BF16 probes; checkpoint weights are real",
            "prefill and multi-row paths are explicit Host fallbacks",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "matrices": len(records),
        "decode_bitwise_equal": result["summary"]["decode_bitwise_equal"],
        "prefill_fallback_bitwise_equal": result["summary"][
            "prefill_fallback_bitwise_equal"],
        "prefill_fallback_checks": len(fallback_checks),
        "max_absolute_error": result["summary"]["max_absolute_error"],
        "all_checks_bitwise_equal": result["summary"]["all_checks_bitwise_equal"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
