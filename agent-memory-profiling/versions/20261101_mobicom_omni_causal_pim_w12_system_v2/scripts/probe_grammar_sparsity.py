"""Replay observed constrained token paths to measure grammar-mask density and cost."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import time

import numpy as np

from omni_proof_grounding import WITNESS_SCHEMA
from omni_proof_query_contract import QUERY_CONTRACT_SCHEMA


SCHEMAS = {
    "parser": QUERY_CONTRACT_SCHEMA,
    "grounding": WITNESS_SCHEMA,
}

_WORKER = {}


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def distribution(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def compile_grammar(tokenizer, vocab_size, schema):
    import xgrammar as xgr

    stop_tokens = getattr(tokenizer, "eos_token_id", None)
    stop_tokens = stop_tokens if isinstance(stop_tokens, list) else [stop_tokens]
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=stop_tokens)
    compiler = xgr.GrammarCompiler(info, max_threads=2)
    return compiler.compile_json_schema(
        schema, strict_mode=True, separators=(",", ":"))


def allowed_token_count(mask, vocab_size):
    words = mask[0].numpy()
    allowed = int(np.bitwise_count(words).sum())
    remainder = vocab_size % 32
    if remainder:
        valid_in_last = np.uint32((1 << remainder) - 1)
        allowed -= int(np.bitwise_count(np.uint32(words[-1]) & ~valid_in_last))
    return allowed


def replay_call(tokenizer, vocab_size, grammar, token_ids):
    import xgrammar as xgr

    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    mask = xgr.allocate_token_bitmask(1, vocab_size)
    rows = []
    for position, token_id in enumerate(token_ids):
        started = time.perf_counter()
        needs_mask = bool(matcher.fill_next_token_bitmask(mask))
        fill_seconds = time.perf_counter() - started
        allowed = allowed_token_count(mask, vocab_size)
        mask_sha256 = hashlib.sha256(mask[0].numpy().tobytes()).hexdigest()
        accepted = bool(matcher.accept_token(int(token_id)))
        rows.append({
            "position": position,
            "token_id": int(token_id),
            "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "allowed_tokens": allowed,
            "allowed_fraction": allowed / vocab_size,
            "mask_sha256": mask_sha256,
            "needs_mask": needs_mask,
            "fill_seconds": fill_seconds,
            "accepted": accepted,
            "terminated_after_accept": bool(matcher.is_terminated()),
        })
        if not accepted:
            raise ValueError(f"observed token path rejected at position {position}: {token_id}")
    return rows


def bucket_counts(fractions):
    buckets = [
        (0.0, 0.0),
        (0.0, 1 / 1024),
        (1 / 1024, 1 / 256),
        (1 / 256, 1 / 64),
        (1 / 64, 1 / 16),
        (1 / 16, 1 / 4),
        (1 / 4, 1 / 2),
        (1 / 2, 3 / 4),
        (3 / 4, 1.0),
    ]
    result = []
    for lower, upper in buckets:
        if lower == upper:
            count = sum(value == 0 for value in fractions)
        else:
            count = sum(lower < value <= upper for value in fractions)
        result.append({"lower_exclusive": lower, "upper_inclusive": upper, "tokens": count})
    return result


def summarize(rows, vocab_size):
    allowed = [row["allowed_tokens"] for row in rows]
    fractions = [row["allowed_fraction"] for row in rows]
    fills = [row["fill_seconds"] for row in rows]
    total_allowed = sum(allowed)
    mask_counts = {}
    for row in rows:
        mask_counts[row["mask_sha256"]] = mask_counts.get(row["mask_sha256"], 0) + 1
    repeated_tokens = sum(count for count in mask_counts.values() if count > 1)
    return {
        "tokens": len(rows),
        "vocab_size": vocab_size,
        "allowed_tokens": distribution(allowed),
        "allowed_fraction": distribution(fractions),
        "vocab_weighted_allowed_fraction": total_allowed / (len(rows) * vocab_size),
        "dense_lm_head_mac_fraction_for_observed_path": total_allowed / (len(rows) * vocab_size),
        "grammar_fill_seconds": distribution(fills),
        "grammar_fill_seconds_total": sum(fills),
        "grammar_fill_milliseconds_per_token": 1000 * statistics.fmean(fills),
        "singleton_masks": sum(row["allowed_tokens"] == 1 for row in rows),
        "empty_masks": sum(row["allowed_tokens"] == 0 for row in rows),
        "allowed_le_64": sum(row["allowed_tokens"] <= 64 for row in rows),
        "allowed_le_1024": sum(row["allowed_tokens"] <= 1024 for row in rows),
        "allowed_le_8192": sum(row["allowed_tokens"] <= 8192 for row in rows),
        "allowed_le_65536": sum(row["allowed_tokens"] <= 65536 for row in rows),
        "needs_mask_false": sum(not row["needs_mask"] for row in rows),
        "unique_masks": len(mask_counts),
        "tokens_reusing_a_previously_seen_mask": repeated_tokens,
        "unique_mask_fraction": len(mask_counts) / len(rows),
        "most_common_mask_counts": sorted(mask_counts.values(), reverse=True)[:16],
        "fraction_buckets": bucket_counts(fractions),
    }


def _initialize_worker(reader):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(reader, local_files_only=True)
    config = AutoConfig.from_pretrained(reader, local_files_only=True)
    vocab_size = int(config.text_config.vocab_size)
    grammars = {kind: compile_grammar(tokenizer, vocab_size, schema)
                for kind, schema in SCHEMAS.items()}
    _WORKER.update(tokenizer=tokenizer, vocab_size=vocab_size, grammars=grammars)


def _replay_path(path):
    tokenizer = _WORKER["tokenizer"]
    vocab_size = _WORKER["vocab_size"]
    grammars = _WORKER["grammars"]
    run = json.loads(Path(path).read_text())
    if not run.get("complete"):
        raise ValueError("complete trace required: " + str(path))
    calls = []
    for row in run["rows"]:
        plan = row["proof_plan"]
        observed = {
            "parser": plan.get("query_parse") or {},
            "grounding": plan.get("grounding") or {},
        }
        for kind, result in observed.items():
            raw = result.get("raw") or ""
            declared_tokens = int(result.get("output_token_count") or 0)
            token_ids = tokenizer.encode(raw, add_special_tokens=False) if raw else []
            if len(token_ids) != declared_tokens:
                raise ValueError(
                    f"tokenizer round-trip mismatch for {row['question_id']} {kind}: "
                    f"{len(token_ids)} != {declared_tokens}")
            if not token_ids:
                continue
            token_rows = replay_call(
                tokenizer, vocab_size, grammars[kind], token_ids)
            calls.append({
                "person_id": row["person_id"],
                "question_id": row["question_id"],
                "kind": kind,
                "stop_reason": result.get("stop_reason"),
                "declared_model_decode_seconds": (result.get("timing") or {}).get(
                    "reader_decode_seconds"),
                "tokens": token_rows,
            })
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_paths = sorted(path for path in args.root.glob("user_*.json")
                       if not path.name.endswith(".partial.json"))
    if len(run_paths) != 4:
        raise ValueError("exactly four complete per-user traces required")
    if isinstance(args.workers, bool) or not 1 <= args.workers <= len(run_paths):
        raise ValueError("workers must be between one and four")
    reader_config = json.loads((args.reader / "config.json").read_text())
    vocab_size = int(reader_config["text_config"]["vocab_size"])

    context = multiprocessing.get_context("fork")
    with context.Pool(processes=args.workers, initializer=_initialize_worker,
                      initargs=(str(args.reader),)) as pool:
        per_path = pool.map(_replay_path, [str(path) for path in run_paths], chunksize=1)
    calls = [call for path_calls in per_path for call in path_calls]

    by_kind = {}
    for kind in SCHEMAS:
        kind_calls = [call for call in calls if call["kind"] == kind]
        kind_rows = [row for call in kind_calls for row in call["tokens"]]
        summary = summarize(kind_rows, vocab_size)
        summary["calls"] = len(kind_calls)
        summary["calls_with_contextual_zero_output"] = {
            "parser": 0,
            "grounding": 16 - len(kind_calls),
        }[kind]
        summary["call_tokens"] = distribution([len(call["tokens"]) for call in kind_calls])
        summary["call_fill_seconds"] = distribution(
            [sum(row["fill_seconds"] for row in call["tokens"]) for call in kind_calls])
        by_kind[kind] = summary

    all_rows = [row for call in calls for row in call["tokens"]]
    result = {
        "schema": "hyperpim-sc-grammar-sparsity-v1",
        "measurement_boundary": (
            "CPU replay of exact observed constrained token paths; no model execution, "
            "no quality claim, and no target-PIM latency claim"),
        "reader": str(args.reader),
        "vocab_size": vocab_size,
        "trace_files": [str(path) for path in run_paths],
        "requests": len(run_paths) * 4,
        "structured_calls": len(calls),
        "by_kind": by_kind,
        "combined": summarize(all_rows, vocab_size),
        "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                      ("structured_calls", "by_kind", "combined")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
