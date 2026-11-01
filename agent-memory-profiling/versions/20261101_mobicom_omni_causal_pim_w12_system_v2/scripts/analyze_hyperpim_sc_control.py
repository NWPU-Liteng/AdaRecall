"""Audit host mask-cache state keys and grammar jump-ahead on observed decode paths."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from probe_grammar_sparsity import SCHEMAS, compile_grammar


def digest_rows(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_raw_outputs(run_paths):
    outputs = {}
    for path in run_paths:
        run = json.loads(Path(path).read_text())
        for row in run["rows"]:
            outputs[(str(row["person_id"]), row["question_id"], "parser")] = (
                (row["proof_plan"].get("query_parse") or {}).get("raw") or "")
            outputs[(str(row["person_id"]), row["question_id"], "grounding")] = (
                (row["proof_plan"].get("grounding") or {}).get("raw") or "")
    return outputs


def buffer_cost(vocab_size, repeats=20000):
    words = np.zeros((vocab_size + 31) // 32, dtype=np.int32)
    source = np.arange(words.size, dtype=np.int32)
    copy_samples = []
    hash_samples = []
    for _ in range(100):
        target = np.empty_like(words)
        started = time.perf_counter_ns()
        np.copyto(target, source)
        copy_samples.append(time.perf_counter_ns() - started)
        payload = source.tobytes()
        started = time.perf_counter_ns()
        hashlib.sha256(payload).digest()
        hash_samples.append(time.perf_counter_ns() - started)
    return {
        "mask_bytes": words.size * words.itemsize,
        "copy_seconds_p50": statistics.median(copy_samples) / 1e9,
        "sha256_seconds_p50": statistics.median(hash_samples) / 1e9,
        "copy_calibration_repeats": 100,
        "requested_repeats": repeats,
    }


def cache_audit(calls, vocab_size, max_context=8):
    contexts = (0, 1, 2, 4, 8)
    contexts = tuple(value for value in contexts if value <= max_context)
    operations = []
    for call in calls:
        tokens = call["tokens"]
        if not tokens:
            continue
        for index, row in enumerate(tokens):
            operations.append({
                "kind": call["kind"],
                "call_id": (str(call["person_id"]), call["question_id"], call["kind"]),
                "index": index,
                "mask": row["mask_sha256"],
                "token": row["token_id"],
                "history": tuple(token["token_id"] for token in tokens[:index]),
                "fill_seconds": float(row["fill_seconds"]),
            })

    results = {}
    for context in contexts:
        next_masks = defaultdict(set)
        occurrence_counts = defaultdict(int)
        for call in calls:
            tokens = call["tokens"]
            for index in range(1, len(tokens)):
                previous = tuple(tokens[position]["token_id"]
                                 for position in range(max(0, index - context), index))
                key = (tokens[index - 1]["mask_sha256"], tokens[index - 1]["token_id"],
                       previous)
                next_masks[key].add(tokens[index]["mask_sha256"])
                occurrence_counts[key] += 1

        conflicts = {key for key, masks in next_masks.items() if len(masks) > 1}
        conflict_occurrences = sum(occurrence_counts[key] for key in conflicts)
        seen = set()
        cache_hits = 0
        cache_misses = 0
        fallback_fills = 0
        fill_seconds = 0.0
        for row in operations:
            key = ("initial", row["kind"]) if row["index"] == 0 else (
                row["mask"], row["token"], row["history"][-context:] if context else ())
            if key in conflicts:
                fallback_fills += 1
                cache_misses += 1
                fill_seconds += row["fill_seconds"]
            elif key in seen:
                cache_hits += 1
            else:
                seen.add(key)
                cache_misses += 1
                fill_seconds += row["fill_seconds"]
        results[str(context)] = {
            "distinct_state_keys": len(set(
                ("initial", row["kind"]) if row["index"] == 0 else (
                    row["mask"], row["token"], row["history"][-context:] if context else ())
                for row in operations)),
            "transition_keys": len(next_masks),
            "transition_conflicts": len(conflicts),
            "transition_conflict_occurrences": conflict_occurrences,
            "conflict_key_examples": [list(key[:2]) for key in list(conflicts)[:8]],
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "conflict_fallback_fills": fallback_fills,
            "fill_seconds": fill_seconds,
        }
    return {
        "tokens": len(operations),
        "calls": len(calls),
        "context_results": results,
        "key_boundary": (
            "mask hash plus recent-token suffix is an empirical state fingerprint, not a complete "
            "pushdown-stack identity; non-conflicting replay does not prove general exactness"),
    }


def jump_ahead_audit(calls, raw_outputs, tokenizer, vocab_size, grammars):
    import xgrammar as xgr

    details = []
    for call in calls:
        token_ids = [row["token_id"] for row in call["tokens"]]
        if not token_ids:
            continue
        key = (str(call["person_id"]), call["question_id"], call["kind"])
        raw = raw_outputs.get(key)
        if raw is None or tokenizer.encode(raw, add_special_tokens=False) != token_ids:
            raise ValueError("raw output does not reproduce replayed token path: " + repr(key))
        matcher = xgr.GrammarMatcher(grammars[call["kind"]],
                                     terminate_without_stop_token=True)
        position = 0
        branches = 0
        skipped_tokens = 0
        jump_characters = 0
        jump_events = 0
        skipped_fill_seconds = 0.0
        while position < len(token_ids):
            jump = matcher.find_jump_forward_string()
            jump_ids = tokenizer.encode(jump, add_special_tokens=False) if jump else []
            if jump_ids and token_ids[position:position + len(jump_ids)] == jump_ids:
                matcher.accept_string(jump)
                skipped_tokens += len(jump_ids)
                skipped_fill_seconds += sum(
                    call["tokens"][index]["fill_seconds"]
                    for index in range(position, position + len(jump_ids)))
                jump_characters += len(jump)
                jump_events += 1
                position += len(jump_ids)
                continue
            if not matcher.accept_token(token_ids[position]):
                raise ValueError("observed token rejected during jump replay: " + repr(key))
            branches += 1
            position += 1
        details.append({
            "person_id": call["person_id"], "question_id": call["question_id"],
            "kind": call["kind"], "tokens": len(token_ids), "model_branches": branches,
            "skipped_tokens": skipped_tokens, "jump_events": jump_events,
            "jump_characters": jump_characters,
            "skipped_fill_seconds": skipped_fill_seconds,
        })
    aggregate = {}
    for kind in SCHEMAS:
        kind_rows = [row for row in details if row["kind"] == kind]
        aggregate[kind] = {
            "calls": len(kind_rows),
            "tokens": sum(row["tokens"] for row in kind_rows),
            "model_branches": sum(row["model_branches"] for row in kind_rows),
            "skipped_tokens": sum(row["skipped_tokens"] for row in kind_rows),
            "skipped_fraction": (
                sum(row["skipped_tokens"] for row in kind_rows)
                / sum(row["tokens"] for row in kind_rows)),
            "jump_events": sum(row["jump_events"] for row in kind_rows),
            "jump_characters": sum(row["jump_characters"] for row in kind_rows),
            "skipped_fill_seconds": sum(row["skipped_fill_seconds"] for row in kind_rows),
        }
    return {"aggregate": aggregate, "calls": details,
            "boundary": "exact tokenizer-prefix jump only; no approximation and no model execution"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=8)
    args = parser.parse_args()
    if args.max_context < 0:
        raise ValueError("nonnegative context length required")

    grammar = json.loads(args.grammar.read_text())
    calls = grammar["calls"]
    vocab_size = int(grammar["vocab_size"])
    cache = cache_audit(calls, vocab_size, args.max_context)
    costs = buffer_cost(vocab_size)
    for result in cache["context_results"].values():
        result["cache_hit_copy_seconds"] = result["cache_hits"] * (
            costs["copy_seconds_p50"] + costs["sha256_seconds_p50"])
        result["effective_control_seconds"] = (
            result["fill_seconds"] + result["cache_hit_copy_seconds"])
        result["fill_avoidance_fraction"] = 1 - (
            result["fill_seconds"] / sum(row["fill_seconds"] for call in calls
                                         for row in call["tokens"]))

    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.reader, local_files_only=True)
    config = AutoConfig.from_pretrained(args.reader, local_files_only=True)
    configured_vocab = int(config.text_config.vocab_size)
    if configured_vocab != vocab_size:
        raise ValueError("grammar and Reader vocabularies differ")
    grammars = {kind: compile_grammar(tokenizer, vocab_size, schema)
                for kind, schema in SCHEMAS.items()}
    raw_outputs = load_raw_outputs(args.runs)
    jump = jump_ahead_audit(calls, raw_outputs, tokenizer, vocab_size, grammars)
    result = {
        "schema": "hyperpim-sc-control-cache-and-jump-audit-v1",
        "inputs": {"grammar": str(args.grammar), "grammar_sha256": hashlib.sha256(
            args.grammar.read_bytes()).hexdigest(), "runs": [str(path) for path in args.runs]},
        "boundary": (
            "trace replay only; cache timing uses observed CPU fill time plus measured copy/hash "
            "costs, and jump-ahead uses exact tokenizer prefixes; not a model or PIM measurement"),
        "buffer_costs": costs,
        "mask_cache": cache,
        "jump_ahead": jump,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "buffer_costs": costs,
        "mask_cache": cache,
        "jump_ahead": jump["aggregate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
