#!/usr/bin/env python3
"""Audit a complete W12 system run without conflating quality and PIM timing."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import yaml


STAGES = ("query_parser", "source_grounding", "reader_generation")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "sum_seconds": None, "mean_seconds": None,
                "p50_seconds": None, "p95_seconds": None}
    return {
        "count": len(values),
        "sum_seconds": sum(values),
        "mean_seconds": statistics.fmean(values),
        "p50_seconds": percentile(values, 0.50),
        "p95_seconds": percentile(values, 0.95),
    }


def load_reader_rows(root: Path, users: list[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    rows: dict[str, dict] = {}
    documents: dict[str, dict] = {}
    for user_id in users:
        for suffix in (".json", ".partial.json"):
            path = root / f"user_{user_id}_shard_0{suffix}"
            if not path.exists():
                continue
            document = json.loads(path.read_text())
            documents[str(path)] = document
            for row in document.get("rows") or []:
                identity = row["question_id"]
                previous = rows.get(identity)
                if previous is not None and previous.get("answer_sha256") != row.get("answer_sha256"):
                    raise ValueError(f"conflicting checkpoint rows for {identity}")
                rows[identity] = row
    return rows, documents


def load_judge_rows(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(root.glob("*/*.json")):
        row = json.loads(path.read_text())
        identity = row["question_id"]
        if identity in rows:
            raise ValueError(f"duplicate Judge row: {identity}")
        rows[identity] = row
    return rows


def correct(row: dict | None) -> bool:
    return bool(row and row.get("parse_valid") and row.get("judge_label") == "CORRECT")


def mcnemar_exact(candidate_only: int, reference_only: int) -> float:
    discordant = candidate_only + reference_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(candidate_only, reference_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def paired_bootstrap(differences_by_user: dict[str, list[int]], *, samples: int,
                     seed: int) -> list[float]:
    users = sorted((user for user, values in differences_by_user.items() if values), key=int)
    if not users:
        return [None, None]
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        numerator = 0
        denominator = 0
        for _ in users:
            values = differences_by_user[generator.choice(users)]
            numerator += sum(values)
            denominator += len(values)
        estimates.append(100.0 * numerator / denominator)
    estimates.sort()
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def block(ids: set[str], judged: dict[str, dict]) -> dict[str, Any]:
    evaluated = ids & set(judged)
    count = sum(correct(judged[identity]) for identity in evaluated)
    return {
        "questions": len(ids),
        "judged": len(evaluated),
        "correct": count,
        "success_rate_judged": count / len(evaluated) if evaluated else None,
        "all_question_lower_bound": count / len(ids) if ids else None,
    }


def pairing(ids: list[str], candidate: dict[str, dict], reference: dict[str, dict],
            users: list[str], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    common = sorted(set(ids) & set(candidate) & set(reference))
    if not common:
        return {"paired_questions": 0}
    candidate_only = sum(correct(candidate[q]) and not correct(reference[q]) for q in common)
    reference_only = sum(correct(reference[q]) and not correct(candidate[q]) for q in common)
    both_correct = sum(correct(candidate[q]) and correct(reference[q]) for q in common)
    both_wrong = len(common) - candidate_only - reference_only - both_correct
    by_user: dict[str, list[int]] = {user: [] for user in users}
    for identity in common:
        user_id = identity.split("_", 1)[0]
        by_user.setdefault(user_id, []).append(
            int(correct(candidate[identity])) - int(correct(reference[identity])))
    delta_pp = 100.0 * (candidate_only - reference_only) / len(common)
    bootstrap_ci = paired_bootstrap(by_user, samples=bootstrap_samples, seed=seed)
    return {
        "paired_questions": len(common),
        "partition": {
            "both_correct": both_correct,
            "candidate_only": candidate_only,
            "reference_only": reference_only,
            "both_wrong": both_wrong,
        },
        "candidate_correct": sum(correct(candidate[q]) for q in common),
        "reference_correct": sum(correct(reference[q]) for q in common),
        "paired_delta_pp": delta_pp,
        "paired_delta_bootstrap_ci95_pp": bootstrap_ci,
        "exact_mcnemar_two_sided_p": mcnemar_exact(candidate_only, reference_only),
        "noninferiority_margin_pp": -5.0,
        "noninferiority_passed": bootstrap_ci[0] is not None and bootstrap_ci[0] > -5.0,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "scope": "paired questions resampled within observed users",
        },
    }


def stage_audit(rows: dict[str, dict]) -> dict[str, Any]:
    stage_totals = {stage: Counter() for stage in STAGES}
    expected_hits = 0
    decode_fallback_zero = 0
    single_layout = 0
    unexpected_zero: Counter = Counter()
    layouts: set[str] = set()
    for row in rows.values():
        ledger = row.get("w12_stage_dispatch")
        if not isinstance(ledger, dict) or set(ledger) != set(STAGES):
            raise ValueError(f"incomplete stage ledger for {row['question_id']}")
        identities = {ledger[stage]["layout_instance_sha256"] for stage in STAGES}
        if len(identities) != 1:
            raise ValueError(f"multiple layouts for {row['question_id']}")
        layouts.update(identities)
        single_layout += 1
        grounding = row.get("proof_plan", {}).get("grounding", {})
        expected = {
            "query_parser": True,
            "source_grounding": grounding.get("stop_reason") != "not_applicable_contextual",
            "reader_generation": True,
        }
        for stage in STAGES:
            item = ledger[stage]
            for name in ("bank_calls", "decode_fallback_calls",
                         "prefill_fallback_calls", "fallback_rows",
                         "bank_external_bytes", "fallback_external_bytes"):
                stage_totals[stage][name] += int(item.get(name, 0))
        if all(ledger[stage]["bank_calls"] > 0
               for stage in STAGES if expected[stage]):
            expected_hits += 1
        for stage in STAGES:
            if expected[stage] and ledger[stage]["bank_calls"] == 0:
                unexpected_zero[stage] += 1
        if all(ledger[stage]["decode_fallback_calls"] == 0 for stage in STAGES):
            decode_fallback_zero += 1
    bank_bytes = sum(stage_totals[stage]["bank_external_bytes"] for stage in STAGES)
    fallback_bytes = sum(stage_totals[stage]["fallback_external_bytes"] for stage in STAGES)
    return {
        "requests": len(rows),
        "stages": {stage: dict(values) for stage, values in stage_totals.items()},
        "all_expected_stages_hit_bank": expected_hits,
        "unexpected_zero_stage_requests": dict(unexpected_zero),
        "decode_fallback_zero_requests": decode_fallback_zero,
        "single_layout_identity_requests": single_layout,
        "layout_count": len(layouts),
        "bank_external_bytes": bank_bytes,
        "host_fallback_external_bytes": fallback_bytes,
        "host_fallback_fraction_of_external_bytes": (
            fallback_bytes / (bank_bytes + fallback_bytes)
            if bank_bytes + fallback_bytes else None),
    }


def summarize(config: dict, baseline_reader: Path, baseline_judge: Path,
              pim_reference: Path | None, bootstrap_samples: int, seed: int) -> dict:
    cohort_path = Path(config["paths"]["cohort"])
    cohort = json.loads(cohort_path.read_text())
    users = sorted(cohort["question_ids_by_user"], key=int)
    ids = [identity for user in users for identity in cohort["question_ids_by_user"][user]]
    prepared = Path(config["paths"]["prepared"])
    manifests = {row["question_id"]: row for row in load_jsonl(prepared / "manifests.jsonl")}
    labels = {row["question_id"]: row for row in load_jsonl(prepared / "labels.jsonl")}
    run_root = Path(config["paths"]["run_root"])
    reader, reader_documents = load_reader_rows(run_root, users)
    judge = load_judge_rows(run_root / "judge_rows")

    by_user = {user: block(set(cohort["question_ids_by_user"][user]), judge)
               for user in users}
    by_split = {}
    for split in sorted({manifests[identity]["split"] for identity in ids}):
        by_split[split] = block(
            {identity for identity in ids if manifests[identity]["split"] == split}, judge)
    by_type = {}
    for question_type in sorted({labels[identity]["question_type"] for identity in ids}):
        by_type[question_type] = block(
            {identity for identity in ids
             if labels[identity]["question_type"] == question_type}, judge)

    audit_path = Path(config.get("subset", {}).get(
        "population_audit", config.get("subset", {}).get("audit", "")))
    type_weights = {}
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        type_weights = {name: float(item["population_share"]) for name, item in
                        audit.get("type_audit", {}).items()}
    natural_weighted = sum(
        stats["success_rate_judged"] * type_weights[name]
        for name, stats in by_type.items()
        if stats["success_rate_judged"] is not None and name in type_weights)

    reference_reader, _ = load_reader_rows(baseline_reader, users)
    reference_judge = load_judge_rows(baseline_judge)
    paired = pairing(ids, judge, reference_judge, users, bootstrap_samples, seed)
    paired_by_type = {}
    for question_type in by_type:
        selected = [identity for identity in ids
                    if labels[identity]["question_type"] == question_type]
        paired_by_type[question_type] = pairing(
            selected, judge, reference_judge, users, bootstrap_samples, seed)

    measured_service = [float(reader[q]["service_seconds"]) for q in ids if q in reader]
    measured_ttft = [float(reader[q]["ttft_seconds"]) for q in ids if q in reader]
    prefill = [float(reader[q]["prefill_seconds"]) for q in ids if q in reader]
    decode = [float(reader[q]["decode_seconds"]) for q in ids if q in reader]
    contract_values = {
        "functional_route": sorted({str(document.get("w12_functional_route"))
                                    for document in reader_documents.values()}),
        "numeric_mode": sorted({str(document.get("w12_numeric_mode"))
                                for document in reader_documents.values()}),
        "bank_count": sorted({int(document.get("w12_bank_count"))
                              for document in reader_documents.values()}),
        "tile_rows": sorted({int(document.get("w12_tile_rows"))
                             for document in reader_documents.values()}),
        "host_contract_rows": sorted({int(document.get("w12_host_contract_rows"))
                                      for document in reader_documents.values()}),
        "group_size": sorted({int(document.get("w12a16_group_size"))
                              for document in reader_documents.values()}),
        "release_source": sorted({bool(document.get("w12_release_source"))
                                  for document in reader_documents.values()}),
    }
    pim = None
    if pim_reference is not None:
        closure = json.loads(pim_reference.read_text())
        pim = {
            "reference": str(pim_reference),
            "reference_sha256": digest(pim_reference),
            "claim_level": closure.get("claim_level"),
            "gates": closure.get("gates"),
            "request_level_target": closure.get("request_level_target"),
            "capacity": closure.get("capacity"),
            "boundary": closure.get("boundary"),
        }

    return {
        "schema": "mobilemem-w12-system-run-audit-v1",
        "complete": len(reader) == len(ids) and len(judge) == len(ids),
        "run_root": str(run_root),
        "cohort": str(cohort_path),
        "cohort_sha256": digest(cohort_path),
        "expected_questions": len(ids),
        "reader": {"completed": len(reader), "expected": len(ids)},
        "judge": {"judged": len(set(ids) & set(judge)), "expected": len(ids)},
        "quality": {
            "overall": block(set(ids), judge),
            "by_user": by_user,
            "by_split": by_split,
            "by_type": by_type,
            "natural_distribution_weighted_success": natural_weighted,
            "natural_weight_source": str(audit_path) if type_weights else None,
        },
        "paired_host_reference": {
            "reader_root": str(baseline_reader),
            "judge_root": str(baseline_judge),
            "overall": paired,
            "by_type": paired_by_type,
            "reference_rows": len(set(ids) & set(reference_reader)),
        },
        "w12_execution": {
            "reader_contract": contract_values,
            **stage_audit(reader),
        },
        "latency": {
            "functional_emulation_wall_clock": {
                "service": latency(measured_service),
                "ttft": latency(measured_ttft),
                "prefill": latency(prefill),
                "answer_decode": latency(decode),
                "boundary": "measured Torch functional-bank emulation; not PIM silicon latency",
            },
            "target_model_projection": pim,
        },
        "boundary": [
            "quality comes from the fixed local Qwen3-VL-8B Judge on development/validation only",
            "functional-bank wall time is emulator cost and must not be reported as PIM latency",
            "target PIM timing and bytes are analytical projections tied to the frozen W12 closure",
            "prefill and contract-inexact calls remain explicit Host fallbacks",
            "no physical PIM, RTL, phone, energy or thermal measurement was executed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-reader-root", type=Path, required=True)
    parser.add_argument("--baseline-judge-root", type=Path, required=True)
    parser.add_argument("--pim-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261031)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("main experiment configuration must be a mapping")
    result = summarize(config, args.baseline_reader_root, args.baseline_judge_root,
                       args.pim_reference, args.bootstrap_samples, args.bootstrap_seed)
    atomic_write(args.output, result)
    print(json.dumps({
        "complete": result["complete"],
        "reader": result["reader"],
        "judge": result["judge"],
        "success_rate_judged": result["quality"]["overall"]["success_rate_judged"],
        "paired_delta_pp": result["paired_host_reference"]["overall"].get("paired_delta_pp"),
        "decode_fallback_zero_requests": result["w12_execution"]["decode_fallback_zero_requests"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
