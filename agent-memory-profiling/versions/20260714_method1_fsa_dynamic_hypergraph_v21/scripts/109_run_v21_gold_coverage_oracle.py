#!/usr/bin/env python3
"""Compute the non-deployable gold-coverage headroom oracle on train OOF only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_sets import fold_for  # noqa: E402
from v21.io import iter_jsonl, load_config, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402


RUN_ID = "v21_gold_coverage_oracle_r1_20260714"
OUTPUT_REVISION = "r1_20260714"
BUDGETS = (8, 16, 32)
PROTECTED_SEEDS = 5


def load_ann_prefix(path: Path) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for row in iter_jsonl(path):
        query_id = str(row["query_id"])
        candidates = tuple(dict.fromkeys(map(str, row.get("ann_candidate_ids", []))))
        if len(candidates) < PROTECTED_SEEDS:
            raise ValueError(f"frozen ANN prefix is shorter than {PROTECTED_SEEDS}: {query_id}")
        previous = output.get(query_id)
        if previous is not None and previous != candidates:
            raise ValueError(f"inconsistent frozen ANN prefix: {query_id}")
        output[query_id] = candidates
    return output


def load_train_rows(data: dict) -> dict[int, list[dict]]:
    rows_by_fold: dict[int, list[dict]] = defaultdict(list)
    for row in iter_jsonl(data["split"]["train_query_plan"]):
        query_id = str(row["query_id"])
        eligible = tuple(dict.fromkeys(map(str, row["eligible_memory_ids"])))
        eligible_set = set(eligible)
        groups = []
        for group in row.get("evidence_groups", []):
            members = tuple(dict.fromkeys(map(str, group.get("acceptable_memory_ids", []))))
            if not members or not set(members).issubset(eligible_set):
                raise ValueError(f"invalid train evidence group for {query_id}")
            groups.append(members)
        fold = fold_for(str(row["persona_id"]), int(row["query_session_id"]), 5)
        rows_by_fold[fold].append({
            "query_id": query_id,
            "persona_id": str(row["persona_id"]),
            "query_session_id": int(row["query_session_id"]),
            "eligible_memory_ids": eligible,
            "groups": tuple(groups),
        })
    if sum(len(rows) for rows in rows_by_fold.values()) != int(data["split"]["train_queries"]):
        raise ValueError("train query plan count does not match frozen data config")
    return rows_by_fold


def _gold_masks(
    protected: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
    budget: int,
) -> tuple[int, int, tuple[str, ...], int, int]:
    """Exact maximum group coverage with mandatory protected members."""
    group_count = len(groups)
    full_mask = (1 << group_count) - 1
    group_by_memory: dict[str, int] = defaultdict(int)
    for group_index, group in enumerate(groups):
        bit = 1 << group_index
        for memory_id in group:
            group_by_memory[memory_id] |= bit
    initial_mask = 0
    for memory_id in protected:
        initial_mask |= group_by_memory.get(memory_id, 0)
    candidates = tuple(sorted(memory_id for memory_id, mask in group_by_memory.items() if memory_id not in protected and mask))
    slots = max(0, int(budget) - len(protected))
    states: dict[int, tuple[str, ...]] = {initial_mask: ()}
    for _slot in range(slots):
        next_states = dict(states)
        for mask, chosen in states.items():
            chosen_set = set(chosen)
            for memory_id in candidates:
                if memory_id in chosen_set:
                    continue
                new_mask = mask | group_by_memory[memory_id]
                new_chosen = chosen + (memory_id,)
                previous = next_states.get(new_mask)
                if previous is None or (len(new_chosen), new_chosen) < (len(previous), previous):
                    next_states[new_mask] = new_chosen
        states = next_states
    best_mask, best_chosen = max(
        states.items(), key=lambda item: (item[0].bit_count(), -len(item[1]), tuple(reversed(item[1])))
    )
    min_slots_to_full = None
    if full_mask == initial_mask:
        min_slots_to_full = 0
    else:
        # The state table has at most `slots` layers; find the first exact cover
        # by replaying the same bounded recurrence with a minimum-cardinality DP.
        minimum: dict[int, int] = {initial_mask: 0}
        for step in range(1, len(candidates) + 1):
            updated = dict(minimum)
            for mask in minimum:
                for memory_id in candidates:
                    updated[mask | group_by_memory[memory_id]] = min(
                        updated.get(mask | group_by_memory[memory_id], step), step
                    )
            minimum = updated
            if full_mask in minimum:
                min_slots_to_full = minimum[full_mask]
                break
    if min_slots_to_full is None:
        min_slots_to_full = -1
    return best_mask, initial_mask, best_chosen, int(min_slots_to_full), len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the train-OOF gold coverage oracle without deployment access.")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("gold oracle run ID is immutable")
    data = load_config("v21_data.json")
    ann_path = resolve(data["frozen_baseline"]["train_retrieval"])
    rows_by_fold = load_train_rows(data)
    ann_prefix = load_ann_prefix(ann_path)
    rows: list[dict] = []
    aggregate: dict[int, dict[str, float]] = defaultdict(lambda: {"rows": 0.0, "group_coverage": 0.0, "all_groups_covered": 0.0, "budget_infeasible": 0.0, "residual_groups": 0.0, "protected_covered_groups": 0.0})
    per_seed_fold: dict[tuple[int, int, int], dict[str, float]] = defaultdict(lambda: {"rows": 0.0, "group_coverage": 0.0, "all_groups_covered": 0.0, "budget_infeasible": 0.0})
    for seed in (20260712, 20260713, 20260714, 20260715, 20260716):
        for fold in range(5):
            for row in rows_by_fold[fold]:
                eligible = set(row["eligible_memory_ids"])
                protected = tuple(memory_id for memory_id in ann_prefix[row["query_id"]] if memory_id in eligible)[:PROTECTED_SEEDS]
                if len(protected) != PROTECTED_SEEDS:
                    raise ValueError(f"protected ANN seeds are not eligible: {row['query_id']}")
                groups = row["groups"]
                protected_mask = 0
                for index, group in enumerate(groups):
                    if set(protected).intersection(group):
                        protected_mask |= 1 << index
                residual_groups = len(groups) - protected_mask.bit_count()
                residual_members = sorted({memory_id for index, group in enumerate(groups) if not (protected_mask & (1 << index)) for memory_id in group if memory_id not in protected})
                for budget in BUDGETS:
                    best_mask, initial_mask, chosen, min_slots, candidate_count = _gold_masks(protected, groups, budget)
                    group_count = len(groups)
                    coverage = float(best_mask.bit_count() / group_count) if group_count else 1.0
                    all_covered = int(best_mask == (1 << group_count) - 1)
                    infeasible = int(not all_covered)
                    record = {
                        "run_id": RUN_ID, "seed": seed, "fold": fold,
                        "query_id": row["query_id"], "persona_id": row["persona_id"], "query_session_id": row["query_session_id"],
                        "budget": budget, "group_count": group_count,
                        "protected_seed_count": len(protected), "protected_covered_groups": initial_mask.bit_count(),
                        "residual_group_count": residual_groups, "residual_group_member_count": len(residual_members),
                        "gold_candidate_member_count": candidate_count, "minimum_additional_members_for_full_coverage": min_slots,
                        "gold_max_groups_covered": best_mask.bit_count(), "group_coverage": coverage,
                        "all_groups_covered": all_covered, "budget_infeasible": infeasible,
                        "selected_memory_ids": json.dumps(list(protected) + list(chosen), ensure_ascii=False),
                        "calibration_read": False, "formal_test_read": False,
                    }
                    rows.append(record)
                    bucket = aggregate[budget]
                    bucket["rows"] += 1; bucket["group_coverage"] += coverage; bucket["all_groups_covered"] += all_covered; bucket["budget_infeasible"] += infeasible; bucket["residual_groups"] += residual_groups; bucket["protected_covered_groups"] += initial_mask.bit_count()
                    sf = per_seed_fold[(seed, fold, budget)]
                    sf["rows"] += 1; sf["group_coverage"] += coverage; sf["all_groups_covered"] += all_covered; sf["budget_infeasible"] += infeasible

    metrics_path = VERSION_ROOT / "aggregate" / f"v21_gold_coverage_oracle_metrics_{OUTPUT_REVISION}.csv"
    write_csv_new(metrics_path, rows)
    pooled = {}
    for budget, bucket in sorted(aggregate.items()):
        count = max(bucket["rows"], 1.0)
        pooled[str(budget)] = {"rows": int(bucket["rows"]), "group_coverage_mean": bucket["group_coverage"] / count, "all_groups_covered_rate": bucket["all_groups_covered"] / count, "budget_infeasible_rate": bucket["budget_infeasible"] / count, "residual_group_count_mean": bucket["residual_groups"] / count, "protected_covered_groups_mean": bucket["protected_covered_groups"] / count}
    per_fold = {}
    for (seed, fold, budget), bucket in sorted(per_seed_fold.items()):
        count = max(bucket["rows"], 1.0)
        per_fold[f"{seed}|{fold}|{budget}"] = {"rows": int(bucket["rows"]), "group_coverage_mean": bucket["group_coverage"] / count, "all_groups_covered_rate": bucket["all_groups_covered"] / count, "budget_infeasible_rate": bucket["budget_infeasible"] / count}
    summary = {
        "schema_version": "v21-gold-coverage-oracle-1", "status": "completed_non_deployable", "run_id": RUN_ID,
        "split": "train_query_plan_only; heldout outer session-blocked folds", "units": 25, "queries_per_seed": sum(len(rows) for rows in rows_by_fold.values()),
        "protected_ann_seeds": PROTECTED_SEEDS, "budgets": list(BUDGETS), "pooled": pooled, "per_seed_fold": per_fold,
        "contract": {"uses_gold_evidence": True, "deployment_allowed": False, "calibration_read": False, "formal_test_read": False, "no_calibration_or_formal_queries": True},
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT))},
        "hashes": {"metrics": sha256_file(metrics_path), "data_config": sha256_file(VERSION_ROOT / "configs" / "v21_data.json"), "ann_prefix": sha256_file(ann_path)},
        "calibration_read": False, "formal_test_read": False,
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_gold_coverage_oracle_summary_{OUTPUT_REVISION}.json"
    write_json_new(summary_path, summary)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_ORACLE_HEADROOM_RESULTS.md"
    report = "# Method 1 v21 Oracle Headroom Results\n\n"
    report += "The gold-coverage oracle is a non-deployable train-query OOF upper-bound analysis. It may read heldout train evidence groups within the five session-blocked folds, but it never reads calibration or formal-test queries. Protected ANN seeds are mandatory in every budget.\n\n"
    report += "| B | Group coverage | All groups covered | Budget-infeasible rate | Residual groups after protected seeds |\n| ---: | ---: | ---: | ---: | ---: |\n"
    for budget, values in pooled.items():
        report += f"| {budget} | {values['group_coverage_mean']:.6f} | {values['all_groups_covered_rate']:.6f} | {values['budget_infeasible_rate']:.6f} | {values['residual_group_count_mean']:.3f} |\n"
    report += "\nThis oracle reports headroom only and is not an online method, a retrieval claim, or a deployable selector.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "metrics_sha256": summary["hashes"]["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
