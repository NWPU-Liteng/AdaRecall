#!/usr/bin/env python3
"""Assemble all v5.1 motivation evidence into one long-form CSV."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT.parents[0] / "20260715_method1_motivation_medmemorybench_v5_alignment"
V3 = ROOT.parents[0] / "20260704_method1_motivation_medmemorybench_v3"

FIELDS = [
    "version", "experiment", "data_role", "query_type", "condition",
    "history_scale", "frontier_budget", "tile_capacity", "layout",
    "growth_percent", "metric", "value", "ci_low", "ci_high", "n",
    "common_query_count", "seed", "source_file", "claim_role",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add(rows: list[dict[str, Any]], *, experiment: str, data_role: str,
        query_type: str = "all", condition: str = "", history_scale: Any = "",
        frontier_budget: Any = "", tile_capacity: Any = "", layout: str = "",
        growth_percent: Any = "", metric: str, value: Any, ci_low: Any = "",
        ci_high: Any = "", n: Any = "", common_query_count: Any = "",
        seed: Any = "", source_file: str, claim_role: str) -> None:
    rows.append({
        "version": "20260716_method1_motivation_medmemorybench_v5p1_controlled",
        "experiment": experiment,
        "data_role": data_role,
        "query_type": query_type,
        "condition": condition,
        "history_scale": history_scale,
        "frontier_budget": frontier_budget,
        "tile_capacity": tile_capacity,
        "layout": layout,
        "growth_percent": growth_percent,
        "metric": metric,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "common_query_count": common_query_count,
        "seed": seed,
        "source_file": source_file,
        "claim_role": claim_role,
    })


def _weighted(rows: list[dict[str, str]], value_key: str) -> tuple[float, int]:
    weights = [int(float(row.get("n", 0))) for row in rows]
    values = [float(row[value_key]) for row in rows]
    total = sum(weights)
    return (sum(v * w for v, w in zip(values, weights)) / total, total)


def _bootstrap_ci(values: list[float], seed: int = 20260716, reps: int = 1000) -> tuple[float, float]:
    if not values:
        return "", ""
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(reps)]
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def main() -> None:
    out = ROOT / "aggregate" / "v5p1_motivation_final_data.csv"
    rows: list[dict[str, Any]] = []

    # M1: controlled, same query IDs in every cell.  The summary already
    # carries bootstrap intervals from the query-level invocation rows.
    m1_path = ROOT / "aggregate" / "M1_controlled_summary.csv"
    m1_rows = read_csv(m1_path)
    m1_map = {
        "t_total_selection_ms": "selection_latency_ms",
        "t_query_encode_ms": "query_encode_latency_ms",
        "t_ann_ms": "ann_latency_ms",
        "t_rerank_ms": "rerank_latency_ms",
        "host_memory_bytes": "host_bytes_touched",
        "metadata_bytes_read": "metadata_bytes",
        "payload_bytes_read": "payload_bytes",
        "index_bytes_read": "candidate_vector_bytes",
        "n_reranked": "n_reranked",
        "frontier_fill_ratio": "frontier_fill_ratio",
    }
    # n_reranked is not in the summary; it is added below from invocations.
    for source_metric, metric in m1_map.items():
        for row in m1_rows:
            mean_key = f"{source_metric}_mean"
            if mean_key not in row or row[mean_key] == "":
                continue
            add(
                rows,
                experiment="M1_controlled_history_pressure",
                data_role="controlled_observation",
                condition=row["history_condition"],
                history_scale=row["history_scale"],
                frontier_budget=row["frontier_budget"],
                metric=metric,
                value=row[mean_key],
                ci_low=row.get(f"{source_metric}_ci95_low", ""),
                ci_high=row.get(f"{source_metric}_ci95_high", ""),
                n=row["n"],
                common_query_count=row["common_query_count"],
                seed=row["seed"],
                source_file=str(m1_path.relative_to(ROOT.parent.parent)),
                claim_role="supporting_motivation" if metric == "host_bytes_touched" else "main_motivation",
            )
    # Query-level n_reranked is a useful occupancy audit, even though the
    # aggregate summary does not expose it.
    inv_path = ROOT / "aggregate" / "M1_controlled_invocations.jsonl"
    inv_rows = [json.loads(line) for line in inv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Primary cross-dataset metric: pair each scale with the same query's
    # 128-memory baseline. This removes dependence on embedding dimension,
    # payload serialization, and absolute corpus size.
    for condition in sorted({str(row["history_condition"]) for row in inv_rows}):
        base = {
            str(row["query_id"]): float(row["host_memory_bytes"])
            for row in inv_rows
            if str(row["history_condition"]) == condition and int(row["history_scale"]) == 128
        }
        for scale in (128, 256, 512):
            ratios = [
                float(row["host_memory_bytes"]) / base[str(row["query_id"])]
                for row in inv_rows
                if str(row["history_condition"]) == condition
                and int(row["history_scale"]) == scale
                and str(row["query_id"]) in base
            ]
            ci_low, ci_high = _bootstrap_ci(ratios)
            add(
                rows,
                experiment="M1_controlled_history_pressure",
                data_role="controlled_observation",
                condition=condition,
                history_scale=scale,
                frontier_budget=32,
                metric="host_traffic_amplification_vs_128",
                value=sum(ratios) / len(ratios),
                ci_low=ci_low,
                ci_high=ci_high,
                n=len(ratios),
                common_query_count=81,
                seed=20260716,
                source_file=str(inv_path.relative_to(ROOT.parent.parent)),
                claim_role="main_motivation",
            )
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in inv_rows:
        grouped[(str(row["history_condition"]), int(row["history_scale"]))].append(float(row["n_reranked"]))
    for (condition, scale), values in sorted(grouped.items()):
        add(
            rows,
            experiment="M1_controlled_history_pressure",
            data_role="controlled_observation",
            condition=condition,
            history_scale=scale,
            frontier_budget=32,
            metric="n_reranked",
            value=sum(values) / len(values),
            n=len(values),
            common_query_count=81,
            seed=20260716,
            source_file=str(inv_path.relative_to(ROOT.parent.parent)),
            claim_role="main_motivation",
        )

    # S1: evidence structure and flat-cosine coverage.  It is a workload
    # diagnostic, not a comparison against the formal method.
    s1_path = V5 / "aggregate/v5_s1_evidence_structure_summary.csv"
    s1_per_query_path = V5 / "aggregate/v5_s1_evidence_structure_per_query.csv"
    s1_rows = read_csv(s1_path)
    s1_per_query = read_csv(s1_per_query_path)
    multi_counts: dict[str, tuple[int, int]] = {}
    for qtype in sorted({row["query_type"] for row in s1_per_query} | {"all"}):
        subset = s1_per_query if qtype == "all" else [row for row in s1_per_query if row["query_type"] == qtype]
        multi_counts[qtype] = (sum(int(float(row["evidence_group_count"])) > 1 for row in subset), len(subset))
    for row in s1_rows:
        qtype = row["query_type"]
        n = row["n"]
        common = n
        for source_metric, metric, budget in [
            ("evidence_group_count_mean", "evidence_group_count_mean", ""),
            ("acceptable_memory_count_mean", "acceptable_memory_count_mean", ""),
            ("acceptable_top2_cosine_margin_mean_mean", "acceptable_top2_cosine_margin_mean", ""),
            ("flat_group_coverage_b8_mean", "group_coverage_at_8", 8),
            ("flat_all_groups_b8_rate", "all_groups_covered_at_8", 8),
            ("flat_redundancy_b8_mean", "flat_redundancy_at_8", 8),
        ]:
            add(
                rows,
                experiment="S1_evidence_structure",
                data_role="workload_structure",
                query_type=qtype,
                condition="flat_cosine_reference" if budget else "train_evidence_mapping",
                frontier_budget=budget,
                metric=metric,
                value=row[source_metric],
                n=n,
                common_query_count=common,
                seed="",
                source_file=str(s1_path.relative_to(ROOT.parent.parent)),
                claim_role="supporting_motivation",
            )
        multi, total = multi_counts[qtype]
        add(
            rows,
            experiment="S1_evidence_structure",
            data_role="workload_structure",
            query_type=qtype,
            condition="train_evidence_mapping",
            metric="multi_group_fraction",
            value=multi / total if total else "",
            n=total,
            common_query_count=total,
            source_file=str(s1_per_query_path.relative_to(ROOT.parent.parent)),
            claim_role="supporting_motivation",
        )

    # M2: weighted across the same 240-query flat-frontier workload.  The
    # reference-overlap metric is explicitly the reranker-at-64 oracle overlap,
    # not a task answer-quality score.
    m2_path = V5 / "aggregate/v5_reused_m2_source.csv"
    m2_rows = read_csv(m2_path)
    for budget in sorted({int(row["frontier_budget"]) for row in m2_rows}):
        subset = [row for row in m2_rows if int(row["frontier_budget"]) == budget]
        for source_metric, metric in [
            ("t_total_selection_ms_mean", "selection_latency_ms"),
            ("host_memory_bytes_mean", "host_bytes_touched"),
            ("reranker_oracle_overlap_at_5_mean", "reference_overlap_at_5"),
        ]:
            value, n = _weighted(subset, source_metric)
            add(
                rows,
                experiment="M2_flat_frontier_tradeoff",
                data_role="reused_observation",
                condition="full_history_flat_retrieval",
                frontier_budget=budget,
                metric=metric,
                value=value,
                n=n,
                common_query_count=240,
                seed=20260704,
                source_file=str(m2_path.relative_to(ROOT.parent.parent)),
                claim_role="main_motivation",
            )

    # M3: Tile scattering under chronological/random placement plus the oracle
    # packing lower bound as a reference, all independent of Method 1 output.
    m3_path = V5 / "aggregate/v5_reused_m3_tile_summary.csv"
    for row in read_csv(m3_path):
        for source_metric, metric in [
            ("activated_tiles_mean", "activated_tiles"),
            ("access_amplification_mean", "access_amplification"),
            ("tile_utilization_mean", "tile_utilization"),
            ("scatter_ratio_mean", "scatter_ratio"),
        ]:
            add(
                rows,
                experiment="M3_tile_scattering",
                data_role="reused_observation" if row["layout"] != "oracle_tile_packing_lower_bound" else "reference_baseline",
                condition="flat_retrieval_selected_memories",
                tile_capacity=row["tile_capacity"],
                layout=row["layout"],
                metric=metric,
                value=row[source_metric],
                n=row["n"],
                common_query_count=240,
                seed=20260704,
                source_file=str(m3_path.relative_to(ROOT.parent.parent)),
                claim_role="main_motivation" if row["layout"] != "oracle_tile_packing_lower_bound" else "supporting_motivation",
            )

    # S2 is appendix-only: generic static content insertion maintenance.
    s2_path = V5 / "aggregate/v5_s2_generic_content_maintenance.csv"
    s2_metrics = [
        "base_memory_count", "appended_memory_count", "active_memory_count",
        "base_postings", "changed_postings", "full_rebuild_postings",
        "changed_posting_fraction_of_full", "incremental_update_ms", "full_rebuild_ms",
        "full_to_incremental_time_ratio", "lookup_mismatch_count",
        "incremental_probe_activated_tiles_mean", "full_rebuild_probe_activated_tiles_mean",
        "new_payload_tile_count",
    ]
    for row in read_csv(s2_path):
        for metric in s2_metrics:
            add(
                rows,
                experiment="S2_generic_content_maintenance",
                data_role="appendix_stress_test",
                condition="append_only_static_index",
                growth_percent=row["growth_percent_of_full_corpus"],
                tile_capacity=row["tile_capacity"],
                metric=metric,
                value=row[metric],
                n=1,
                common_query_count=row["probe_count"],
                seed="",
                source_file=str(s2_path.relative_to(ROOT.parent.parent)),
                claim_role="appendix_only",
            )

    # Feasibility audit rows make the cohort-selection decision machine-checkable
    # in the same artifact without turning it into a motivation claim.
    audit_path = ROOT / "aggregate/v5p1_m1_common_query_ids.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for scale, count in sorted(audit["common_feasible_counts"].items(), key=lambda item: int(item[0])):
        add(
            rows,
            experiment="M1_feasibility_audit",
            data_role="audit",
            condition="both_conditions",
            history_scale=scale,
            metric="common_feasible_queries",
            value=count,
            n=1,
            common_query_count=audit["source_query_count"],
            seed=20260716,
            source_file=str(audit_path.relative_to(ROOT.parent.parent)),
            claim_role="audit_only",
        )
    add(
        rows,
        experiment="M1_feasibility_audit",
        data_role="audit",
        condition="frozen_cohort",
        metric="selected_query_count",
        value=audit["selected_query_count"],
        n=1,
        common_query_count=audit["selected_query_count"],
        seed=20260716,
        source_file=str(audit_path.relative_to(ROOT.parent.parent)),
        claim_role="audit_only",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT / "aggregate/v5p1_motivation_final_data_manifest.json").write_text(
        json.dumps({"schema_version": "v5p1-motivation-long-1", "rows": len(rows), "sha256": digest, "path": str(out)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "sha256": digest, "path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
