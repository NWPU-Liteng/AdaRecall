#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGES = ["ground", "plan", "reason", "reflect"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def split_manifest(rows: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    by_persona = defaultdict(list)
    query_meta = {}
    for row in rows:
        qid = str(row["query_id"])
        if qid not in query_meta:
            query_meta[qid] = row
            by_persona[str(row["persona_id"])].append(row)
    period = {}
    manifest = []
    for persona, items in by_persona.items():
        items = sorted(items, key=lambda r: (int(r.get("query_time_order", 0)), str(r["query_id"])))
        split_idx = max(1, int(math.floor(len(items) * 0.6)))
        if split_idx >= len(items) and len(items) > 1:
            split_idx = len(items) - 1
        train = items[:split_idx]
        test = items[split_idx:]
        for row in train:
            period[str(row["query_id"])] = "train"
        for row in test:
            period[str(row["query_id"])] = "test"
        def session_bounds(items: list[dict[str, Any]]) -> tuple[str, str]:
            if not items:
                return "", ""
            values = sorted([str(r.get("query_session_id")) for r in items], key=lambda v: (0, int(float(v))) if str(v).replace(".", "", 1).isdigit() else (1, str(v)))
            return values[0], values[-1]

        train_session_min, train_session_max = session_bounds(train)
        test_session_min, test_session_max = session_bounds(test)
        manifest.append(
            {
                "persona_id": persona,
                "n_queries": len(items),
                "n_train": len(train),
                "n_test": len(test),
                "train_order_min": min([int(r.get("query_time_order", 0)) for r in train], default=""),
                "train_order_max": max([int(r.get("query_time_order", 0)) for r in train], default=""),
                "test_order_min": min([int(r.get("query_time_order", 0)) for r in test], default=""),
                "test_order_max": max([int(r.get("query_time_order", 0)) for r in test], default=""),
                "train_session_min": train_session_min,
                "train_session_max": train_session_max,
                "test_session_min": test_session_min,
                "test_session_max": test_session_max,
            }
        )
    return period, manifest


def build_graphs(rows: list[dict[str, Any]]) -> tuple[dict[str, Counter], Counter, set[str]]:
    stage_graphs = {stage: Counter() for stage in STAGES}
    global_graph = Counter()
    nodes = set()
    for row in rows:
        stage = str(row["stage"])
        ids = sorted(set(str(x) for x in row.get("selected_memory_ids") or row.get("final_selected_ids") or []))
        nodes.update(ids)
        for pair in itertools.combinations(ids, 2):
            if stage in stage_graphs:
                stage_graphs[stage][pair] += 1
            global_graph[pair] += 1
    return stage_graphs, global_graph, nodes


def cohesion(ids: list[str], graph: Counter, train_nodes: set[str]) -> tuple[float, int]:
    pairs = [tuple(sorted(pair)) for pair in itertools.combinations(sorted(set(ids)), 2) if pair[0] in train_nodes and pair[1] in train_nodes]
    if not pairs:
        return 0.0, 0
    return float(mean([math.log1p(graph.get(pair, 0)) for pair in pairs])), len(pairs)


def compute_per_query(rows: list[dict[str, Any]], period: dict[str, str]) -> list[dict[str, Any]]:
    train_rows = [r for r in rows if period.get(str(r["query_id"])) == "train"]
    test_rows = [r for r in rows if period.get(str(r["query_id"])) == "test"]
    stage_graphs, global_graph, train_nodes = build_graphs(train_rows)
    output = []
    for row in test_rows:
        stage = str(row["stage"])
        ids = [str(x) for x in row.get("selected_memory_ids") or row.get("final_selected_ids") or []]
        seen = sum(1 for mid in ids if mid in train_nodes)
        seen_cov = seen / float(len(ids)) if ids else 0.0
        stage_coh, eligible_pairs = cohesion(ids, stage_graphs.get(stage, Counter()), train_nodes)
        global_coh, _ = cohesion(ids, global_graph, train_nodes)
        cross_vals = []
        for other in STAGES:
            if other == stage:
                continue
            value, _ = cohesion(ids, stage_graphs.get(other, Counter()), train_nodes)
            cross_vals.append(value)
        cross_coh = float(mean(cross_vals)) if cross_vals else 0.0
        output.append(
            {
                "query_id": row["query_id"],
                "persona_id": row["persona_id"],
                "query_type": row.get("query_type", ""),
                "query_session_id": row.get("query_session_id", ""),
                "query_time_order": row.get("query_time_order", 0),
                "stage": stage,
                "seen_node_coverage": seen_cov,
                "eligible_pair_count": eligible_pairs,
                "stage_cohesion": stage_coh,
                "global_cohesion": global_coh,
                "cross_stage_cohesion": cross_coh,
                "stage_advantage": stage_coh - cross_coh,
            }
        )
    return output


def summarize(per_query: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for label in STAGES + ["aggregate"]:
        sub = per_query if label == "aggregate" else [r for r in per_query if r["stage"] == label]
        advantages = [float(r["stage_advantage"]) for r in sub]
        coverages = [float(r["seen_node_coverage"]) for r in sub]
        boot = []
        if advantages:
            for _ in range(bootstrap_samples):
                sample = [advantages[rng.randrange(len(advantages))] for _ in advantages]
                boot.append(float(mean(sample)))
        out.append(
            {
                "stage": label,
                "n_test_invocations": len(sub),
                "seen_node_coverage_mean": float(mean(coverages)) if coverages else 0.0,
                "stage_advantage_mean": float(mean(advantages)) if advantages else 0.0,
                "stage_advantage_ci95_low": percentile(boot, 0.025),
                "stage_advantage_ci95_high": percentile(boot, 0.975),
                "bootstrap_samples": bootstrap_samples,
            }
        )
    return out


def shuffle_null(rows: list[dict[str, Any]], period: dict[str, str], permutations: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    observed = compute_per_query(rows, period)
    obs_agg = mean([float(r["stage_advantage"]) for r in observed]) if observed else 0.0
    null_rows = []
    values = []
    for idx in range(permutations):
        shuffled = [dict(r) for r in rows]
        groups = defaultdict(list)
        for i, row in enumerate(shuffled):
            groups[(str(row["persona_id"]), period.get(str(row["query_id"]), ""))].append(i)
        for indices in groups.values():
            labels = [shuffled[i]["stage"] for i in indices]
            rng.shuffle(labels)
            for i, label in zip(indices, labels):
                shuffled[i]["stage"] = label
        pq = compute_per_query(shuffled, period)
        stat = mean([float(r["stage_advantage"]) for r in pq]) if pq else 0.0
        values.append(stat)
        null_rows.append({"permutation": idx, "statistic": stat})
    sorted_values = sorted(values)
    p = (sum(1 for v in values if v >= obs_agg) + 1) / float(len(values) + 1)
    null_mean = float(mean(values)) if values else 0.0
    null_std = float(np.std(values)) if values else 0.0
    effect = (obs_agg - null_mean) / null_std if null_std > 0 else 0.0
    for row in null_rows:
        row.update(
            {
                "observed_statistic": obs_agg,
                "null_mean": null_mean,
                "null_ci95_low": percentile(sorted_values, 0.025),
                "null_ci95_high": percentile(sorted_values, 0.975),
                "empirical_p_value": p,
                "standardized_effect_size": effect,
                "num_permutations": permutations,
            }
        )
    return null_rows


def report(summary: list[dict[str, Any]], null_rows: list[dict[str, Any]]) -> None:
    agg = next((r for r in summary if r["stage"] == "aggregate"), {})
    null = null_rows[0] if null_rows else {}
    lines = [
        "# B Held-Out Stage Transfer",
        "",
        "Chronological split is performed independently per persona: earliest 60% train, latest 40% held-out test.",
        "",
        "## Summary",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['stage']}`: n={row['n_test_invocations']}, seen-node coverage={row['seen_node_coverage_mean']:.3f}, stage advantage={row['stage_advantage_mean']:.4f}, 95% bootstrap CI=[{row['stage_advantage_ci95_low']:.4f}, {row['stage_advantage_ci95_high']:.4f}]"
        )
    lines.extend(
        [
            "",
            "## Stage-Shuffle Null",
            "",
            f"- observed aggregate statistic: {float(null.get('observed_statistic', 0.0)):.4f}",
            f"- null mean: {float(null.get('null_mean', 0.0)):.4f}",
            f"- null 95% interval: [{float(null.get('null_ci95_low', 0.0)):.4f}, {float(null.get('null_ci95_high', 0.0)):.4f}]",
            f"- empirical p-value: {float(null.get('empirical_p_value', 1.0)):.4f}",
            f"- standardized effect size: {float(null.get('standardized_effect_size', 0.0)):.4f}",
            "",
            "## Guardrail Interpretation",
            "",
        ]
    )
    if float(agg.get("seen_node_coverage_mean", 0.0)) < 0.30:
        lines.append("Held-out seen-node coverage is below 30%; exact-ID transfer is sparse and a strong offline-layout-stability claim is not supported.")
    if float(null.get("empirical_p_value", 1.0)) >= 0.05 or float(agg.get("stage_advantage_mean", 0.0)) <= float(null.get("null_mean", 0.0)):
        lines.append("The aggregate stage advantage is not above the shuffled-stage null at p < 0.05; v4 does not establish future stage transfer.")
    else:
        lines.append("Earlier controller-defined stage-conditioned access structure shows measurable transfer to later held-out accesses.")
    stage_adv = {r["stage"]: float(r["stage_advantage_mean"]) for r in summary if r["stage"] != "aggregate"}
    if stage_adv and max(stage_adv, key=stage_adv.get) == "plan":
        lines.append("Transfer is concentrated in planning-stage retrieval.")
    if stage_adv.get("reflect", 0.0) <= 0:
        lines.append("Reflect is weak or negative and should be treated as a dynamic/update-sensitive regime rather than evidence for static structural placement.")
    (ROOT / "reports" / "B_HELDOUT_STAGE_TRANSFER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute B held-out stage transfer.")
    parser.add_argument("--trace", default="aggregate/B_stage_trace_v4.jsonl")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()
    rows = [r for r in read_jsonl(ROOT / args.trace) if not r.get("error")]
    period, manifest = split_manifest(rows)
    write_csv(ROOT / "aggregate" / "B_chronological_split_manifest.csv", manifest)
    per_query = compute_per_query(rows, period)
    write_csv(ROOT / "aggregate" / "B_heldout_stage_transfer_per_query.csv", per_query)
    summary = summarize(per_query, args.bootstrap_samples, args.seed)
    write_csv(ROOT / "aggregate" / "B_heldout_stage_transfer_summary.csv", summary)
    null_rows = shuffle_null(rows, period, args.permutations, args.seed)
    write_csv(ROOT / "aggregate" / "B_heldout_stage_shuffle_null.csv", null_rows)
    report(summary, null_rows)
    print("b_transfer=complete")


if __name__ == "__main__":
    main()
