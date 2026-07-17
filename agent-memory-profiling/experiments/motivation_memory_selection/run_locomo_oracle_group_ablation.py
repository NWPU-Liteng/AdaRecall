#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT
    / "versions"
    / "20260621_real_trace_adaptive_phase_v4_formal"
    / "manifests"
    / "locomo"
)
DEFAULT_PHASE_TRACE = (
    PROJECT_ROOT
    / "versions"
    / "20260621_real_trace_adaptive_phase_v4_formal"
    / "runs"
    / "locomo_official_qa_full_shared_once_controller_lite_selection_only"
    / "phase_trace.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "versions" / "v0.4" / "locomo_oracle_group_ablation"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def paired_bootstrap_ci(deltas: list[float], *, seed: int, rounds: int = 2000) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    samples = []
    for _ in range(rounds):
        samples.append(mean([deltas[rng.randrange(n)] for _idx in range(n)]))
    samples.sort()
    return {
        "mean": mean(deltas),
        "ci_low": samples[int(0.025 * (len(samples) - 1))],
        "ci_high": samples[int(0.975 * (len(samples) - 1))],
    }


def sign_test_p_value(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def build_train_structure(groups: list[set[str]]):
    pairwise_edges: dict[tuple[str, str], int] = defaultdict(int)
    canonical_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for group in groups:
        members = tuple(sorted(group))
        if len(members) < 2:
            continue
        canonical_counts[members] += 1
        for idx, left in enumerate(members):
            for right in members[idx + 1 :]:
                pairwise_edges[(left, right)] += 1
    hyperedges = [(members, count) for members, count in canonical_counts.items()]
    adjacency: dict[str, dict[str, int]] = defaultdict(dict)
    incident: dict[str, list[tuple[tuple[str, ...], int]]] = defaultdict(list)
    for (left, right), weight in pairwise_edges.items():
        adjacency[left][right] = weight
        adjacency[right][left] = weight
    for members, weight in hyperedges:
        for memory_id in members:
            incident[memory_id].append((members, weight))
    return dict(pairwise_edges), dict(adjacency), dict(incident), hyperedges


def compose_hybrid(*, dense_anchor: list[str], ranked: list[str], budget: int) -> list[str]:
    out: list[str] = []
    seen = set()
    for memory_id in dense_anchor:
        if memory_id not in seen:
            seen.add(memory_id)
            out.append(memory_id)
        if len(out) >= budget:
            return out
    for memory_id in ranked:
        if memory_id not in seen:
            seen.add(memory_id)
            out.append(memory_id)
        if len(out) >= budget:
            return out
    return out


def rank_pairwise(*, seeds: list[str], visible: set[str], adjacency: dict[str, dict[str, int]], dense_rank: dict[str, int]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for seed in seeds:
        for candidate, weight in adjacency.get(seed, {}).items():
            if candidate in visible and candidate not in seeds:
                scores[candidate] += float(weight)
    return sorted(scores, key=lambda mid: (-scores[mid], dense_rank.get(mid, 10**9), mid))


def rank_hypergraph(
    *,
    seeds: list[str],
    visible: set[str],
    incident: dict[str, list[tuple[tuple[str, ...], int]]],
    dense_rank: dict[str, int],
    require_multi_seed: bool,
) -> tuple[list[str], dict[str, float]]:
    seed_set = set(seeds)
    activated: dict[tuple[str, ...], tuple[int, int]] = {}
    for seed in seeds:
        for members, weight in incident.get(seed, []):
            overlap = len(seed_set & set(members))
            if overlap <= 0:
                continue
            if require_multi_seed and overlap < 2:
                continue
            previous = activated.get(members)
            if previous is None or overlap > previous[1]:
                activated[members] = (weight, overlap)
    scores: dict[str, float] = defaultdict(float)
    for members, (weight, overlap) in activated.items():
        members_set = set(members)
        edge_size = max(2, len(members))
        denom = max(1, min(len(seed_set), edge_size - 1))
        gain = (overlap / denom) ** 2
        for candidate in members_set:
            if candidate in visible and candidate not in seed_set:
                scores[candidate] += float(weight) * gain * math.log2(edge_size + 1)
    return (
        sorted(scores, key=lambda mid: (-scores[mid], dense_rank.get(mid, 10**9), mid)),
        {
            "activated_hyperedges": float(len(activated)),
            "multi_seed_activated_hyperedges": float(sum(1 for _members, (_weight, overlap) in activated.items() if overlap >= 2)),
        },
    )


def recall(selected: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(set(selected) & gold) / len(gold)


def full_group_hit(selected: list[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return 1.0 if gold.issubset(set(selected)) else 0.0


def load_dense_phase_trace(path: Path) -> dict[str, dict[str, Any]]:
    planning_by_event: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("phase") != "planning":
            continue
        event_id = str(row["event_id"])
        planning_by_event[event_id] = row
    return planning_by_event


def run_experiment(args: argparse.Namespace) -> list[dict[str, Any]]:
    labels = list(iter_jsonl(args.manifest_dir / "evaluation_labels.jsonl"))
    runtime_rows = {row["event_id"]: row for row in iter_jsonl(args.manifest_dir / "runtime_manifest.jsonl")}
    planning_by_event = load_dense_phase_trace(args.phase_trace)

    labels_by_tenant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        tenant_id = str(row["event_id"]).split("::qa::", 1)[0]
        labels_by_tenant[tenant_id].append(row)

    rows: list[dict[str, Any]] = []
    for tenant_id, tenant_labels in sorted(labels_by_tenant.items()):
        tenant_labels = sorted(tenant_labels, key=lambda row: row["event_id"])
        gold_groups = {row["event_id"]: set(row.get("gold_turn_ids") or []) for row in tenant_labels}
        for label in tenant_labels:
            event_id = label["event_id"]
            planning = planning_by_event.get(event_id)
            runtime = runtime_rows.get(event_id)
            if planning is None or runtime is None:
                continue
            visible = set(runtime.get("memory_ids_available") or [])
            dense_order = [mid for mid in planning.get("dense_top_r_ids") or [] if mid in visible]
            dense_rank = {mid: idx for idx, mid in enumerate(dense_order)}
            gold = gold_groups[event_id]
            train_groups = [group for other_event_id, group in gold_groups.items() if other_event_id != event_id and group]
            _pairwise_edges, adjacency, incident, hyperedges = build_train_structure(train_groups)
            for seed_count in args.seed_counts:
                seeds = dense_order[:seed_count]
                if not seeds:
                    continue
                pairwise_ranked = rank_pairwise(seeds=seeds, visible=visible, adjacency=adjacency, dense_rank=dense_rank)
                hyper_ranked, hyper_diag = rank_hypergraph(
                    seeds=seeds,
                    visible=visible,
                    incident=incident,
                    dense_rank=dense_rank,
                    require_multi_seed=False,
                )
                strict_ranked, strict_diag = rank_hypergraph(
                    seeds=seeds,
                    visible=visible,
                    incident=incident,
                    dense_rank=dense_rank,
                    require_multi_seed=True,
                )
                for budget in args.candidate_budgets:
                    anchor_count = min(budget, max(seed_count, int(math.ceil(budget * args.anchor_multiplier))))
                    dense_anchor = dense_order[:anchor_count]
                    outputs = {
                        "dense_prefix": (dense_order[:budget], {}),
                        "pairwise_oracle_expand": (
                            compose_hybrid(dense_anchor=dense_anchor, ranked=pairwise_ranked, budget=budget),
                            {},
                        ),
                        "hypergraph_oracle_expand": (
                            compose_hybrid(dense_anchor=dense_anchor, ranked=hyper_ranked, budget=budget),
                            hyper_diag,
                        ),
                        "hypergraph_oracle_multi_seed": (
                            compose_hybrid(dense_anchor=dense_anchor, ranked=strict_ranked, budget=budget),
                            strict_diag,
                        ),
                    }
                    for method, (selected, diag) in outputs.items():
                        rows.append(
                            {
                                "tenant_id": tenant_id,
                                "event_id": event_id,
                                "method": method,
                                "seed_count": seed_count,
                                "candidate_budget": budget,
                                "gold_size": len(gold),
                                "train_group_count": len(train_groups),
                                "train_hyperedge_count": len(hyperedges),
                                "selected_ids": selected,
                                "evidence_recall": recall(selected, gold),
                                "full_group_hit": full_group_hit(selected, gold),
                                **diag,
                            }
                        )
    return rows


def build_comparison_rows(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_key = {(row["event_id"], row["method"], int(row["seed_count"]), int(row["candidate_budget"])): row for row in rows}
    event_ids = sorted({row["event_id"] for row in rows})
    methods = ["pairwise_oracle_expand", "hypergraph_oracle_expand", "hypergraph_oracle_multi_seed"]
    subsets = {
        "all": lambda row: True,
        "gold_ge_2": lambda row: int(row["gold_size"]) >= 2,
        "gold_ge_3": lambda row: int(row["gold_size"]) >= 3,
    }
    out: list[dict[str, Any]] = []
    for subset_name, predicate in subsets.items():
        for seed_count in sorted({int(row["seed_count"]) for row in rows}):
            for budget in sorted({int(row["candidate_budget"]) for row in rows}):
                eligible_event_ids = [
                    event_id
                    for event_id in event_ids
                    if predicate(by_key.get((event_id, "dense_prefix", seed_count, budget), {"gold_size": 0}))
                ]
                for left, right in [
                    ("pairwise_oracle_expand", "dense_prefix"),
                    ("hypergraph_oracle_expand", "dense_prefix"),
                    ("hypergraph_oracle_multi_seed", "dense_prefix"),
                    ("hypergraph_oracle_expand", "pairwise_oracle_expand"),
                    ("hypergraph_oracle_multi_seed", "pairwise_oracle_expand"),
                ]:
                    deltas_recall = []
                    deltas_full_hit = []
                    wins = losses = ties = 0
                    identical = 0
                    active = 0
                    paired = 0
                    for event_id in eligible_event_ids:
                        left_row = by_key.get((event_id, left, seed_count, budget))
                        right_row = by_key.get((event_id, right, seed_count, budget))
                        if left_row is None or right_row is None:
                            continue
                        paired += 1
                        if float(left_row.get("multi_seed_activated_hyperedges", 0.0)) > 0.0:
                            active += 1
                        delta = float(left_row["evidence_recall"]) - float(right_row["evidence_recall"])
                        deltas_recall.append(delta)
                        deltas_full_hit.append(float(left_row["full_group_hit"]) - float(right_row["full_group_hit"]))
                        if left_row["selected_ids"] == right_row["selected_ids"]:
                            identical += 1
                        if delta > 1e-12:
                            wins += 1
                        elif delta < -1e-12:
                            losses += 1
                        else:
                            ties += 1
                    ci = paired_bootstrap_ci(deltas_recall, seed=seed + len(subset_name) + seed_count * 100 + budget * 10 + len(left))
                    out.append(
                        {
                            "subset": subset_name,
                            "seed_count": seed_count,
                            "candidate_budget": budget,
                            "left_method": left,
                            "right_method": right,
                            "paired_events": paired,
                            "active_events": active,
                            "mean_delta_evidence_recall": mean(deltas_recall),
                            "ci95_low_delta_evidence_recall": ci["ci_low"],
                            "ci95_high_delta_evidence_recall": ci["ci_high"],
                            "mean_delta_full_group_hit": mean(deltas_full_hit),
                            "wins": wins,
                            "losses": losses,
                            "ties": ties,
                            "sign_test_p_value": sign_test_p_value(wins, losses),
                            "identical_output_rate": identical / paired if paired else 0.0,
                        }
                    )
    return out


def write_reports(rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]], output_dir: Path) -> None:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subset = "gold_ge_2" if int(row["gold_size"]) >= 2 else "all"
        grouped[(subset, int(row["seed_count"]), int(row["candidate_budget"]), row["method"])].append(row)
        grouped[("all", int(row["seed_count"]), int(row["candidate_budget"]), row["method"])].append(row)
    seen = set()
    dedup_rows = []
    for key, members in grouped.items():
        if key in seen:
            continue
        seen.add(key)
        subset, seed_count, budget, method = key
        dedup_rows.append(
            {
                "subset": subset,
                "seed_count": seed_count,
                "candidate_budget": budget,
                "method": method,
                "events": len(members),
                "mean_gold_size": mean([float(row["gold_size"]) for row in members]),
                "mean_evidence_recall": mean([float(row["evidence_recall"]) for row in members]),
                "full_group_hit_rate": mean([float(row["full_group_hit"]) for row in members]),
                "mean_activated_hyperedges": mean([float(row.get("activated_hyperedges", 0.0)) for row in members]),
                "mean_multi_seed_activated_hyperedges": mean([float(row.get("multi_seed_activated_hyperedges", 0.0)) for row in members]),
            }
        )
    summary_rows = sorted(dedup_rows, key=lambda row: (row["subset"], row["seed_count"], row["candidate_budget"], row["method"]))

    summary_path = reports_dir / "locomo_oracle_group_summary.csv"
    comparison_path = reports_dir / "locomo_oracle_group_comparisons.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    focus_rows = [
        row
        for row in comparison_rows
        if row["subset"] == "gold_ge_2" and row["left_method"] == "hypergraph_oracle_expand" and row["right_method"] == "pairwise_oracle_expand"
    ]
    lines = [
        "# LOCOMO Oracle Group Ablation",
        "",
        "- This is a diagnostic, not an online-valid main result.",
        "- Structure is built from other official QA gold evidence groups within the same tenant.",
        "- The purpose is to test whether LOCOMO evidence groups have higher-order separability beyond pairwise projection.",
        "",
        f"- per-event trace: `{output_dir / 'runs' / 'locomo_oracle_group_ablation_per_event.jsonl'}`",
        f"- summary: `{summary_path}`",
        f"- comparisons: `{comparison_path}`",
        "",
        "## Focus: Multi-Evidence Questions",
        "",
    ]
    for row in focus_rows:
        lines.append(
            "- "
            f"S={row['seed_count']} B={row['candidate_budget']}: "
            f"delta_recall={float(row['mean_delta_evidence_recall']):.4f} "
            f"CI=[{float(row['ci95_low_delta_evidence_recall']):.4f},{float(row['ci95_high_delta_evidence_recall']):.4f}] "
            f"delta_full_hit={float(row['mean_delta_full_group_hit']):.4f} "
            f"wins/losses/ties={row['wins']}/{row['losses']}/{row['ties']} "
            f"identical={float(row['identical_output_rate']):.3f} active={row['active_events']}"
        )
    (reports_dir / "locomo_oracle_group_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--phase-trace", type=Path, default=DEFAULT_PHASE_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--candidate-budgets", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--anchor-multiplier", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    rows = run_experiment(args)
    comparison_rows = build_comparison_rows(rows, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "runs" / "locomo_oracle_group_ablation_per_event.jsonl", rows)
    write_reports(rows, comparison_rows, args.output_dir)
    write_json(
        args.output_dir / "artifacts" / "run_manifest.json",
        {
            "manifest_dir": str(args.manifest_dir),
            "phase_trace": str(args.phase_trace),
            "rows": len(rows),
            "selection_uses_gold_structure": True,
            "purpose": "oracle separability diagnostic only",
        },
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
