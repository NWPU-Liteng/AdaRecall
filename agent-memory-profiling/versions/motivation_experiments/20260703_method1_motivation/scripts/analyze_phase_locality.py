#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agent-memory-motivation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agent-memory-motivation-cache")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_phase_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            mapping[key.strip()] = value.strip()
    return mapping


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / float(len(union))


def _coaccess_matrix(rows: list[dict]) -> pd.DataFrame:
    phase_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    for dataset_rows in grouped.values():
        ordered = sorted(dataset_rows, key=lambda row: int(row["sequence_index"]))
        for left, right in zip(ordered, ordered[1:]):
            phase_pairs[(left["normalized_phase"], right["normalized_phase"])].append(
                _jaccard(set(left["selected_ids"]), set(right["selected_ids"]))
            )
    output_rows = []
    phases = sorted({row["normalized_phase"] for row in rows})
    for phase_left in phases:
        for phase_right in phases:
            values = phase_pairs.get((phase_left, phase_right), [])
            output_rows.append(
                {
                    "phase_left": phase_left,
                    "phase_right": phase_right,
                    "mean_jaccard": sum(values) / float(len(values)) if values else 0.0,
                    "count": len(values),
                }
            )
    return pd.DataFrame(output_rows)


def _metric_for_rows(rows: list[dict], window_size: int) -> tuple[float, float, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    same_scores: list[float] = []
    diff_scores: list[float] = []
    all_scores: list[float] = []
    for dataset_rows in grouped.values():
        ordered = sorted(dataset_rows, key=lambda row: int(row["sequence_index"]))
        for idx, left in enumerate(ordered):
            for offset in range(1, window_size + 1):
                if idx + offset >= len(ordered):
                    break
                right = ordered[idx + offset]
                score = _jaccard(set(left["selected_ids"]), set(right["selected_ids"]))
                all_scores.append(score)
                if left["normalized_phase"] == right["normalized_phase"]:
                    same_scores.append(score)
                else:
                    diff_scores.append(score)
    same_mean = sum(same_scores) / float(len(same_scores)) if same_scores else 0.0
    diff_mean = sum(diff_scores) / float(len(diff_scores)) if diff_scores else 0.0
    overall_mean = sum(all_scores) / float(len(all_scores)) if all_scores else 1e-9
    jaccard_gap = same_mean - diff_mean
    coaccess_lift = same_mean / max(overall_mean, 1e-9)
    adjacent_overlap = sum(all_scores) / float(len(all_scores)) if all_scores else 0.0
    return jaccard_gap, coaccess_lift, adjacent_overlap


def _phase_shuffle(rows: list[dict], rng: random.Random) -> list[dict]:
    shuffled: list[dict] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(dict(row))
    for dataset_rows in grouped.values():
        phases = [row["normalized_phase"] for row in dataset_rows]
        rng.shuffle(phases)
        for row, phase in zip(dataset_rows, phases):
            row["normalized_phase"] = phase
            shuffled.append(row)
    return shuffled


def _memory_shuffle(rows: list[dict], rng: random.Random) -> list[dict]:
    shuffled: list[dict] = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(dict(row))
    for dataset_rows in grouped.values():
        pool = sorted({memory_id for row in dataset_rows for memory_id in row["selected_ids"]})
        for row in dataset_rows:
            k = len(row["selected_ids"])
            row["selected_ids"] = rng.sample(pool, k) if len(pool) >= k else list(pool)
            shuffled.append(row)
    return shuffled


def _plot(matrix: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    phases = sorted(matrix["phase_left"].unique())
    pivot = matrix.pivot(index="phase_left", columns="phase_right", values="mean_jaccard").reindex(index=phases, columns=phases).fillna(0.0)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(pivot.values, cmap="Blues")
    ax1.set_xticks(range(len(phases)))
    ax1.set_xticklabels(phases, rotation=20)
    ax1.set_yticks(range(len(phases)))
    ax1.set_yticklabels(phases)
    ax1.set_title("Phase-Conditioned Co-access Matrix")
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    fig1.tight_layout()
    fig1.savefig(figures_dir / "fig_o1_phase_locality_matrix.png", dpi=180)
    fig1.savefig(figures_dir / "fig_o1_phase_locality_matrix.pdf")
    plt.close(fig1)

    focus = summary[(summary["metric"] == "adjacent_overlap") & (summary["window_size"] == 1)].copy()
    focus = focus.groupby("model", as_index=False)["value"].agg(["mean", "std", "count"]).reset_index()
    focus["stderr"] = focus["std"].fillna(0.0) / focus["count"].clip(lower=1) ** 0.5
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.bar(focus["model"], focus["mean"], yerr=focus["stderr"], capsize=4)
    ax2.set_title("Real Traces vs Null Models")
    ax2.set_ylabel("Adjacent overlap")
    fig2.tight_layout()
    fig2.savefig(figures_dir / "fig_o1_null_model_gap.png", dpi=180)
    fig2.savefig(figures_dir / "fig_o1_null_model_gap.pdf")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze phase-conditioned locality from adaptive phase traces.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--phase-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = _read_jsonl(Path(args.trace))
    phase_map = _load_phase_map(Path(args.phase_map))

    parsed_rows: list[dict] = []
    audit_rows = []
    raw_counts: dict[str, int] = defaultdict(int)
    raw_examples: dict[str, list[str]] = defaultdict(list)
    seen_rows: set[tuple] = set()
    for row in rows:
        raw_phase = str(row["phase"])
        normalized = phase_map.get(raw_phase, "unknown")
        selected_ids = list(row.get("reranked_top_k_ids") or row.get("phase_selected_memory_ids") or [])
        dedupe_key = (
            str(row.get("dataset")),
            str(row.get("source_run")),
            str(row.get("event_id")),
            raw_phase,
            tuple(selected_ids),
        )
        if dedupe_key in seen_rows:
            continue
        seen_rows.add(dedupe_key)
        parsed_rows.append(
            {
                "dataset": row["dataset"],
                "sequence_index": int(row["sequence_index"]),
                "normalized_phase": normalized,
                "selected_ids": selected_ids,
            }
        )
        raw_counts[raw_phase] += 1
        if len(raw_examples[raw_phase]) < 3:
            raw_examples[raw_phase].append(str(row.get("controller_goal", ""))[:120])

    for raw_phase, count in sorted(raw_counts.items()):
        audit_rows.append(
            {
                "raw_label": raw_phase,
                "normalized_phase": phase_map.get(raw_phase, "unknown"),
                "count": count,
                "examples": " | ".join(raw_examples[raw_phase]),
            }
        )
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(output_dir / "manifests" / "phase_mapping_audit.csv", index=False)

    matrix = _coaccess_matrix(parsed_rows)
    matrix.to_csv(output_dir / "aggregate" / "O1_phase_matrix.csv", index=False)

    summary_rows = []
    rng = random.Random(args.seed)
    for window_size in args.window_size:
        real_gap, real_lift, real_overlap = _metric_for_rows(parsed_rows, window_size)
        summary_rows.append({"model": "real", "window_size": window_size, "metric": "jaccard_gap", "value": real_gap, "repetition": 0})
        summary_rows.append({"model": "real", "window_size": window_size, "metric": "coaccess_lift", "value": real_lift, "repetition": 0})
        summary_rows.append({"model": "real", "window_size": window_size, "metric": "adjacent_overlap", "value": real_overlap, "repetition": 0})
        for repetition in range(1, args.repetitions + 1):
            phase_rows = _phase_shuffle(parsed_rows, random.Random(rng.randint(0, 10**9)))
            phase_gap, phase_lift, phase_overlap = _metric_for_rows(phase_rows, window_size)
            summary_rows.append({"model": "phase_shuffle", "window_size": window_size, "metric": "jaccard_gap", "value": phase_gap, "repetition": repetition})
            summary_rows.append({"model": "phase_shuffle", "window_size": window_size, "metric": "coaccess_lift", "value": phase_lift, "repetition": repetition})
            summary_rows.append({"model": "phase_shuffle", "window_size": window_size, "metric": "adjacent_overlap", "value": phase_overlap, "repetition": repetition})

            memory_rows = _memory_shuffle(parsed_rows, random.Random(rng.randint(0, 10**9)))
            memory_gap, memory_lift, memory_overlap = _metric_for_rows(memory_rows, window_size)
            summary_rows.append({"model": "memory_shuffle", "window_size": window_size, "metric": "jaccard_gap", "value": memory_gap, "repetition": repetition})
            summary_rows.append({"model": "memory_shuffle", "window_size": window_size, "metric": "coaccess_lift", "value": memory_lift, "repetition": repetition})
            summary_rows.append({"model": "memory_shuffle", "window_size": window_size, "metric": "adjacent_overlap", "value": memory_overlap, "repetition": repetition})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "aggregate" / "O1_phase_locality.csv", index=False)
    _plot(matrix, summary_df, output_dir / "figures")


if __name__ == "__main__":
    main()
