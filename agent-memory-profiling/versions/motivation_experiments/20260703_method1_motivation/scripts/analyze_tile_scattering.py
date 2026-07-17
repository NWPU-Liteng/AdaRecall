#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import math
import random
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agent-memory-motivation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agent-memory-motivation-cache")


SCRIPT_ROOT = Path(__file__).resolve().parent
VERSION_ROOT = SCRIPT_ROOT.parent
PROJECT_ROOT = VERSION_ROOT.parent.parent

DATASET_MEMORY_PATHS = {
    "locomo": PROJECT_ROOT / "data" / "motivation_memory_selection" / "locomo_memories_1000.jsonl",
    "longmemeval": PROJECT_ROOT / "data" / "motivation_memory_selection" / "longmemeval_memories_1000.jsonl",
    "msc": PROJECT_ROOT / "data" / "motivation_memory_selection" / "msc_memories_5000.jsonl",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def _load_orders(seed: int) -> dict[str, dict[str, list[str] | dict[str, int]]]:
    dataset_orders: dict[str, dict[str, list[str] | dict[str, int]]] = {}
    for dataset, path in DATASET_MEMORY_PATHS.items():
        memories = _read_jsonl(path)
        chronological = [str(row["memory_id"]) for row in memories]
        random_ids = list(chronological)
        rng = random.Random(seed + len(chronological))
        rng.shuffle(random_ids)
        dataset_orders[dataset] = {
            "chronological_ids": chronological,
            "chronological_index": {memory_id: idx for idx, memory_id in enumerate(chronological)},
            "random_index": {memory_id: idx for idx, memory_id in enumerate(random_ids)},
        }
    return dataset_orders


def _compute_rows(input_path: Path, capacities: list[int], seed: int) -> list[dict]:
    df = pd.read_json(input_path, lines=True)
    focused = df[
        (df["baseline_mode"] == "full_selection")
        & (df["frontier_budget"] == df.groupby("dataset")["frontier_budget"].transform("max"))
        & (df["relation_expansion_depth"] == df.groupby("dataset")["relation_expansion_depth"].transform("max"))
        & (df["error"].fillna("") == "")
    ].copy()
    orders = _load_orders(seed)
    rows: list[dict] = []

    for _, record in focused.iterrows():
        dataset = str(record["dataset"])
        selected_ids = list(record.get("selected_memory_ids") or [])
        if not selected_ids:
            continue
        k_o = len(selected_ids)
        for capacity in capacities:
            ideal_tiles = int(math.ceil(k_o / float(capacity)))
            for layout in ["chronological", "random", "ideal"]:
                if layout == "ideal":
                    tile_count = ideal_tiles
                else:
                    index_key = "chronological_index" if layout == "chronological" else "random_index"
                    index_map = orders[dataset][index_key]
                    tile_count = len({int(index_map[memory_id]) // capacity for memory_id in selected_ids if memory_id in index_map})
                tile_utilization = k_o / float(capacity * max(tile_count, 1))
                access_amplification = (capacity * tile_count) / float(k_o)
                scatter_ratio = tile_count / float(max(ideal_tiles, 1))
                rows.append(
                    {
                        "dataset": dataset,
                        "query_id": record["query_id"],
                        "query_type": record["query_type"],
                        "query_stratum": record["query_stratum"],
                        "layout": layout,
                        "tile_capacity": capacity,
                        "final_evidence_k": k_o,
                        "activated_tiles": tile_count,
                        "ideal_tiles": ideal_tiles,
                        "tile_utilization": tile_utilization,
                        "access_amplification": access_amplification,
                        "scatter_ratio": scatter_ratio,
                    }
                )
    return rows


def _plot(summary: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    util = summary.groupby(["layout", "tile_capacity"], as_index=False)["mean_tile_utilization"].mean()
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for layout in ["chronological", "random", "ideal"]:
        subset = util[util["layout"] == layout].sort_values("tile_capacity")
        if subset.empty:
            continue
        ax1.plot(subset["tile_capacity"], subset["mean_tile_utilization"], marker="o", label=layout)
    ax1.set_title("Tile Utilization by Layout")
    ax1.set_xlabel("Tile capacity")
    ax1.set_ylabel("Mean tile utilization")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(figures_dir / "fig_m3_tile_scattering_utilization.png", dpi=180)
    fig1.savefig(figures_dir / "fig_m3_tile_scattering_utilization.pdf")
    plt.close(fig1)

    act = summary.groupby(["layout", "tile_capacity"], as_index=False)["mean_activated_tiles"].mean()
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for layout in ["chronological", "random", "ideal"]:
        subset = act[act["layout"] == layout].sort_values("tile_capacity")
        if subset.empty:
            continue
        ax2.plot(subset["tile_capacity"], subset["mean_activated_tiles"], marker="o", label=layout)
    ax2.set_title("Activated Tiles vs Ideal Lower Bound")
    ax2.set_xlabel("Tile capacity")
    ax2.set_ylabel("Mean activated tiles")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(figures_dir / "fig_m3_tile_activation_vs_ideal.png", dpi=180)
    fig2.savefig(figures_dir / "fig_m3_tile_activation_vs_ideal.pdf")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tile scattering from historical M1 outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-capacity", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_rows = _compute_rows(Path(args.input), args.tile_capacity, args.seed)
    raw_path = output_dir / "raw" / "M3_tile_mapping.jsonl"
    _write_jsonl(raw_path, raw_rows)

    df = pd.DataFrame(raw_rows)
    summary = (
        df.groupby(["dataset", "query_stratum", "layout", "tile_capacity"], dropna=False)
        .agg(
            n_queries=("query_id", "count"),
            mean_activated_tiles=("activated_tiles", "mean"),
            mean_tile_utilization=("tile_utilization", "mean"),
            median_tile_utilization=("tile_utilization", "median"),
            p95_access_amplification=("access_amplification", lambda s: _safe_quantile(s, 0.95)),
            mean_access_amplification=("access_amplification", "mean"),
            mean_scatter_ratio=("scatter_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "query_stratum", "layout", "tile_capacity"])
    )
    summary_path = output_dir / "aggregate" / "M3_tile_scattering.csv"
    summary.to_csv(summary_path, index=False)
    _plot(summary, output_dir / "figures")


if __name__ == "__main__":
    main()
