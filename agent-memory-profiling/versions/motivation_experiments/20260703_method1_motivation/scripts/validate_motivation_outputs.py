#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _require(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing: {path}")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the consolidated motivation package.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    errors: list[str] = []

    required_files = [
        root / "README.md",
        root / "RESULTS.md",
        root / "LIMITATIONS.md",
        root / "config" / "phase_mapping.yaml",
        root / "manifests" / "dataset_manifest.json",
        root / "manifests" / "run_manifest.json",
        root / "manifests" / "source_runs.json",
        root / "manifests" / "phase_mapping_audit.csv",
        root / "raw" / "M1_profile_cross_encoder_v3.jsonl",
        root / "raw" / "M3_tile_mapping.jsonl",
        root / "raw" / "O1_phase_trace_v4_adaptive.jsonl",
        root / "aggregate" / "M1_stage_latency.csv",
        root / "aggregate" / "M2_pareto.csv",
        root / "aggregate" / "M3_tile_scattering.csv",
        root / "aggregate" / "O1_phase_locality.csv",
        root / "aggregate" / "O1_phase_matrix.csv",
        root / "figures" / "fig_m1_latency_breakdown.png",
        root / "figures" / "fig_m1_latency_breakdown.pdf",
        root / "figures" / "fig_m1_budget_vs_downstream_cost.png",
        root / "figures" / "fig_m1_budget_vs_downstream_cost.pdf",
        root / "figures" / "fig_m2_quality_cost_pareto.png",
        root / "figures" / "fig_m2_quality_cost_pareto.pdf",
        root / "figures" / "fig_m3_tile_scattering_utilization.png",
        root / "figures" / "fig_m3_tile_scattering_utilization.pdf",
        root / "figures" / "fig_m3_tile_activation_vs_ideal.png",
        root / "figures" / "fig_m3_tile_activation_vs_ideal.pdf",
        root / "figures" / "fig_o1_phase_locality_matrix.png",
        root / "figures" / "fig_o1_phase_locality_matrix.pdf",
        root / "figures" / "fig_o1_null_model_gap.png",
        root / "figures" / "fig_o1_null_model_gap.pdf",
    ]
    for path in required_files:
        _require(path, errors)

    if errors:
        raise SystemExit("\n".join(errors))

    m1_rows = _read_jsonl(root / "raw" / "M1_profile_cross_encoder_v3.jsonl")
    if not m1_rows:
        errors.append("M1 raw profile is empty")
    else:
        required_m1_keys = {"dataset", "query_id", "frontier_budget", "selected_memory_ids", "t_total_ms", "seed"}
        missing_keys = sorted(required_m1_keys - set(m1_rows[0].keys()))
        if missing_keys:
            errors.append(f"M1 raw first row missing keys: {missing_keys}")

    m3_rows = _read_jsonl(root / "raw" / "M3_tile_mapping.jsonl")
    if not m3_rows:
        errors.append("M3 raw tile mapping is empty")
    else:
        if any(row["activated_tiles"] <= 0 for row in m3_rows):
            errors.append("M3 contains non-positive activated tile counts")

    o1_rows = _read_jsonl(root / "raw" / "O1_phase_trace_v4_adaptive.jsonl")
    if not o1_rows:
        errors.append("O1 phase trace is empty")

    m1_df = pd.read_csv(root / "aggregate" / "M1_stage_latency.csv")
    m2_df = pd.read_csv(root / "aggregate" / "M2_pareto.csv")
    m3_df = pd.read_csv(root / "aggregate" / "M3_tile_scattering.csv")
    o1_df = pd.read_csv(root / "aggregate" / "O1_phase_locality.csv")

    if m1_df.empty:
        errors.append("M1 aggregate is empty")
    if m2_df.empty:
        errors.append("M2 aggregate is empty")
    if m3_df.empty:
        errors.append("M3 aggregate is empty")
    if o1_df.empty:
        errors.append("O1 aggregate is empty")

    if "frontier_budget" in m1_df.columns and (m1_df["frontier_budget"] <= 0).any():
        errors.append("M1 aggregate contains non-positive frontier budget")
    if "tile_capacity" in m3_df.columns and (m3_df["tile_capacity"] <= 0).any():
        errors.append("M3 aggregate contains non-positive tile capacity")
    if not {"real", "phase_shuffle", "memory_shuffle"}.issubset(set(o1_df["model"].unique())):
        errors.append("O1 aggregate is missing one or more required models: real/phase_shuffle/memory_shuffle")

    if errors:
        raise SystemExit("\n".join(errors))

    print(f"validated_root={root}")


if __name__ == "__main__":
    main()
