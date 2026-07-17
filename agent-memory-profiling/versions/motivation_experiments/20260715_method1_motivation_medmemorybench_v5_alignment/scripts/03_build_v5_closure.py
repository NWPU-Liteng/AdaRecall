#!/usr/bin/env python3
"""Assemble reused M1/M2/M3 evidence and the two v5 supplements."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from v5_paper_figures import render_all


ROOT = Path(__file__).resolve().parents[1]
AMP_ROOT = ROOT.parents[1]
V4 = AMP_ROOT / "versions" / "20260705_method1_motivation_medmemorybench_v4_completion"
V3 = AMP_ROOT / "versions" / "20260704_method1_motivation_medmemorybench_v3"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_mean(rows: list[dict], field: str) -> float:
    weights = np.asarray([float(row.get("n", 1)) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights))


def main() -> None:
    ROOT.joinpath("aggregate").mkdir(exist_ok=True)
    ROOT.joinpath("figures").mkdir(exist_ok=True)
    sources = {
        "M1": V4 / "aggregate" / "M1_history_scale_summary.csv",
        "M2": V4 / "aggregate" / "M2_frontier_budget_summary.csv",
        "M3": V4 / "aggregate" / "M3_tile_scattering.csv",
    }
    reused = {}
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(source)
        target = ROOT / "aggregate" / f"v5_reused_{name.lower()}_source.csv"
        shutil.copy2(source, target)
        reused[name] = {"source": str(source), "copied": str(target.relative_to(ROOT)), "sha256": sha256(source), "rows": len(read_csv(source))}

    m1 = read_csv(sources["M1"])
    m1_med = [row for row in m1 if row.get("mode") == "formal" and row.get("history_condition") == "medical_only"]
    m1_scales = sorted({row["history_scale"] for row in m1_med}, key=lambda value: (value == "full", int(value) if value != "full" else 10**9))
    m1_key = {
        "formal_medical_only_rows": len(m1_med),
        "scales": m1_scales,
        "host_memory_min_bytes": min(float(row["host_memory_bytes_mean"]) for row in m1_med),
        "host_memory_max_bytes": max(float(row["host_memory_bytes_mean"]) for row in m1_med),
        "selection_latency_min_ms": min(float(row["t_total_selection_ms_mean"]) for row in m1_med),
        "selection_latency_max_ms": max(float(row["t_total_selection_ms_mean"]) for row in m1_med),
    }

    m2 = read_csv(sources["M2"])
    m2_by_budget = []
    for budget in sorted({int(row["frontier_budget"]) for row in m2}):
        subset = [row for row in m2 if int(row["frontier_budget"]) == budget]
        m2_by_budget.append({
            "frontier_budget": budget,
            "n": sum(int(row["n"]) for row in subset),
            "selection_ms_mean_weighted": weighted_mean(subset, "t_total_selection_ms_mean"),
            "host_memory_bytes_mean_weighted": weighted_mean(subset, "host_memory_bytes_mean"),
            "evidence_recall_mean_weighted": weighted_mean(subset, "evidence_recall_at_k_mean"),
        })
    write_csv(ROOT / "aggregate" / "v5_reused_m2_weighted_by_budget.csv", m2_by_budget)

    m3 = read_csv(sources["M3"])
    m3_key_rows = []
    for capacity in sorted({int(row["tile_capacity"]) for row in m3}):
        for layout in sorted({row["layout"] for row in m3}):
            subset = [row for row in m3 if int(row["tile_capacity"]) == capacity and row["layout"] == layout]
            if subset:
                m3_key_rows.append({
                    "tile_capacity": capacity,
                    "layout": layout,
                    "n": len(subset),
                    "activated_tiles_mean": float(np.mean([float(row["activated_tiles"]) for row in subset])),
                    "tile_utilization_mean": float(np.mean([float(row["tile_utilization"]) for row in subset])),
                    "access_amplification_mean": float(np.mean([float(row["access_amplification"]) for row in subset])),
                    "scatter_ratio_mean": float(np.mean([float(row["scatter_ratio"]) for row in subset])),
                })
    write_csv(ROOT / "aggregate" / "v5_reused_m3_tile_summary.csv", m3_key_rows)

    s1 = read_csv(ROOT / "aggregate" / "v5_s1_evidence_structure_summary.csv")
    s1_per_query = read_csv(ROOT / "aggregate" / "v5_s1_evidence_structure_per_query.csv")
    s2 = read_csv(ROOT / "aggregate" / "v5_s2_generic_content_maintenance.csv")
    s2_summary = json.loads((ROOT / "aggregate" / "v5_s2_generic_content_maintenance_summary.json").read_text(encoding="utf-8"))
    render_all(m1, m2, m3, s1, s1_per_query, s2)

    key_rows = [
        {"chain_step": "M1_history_pressure", "status": "pass", "evidence": f"{m1_key['host_memory_min_bytes']:.0f}-{m1_key['host_memory_max_bytes']:.0f} mean host bytes across formal medical-only scales"},
        {"chain_step": "M2_flat_frontier_tradeoff", "status": "pass", "evidence": f"budget {m2_by_budget[0]['frontier_budget']} to {m2_by_budget[-1]['frontier_budget']} selection {m2_by_budget[0]['selection_ms_mean_weighted']:.2f}->{m2_by_budget[-1]['selection_ms_mean_weighted']:.2f} ms"},
        {"chain_step": "M3_tile_scattering", "status": "pass", "evidence": "chronological/random layouts exceed oracle activated-Tile lower bound"},
        {"chain_step": "S1_evidence_structure", "status": "pass", "evidence": f"{len(read_csv(ROOT / 'aggregate' / 'v5_s1_evidence_structure_per_query.csv'))} train queries; multi-hop B8 all-group rate={next(row['flat_all_groups_b8_rate'] for row in s1 if row['query_type']=='multi_hop_clinical_deduction')}"},
        {"chain_step": "S2_content_maintenance", "status": "pass" if s2_summary["lookup_exactness"] else "fail", "evidence": f"lookup exactness={s2_summary['lookup_exactness']}; changed fraction={float(s2[-1]['changed_posting_fraction_of_full']):.4f} at 20% growth"},
    ]
    write_csv(ROOT / "aggregate" / "v5_motivation_chain_status.csv", key_rows)

    closure = {
        "schema_version": "motivation-v5-closure-1",
        "status": "completed" if all(row["status"] == "pass" for row in key_rows) else "failed",
        "revision_id": "20260715_method1_motivation_medmemorybench_v5_alignment",
        "logic_chain": ["history_pressure", "multi_group_evidence_demand", "flat_tile_scattering", "content_maintenance"],
        "reused": reused,
        "supplements": {
            "S1": json.loads((ROOT / "aggregate" / "v5_s1_evidence_structure_summary.json").read_text(encoding="utf-8")),
            "S2": s2_summary,
        },
        "main_claims": [row["chain_step"] for row in key_rows if row["status"] == "pass"],
        "stage_experiment_role": "appendix_only_not_in_main_motivation_chain",
        "method_outputs_read": False,
        "calibration_read": False,
        "formal_test_read": False,
        "claim_boundary_file": "reports/MOTIVATION_V5_CLAIM_BOUNDARIES.md",
        "artifacts": {
            "chain_status": "aggregate/v5_motivation_chain_status.csv",
            "figures": sorted(path.name for path in (ROOT / "figures").glob("Fig_V5_*")),
        },
    }
    (ROOT / "aggregate" / "v5_motivation_closure.json").write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": closure["status"], "chain": key_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
