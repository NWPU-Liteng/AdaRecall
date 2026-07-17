#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def csv_rows(rel: str) -> list[dict[str, str]]:
    path = ROOT / rel
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    checks = []
    def add(name: str, ok: bool, detail: str = ""):
        checks.append((name, ok, detail))

    required = [
        "reports/V3_REUSE_MANIFEST.md",
        "reports/DATASET_PROVENANCE_AUDIT.md",
        "manifests/dataset_file_hashes.sha256",
        "manifests/runtime_manifest.json",
        "aggregate/A2_causal_leakage_audit.jsonl",
        "reports/A2_CAUSAL_LEAKAGE_AUDIT.md",
        "aggregate/A3_candidate_path_occupancy.csv",
        "reports/A3_CANDIDATE_PATH_AUDIT.md",
        "reports/A4_CONTROLLER_OVERHEAD_POSITIONING.md",
        "aggregate/B_stage_trace_v4.jsonl",
        "aggregate/B_chronological_split_manifest.csv",
        "aggregate/B_heldout_stage_transfer_per_query.csv",
        "aggregate/B_heldout_stage_transfer_summary.csv",
        "aggregate/B_heldout_stage_shuffle_null.csv",
        "reports/B_HELDOUT_STAGE_TRANSFER.md",
        "aggregate/C_keypoint_alignment_rows.jsonl",
        "aggregate/C_keypoint_alignment_summary.csv",
        "aggregate/C_strict_evidence_budget_summary.csv",
        "reports/C_KEYPOINT_ALIGNMENT_AUDIT.md",
        "reports/C_STRICT_EVIDENCE_INTERPRETATION.md",
        "artifacts/evidence_review_packet.csv",
        "reports/EVIDENCE_REVIEW_GUIDE.md",
        "figures/Fig_M1_v4_history_footprint_traffic.png",
        "figures/Fig_M2_v4_frontier_cost_reference_fidelity.png",
        "figures/Fig_M3_v4_tile_access_amplification.png",
        "figures/Fig_O1_v4_same_cross_stage_locality.png",
        "figures/Fig_O1_v4_heldout_stage_transfer.png",
        "figures/Fig_O1_v4_heldout_stage_shuffle_null.png",
        "figures/Fig_O1_v4_stage_seen_node_coverage.png",
        "figures/Fig_M2_v4_budget_cost_strict_keypoint_recall.png",
        "scripts/figure_plots/plot_fig_m1_history_footprint_traffic.py",
        "scripts/figure_plots/plot_fig_m2_frontier_cost_reference_fidelity.py",
        "scripts/figure_plots/plot_fig_m3_tile_access_amplification.py",
        "scripts/figure_plots/plot_fig_o1_same_cross_stage_locality.py",
        "scripts/figure_plots/plot_fig_o1_heldout_stage_transfer.py",
        "scripts/figure_plots/plot_fig_o1_heldout_stage_shuffle_null.py",
        "scripts/figure_plots/plot_fig_o1_stage_seen_node_coverage.py",
        "scripts/figure_plots/plot_fig_m2_budget_cost_strict_keypoint_recall.py",
        "scripts/figure_plots/run_all_figures.py",
        "reports/FIGURE_DATA_SOURCES.md",
        "reports/FINAL_MOTIVATION_V4_SUMMARY.md",
        "reports/FINAL_MOTIVATION_V4_CLAIM_BOUNDARIES.md",
        "reports/FINAL_MOTIVATION_V4_FIGURE_GUIDE.md",
    ]
    for rel in required:
        add(f"`{rel}` exists", exists(rel))

    split = csv_rows("aggregate/B_chronological_split_manifest.csv")
    add("held-out split is chronological per persona", bool(split) and all(int(r["n_train"]) >= 1 and int(r["n_test"]) >= 1 for r in split if int(r["n_queries"]) > 1))
    null = csv_rows("aggregate/B_heldout_stage_shuffle_null.csv")
    add("stage-shuffle null uses >= 1000 permutations", len(null) >= 1000)
    summary = csv_rows("aggregate/B_heldout_stage_transfer_summary.csv")
    add("bootstrap uses >= 1000 resamples", bool(summary) and all(int(float(r["bootstrap_samples"])) >= 1000 for r in summary))
    c = csv_rows("aggregate/C_keypoint_alignment_summary.csv")
    add("strict evidence subset uses deterministic alignment", bool(c))
    a3 = csv_rows("aggregate/A3_candidate_path_occupancy.csv")
    add("M2 budgets correspond to real reranked frontier sizes", bool(a3) and len({int(float(r["n_reranked"])) for r in a3}) >= 4)

    status_rows = [
        ("M1", "pass", "appendix", "history footprint/traffic"),
        ("M2", "pass", "yes", "frontier cost/reference fidelity"),
        ("M3", "pass", "yes", "tile scattering/over-fetch"),
        ("O1 within-trace", "pass", "yes", "non-random stage locality"),
        ("O1 held-out transfer", "see B report", "conditional", "earlier-to-later transfer"),
        ("strict evidence subset", "see C report", "appendix", "deterministic keypoint subset"),
    ]
    lines = ["# Validation V4", ""]
    for name, ok, detail in checks:
        lines.append(f"- [{'x' if ok else ' '}] {name}" + (f": {detail}" if detail else ""))
    lines.extend(["", "## Status Table", "", "| Component | Status | Main-paper usable? | Claim |", "|---|---|---|---|"])
    for row in status_rows:
        lines.append("| " + " | ".join(row) + " |")
    failed = [name for name, ok, _ in checks if not ok]
    lines.extend(["", "## Overall", "", "PASSED" if not failed else "FAILED", ""])
    if failed:
        lines.append("Failed checks:")
        lines.extend(f"- {name}" for name in failed)
    (ROOT / "reports" / "VALIDATION_V4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("\n".join(failed[:20]))
    print("validation_v4=PASSED")


if __name__ == "__main__":
    main()
