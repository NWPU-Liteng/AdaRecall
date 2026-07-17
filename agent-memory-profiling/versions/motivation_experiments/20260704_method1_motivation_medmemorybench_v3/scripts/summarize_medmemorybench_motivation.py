#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _exists(root: Path, rel: str) -> str:
    return "present" if (root / rel).exists() else "missing"


def _read_head(path: Path, lines: int = 5) -> str:
    if not path.exists():
        return "missing"
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:lines])


def main() -> None:
    parser = argparse.ArgumentParser(description="Write final MedMemoryBench motivation summary.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    required_figures = [
        "Fig_M1_history_scale_latency.png",
        "Fig_M1_history_scale_traffic.png",
        "Fig_M2_frontier_cost_evidence.png",
        "Fig_M2_budget_task_quality.png",
        "Fig_M2_budget_oracle_overlap.png",
        "Fig_M3_tile_utilization.png",
        "Fig_M3_access_amplification.png",
        "Fig_M3_scatter_ratio_by_query_type.png",
        "Fig_O1_same_vs_cross_stage_overlap.png",
        "Fig_O1_stage_shuffle_null.png",
        "Fig_O1_stage_pair_recurrence.png",
    ]
    lines = [
        "# Final Motivation Summary",
        "",
        "## Environment and provenance",
        "",
        f"- Dataset provenance: {_exists(root, 'datasets/DATASET_PROVENANCE.md')}",
        f"- Pipeline audit: {_exists(root, 'reports/00_pipeline_audit.md')}",
        f"- Pipeline delta: {_exists(root, 'reports/00_pipeline_delta.md')}",
        "",
        "## MedMemoryBench schema and workload characterization",
        "",
        _read_head(root / "reports/01_medmemorybench_schema_audit.md", 12),
        "",
        "## Memory-item and evidence-mapping coverage",
        "",
        _read_head(root / "reports/02_memory_item_and_evidence_audit.md", 16),
        "",
        "## M1: History-scale cost behavior",
        "",
        f"- M1 invocations: {_exists(root, 'aggregate/M1_history_scale_invocations.jsonl')}",
        f"- M1 summary: {_exists(root, 'aggregate/M1_history_scale_summary.csv')}",
        "",
        "## M2: Flat frontier cost versus evidence/task returns",
        "",
        f"- M2 invocations: {_exists(root, 'aggregate/M2_frontier_budget_invocations.jsonl')}",
        f"- M2 occupancy: {_exists(root, 'aggregate/M2_frontier_occupancy.csv')}",
        f"- M2 summary: {_exists(root, 'aggregate/M2_frontier_budget_summary.csv')}",
        "",
        "## M3: Chronological physical scattering",
        "",
        f"- M3 scattering: {_exists(root, 'aggregate/M3_tile_scattering.csv')}",
        "",
        "## O1: Controller-defined stage locality",
        "",
        f"- O1 trace: {_exists(root, 'aggregate/O1_formal_stage_trace.jsonl')}",
        f"- O1 same/cross: {_exists(root, 'aggregate/O1_same_cross_stage_stats.csv')}",
        f"- O1 shuffle null: {_exists(root, 'aggregate/O1_stage_shuffle_null.csv')}",
        "",
        "## Negative findings and claim boundaries",
        "",
        "- Do not claim hypergraph or PIM superiority from this package.",
        "- Use `controller-defined execution stage`; do not describe O1 labels as source-provided execution-stage labels.",
        "- If O1 null statistics are missing or insignificant, only temporal locality or a negative result may be claimed.",
        "",
        "## Figures ready for paper",
        "",
    ]
    for fig in required_figures:
        lines.append(f"- `{fig}`: {_exists(root, 'figures/' + fig)}")
    lines.extend(
        [
            "",
            "## What remains for method evaluation",
            "",
            "- Dense-vs-graph-vs-hypergraph comparisons.",
            "- PIM runtime measurements or simulation.",
            "- Structural placement, replication, and online remapping evaluation.",
            "",
            "## Claim Table",
            "",
            "| Claim | Evidence | Status | Exact allowed wording |",
            "|---|---|---|---|",
            "| Longitudinal history increases selection cost | M1 | pending validation | Report observed M1 component behavior only. |",
            "| Flat expansion has diminishing evidence return | M2 | pending validation | Use only if evidence/task metrics saturate while cost rises. |",
            "| Chronological layout causes scattering | M3 | pending validation | Chronological placement scatters selected memories if M3 supports it. |",
            "| Controller stage explains co-access locality | O1 | pending validation | Use the O1 conclusion gate after shuffle-null results. |",
        ]
    )
    out = reports / "FINAL_MOTIVATION_SUMMARY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary={out}")


if __name__ == "__main__":
    main()
