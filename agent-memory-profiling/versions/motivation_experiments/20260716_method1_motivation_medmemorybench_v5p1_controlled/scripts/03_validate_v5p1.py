#!/usr/bin/env python3
"""Validate the v5.1 controlled cohort and unified motivation data."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, observed: Any, expected: Any, detail: str = "") -> None:
        checks.append({"check": name, "status": "pass" if ok else "fail", "observed": observed, "expected": expected, "detail": detail})

    cohort_path = ROOT / "aggregate/v5p1_m1_common_query_ids.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    expected_ids = set(cohort["selected_query_ids"])
    inv_path = ROOT / "aggregate/M1_controlled_invocations.jsonl"
    inv_rows = [json.loads(line) for line in inv_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in inv_rows:
        cells[(str(row["history_condition"]), int(row["history_scale"]))].append(row)
    expected_cells = {(condition, scale) for condition in ["medical_only", "medical_plus_official_noise"] for scale in [128, 256, 512]}
    check("m1_invocation_row_count", len(inv_rows) == 486, len(inv_rows), 486)
    check("m1_expected_cells", set(cells) == expected_cells, sorted(cells), sorted(expected_cells))
    cell_sets_equal = all({str(row["query_id"]) for row in rows} == expected_ids for rows in cells.values())
    check("m1_identical_query_ids_per_cell", cell_sets_equal, sorted({len({str(r['query_id']) for r in rows}) for rows in cells.values()}), 81)
    cell_counts_equal = all(len(rows) == 81 for rows in cells.values())
    check("m1_n_queries_per_cell", cell_counts_equal, {str(key): len(value) for key, value in sorted(cells.items())}, 81)
    hashes = {str(row.get("query_ids_hash", "")) for row in inv_rows}
    check("m1_query_ids_hash", hashes == {cohort["selected_query_ids_hash"]}, sorted(hashes), cohort["selected_query_ids_hash"])
    check("m1_no_errors", all(not row.get("error") for row in inv_rows), sum(bool(row.get("error")) for row in inv_rows), 0)

    summary_path = ROOT / "aggregate/M1_controlled_summary.csv"
    summary = read_csv(summary_path)
    check("m1_summary_cell_count", len(summary) == 6, len(summary), 6)
    check("m1_summary_n", all(int(row["n"]) == 81 for row in summary), sorted({int(row["n"]) for row in summary}), 81)
    check("m1_summary_scales", sorted({int(row["history_scale"]) for row in summary}) == [128, 256, 512], sorted({int(row["history_scale"]) for row in summary}), [128, 256, 512])

    final_path = ROOT / "aggregate/v5p1_motivation_final_data.csv"
    final_rows = read_csv(final_path)
    root_copy = ROOT.parents[2] / "aggregate/v5p1_motivation_final_data.csv"
    expected_experiments = {
        "M1_controlled_history_pressure", "M1_feasibility_audit", "S1_evidence_structure",
        "M2_flat_frontier_tradeoff", "M3_tile_scattering", "S2_generic_content_maintenance",
    }
    present_experiments = {row["experiment"] for row in final_rows}
    check("final_file_nonempty", bool(final_rows), len(final_rows), ">0")
    check(
        "final_root_copy_matches",
        root_copy.exists() and hashlib.sha256(root_copy.read_bytes()).hexdigest() == hashlib.sha256(final_path.read_bytes()).hexdigest(),
        str(root_copy) if root_copy.exists() else "missing",
        "same SHA-256 as version package",
    )
    check("final_expected_experiments", expected_experiments <= present_experiments, sorted(present_experiments), sorted(expected_experiments))
    check("final_no_forbidden_v21_sources", all("v21" not in row["source_file"].lower() for row in final_rows), sorted({row["source_file"] for row in final_rows if "v21" in row["source_file"].lower()}), "none")
    check("final_sources_exist", all((PROJECT / row["source_file"]).exists() for row in final_rows), sorted({row["source_file"] for row in final_rows if not (PROJECT / row["source_file"]).exists()}), "all source files exist")
    check("final_no_missing_values", all(row["metric"] and row["value"] != "" for row in final_rows), sum(not row["metric"] or row["value"] == "" for row in final_rows), 0)
    finite = True
    for row in final_rows:
        try:
            value = float(row["value"])
            finite = finite and math.isfinite(value)
        except (TypeError, ValueError):
            finite = False
    check("final_numeric_values_finite", finite, "finite" if finite else "nonfinite", "finite")
    m1_final = [row for row in final_rows if row["experiment"] == "M1_controlled_history_pressure"]
    m1_amp = [row for row in m1_final if row["metric"] == "host_traffic_amplification_vs_128"]
    check("final_m1_normalized_metric", len(m1_amp) == 6, len(m1_amp), 6)
    baseline_ok = all(abs(float(row["value"]) - 1.0) < 1e-9 for row in m1_amp if row["history_scale"] == "128")
    high_scale = [float(row["value"]) for row in m1_amp if row["history_scale"] == "512"]
    check("m1_normalized_baseline", baseline_ok, [row["value"] for row in m1_amp if row["history_scale"] == "128"], "1.0")
    check("m1_normalized_growth", len(high_scale) == 2 and min(high_scale) > 2.5, high_scale, ">2.5x at 512")
    check("final_m1_normalized_main_role", bool(m1_amp) and all(row["claim_role"] == "main_motivation" for row in m1_amp), sorted({row["claim_role"] for row in m1_amp}), "main_motivation")

    s1_multi = [float(row["value"]) for row in final_rows if row["experiment"] == "S1_evidence_structure" and row["metric"] == "multi_group_fraction" and row["query_type"] == "all"]
    check("s1_multi_group_signal", bool(s1_multi) and s1_multi[0] >= 0.1, s1_multi[0] if s1_multi else None, ">=0.1")
    m2_budget = sorted({int(row["frontier_budget"]) for row in final_rows if row["experiment"] == "M2_flat_frontier_tradeoff" and row["metric"] == "selection_latency_ms"})
    m2_latency = {int(row["frontier_budget"]): float(row["value"]) for row in final_rows if row["experiment"] == "M2_flat_frontier_tradeoff" and row["metric"] == "selection_latency_ms"}
    check("m2_budget_grid", m2_budget == [8, 16, 32, 64], m2_budget, [8, 16, 32, 64])
    check("m2_latency_increases", all(m2_latency[a] < m2_latency[b] for a, b in zip(m2_budget, m2_budget[1:])), m2_latency, "strictly increasing")
    m3_nonoracle = [float(row["value"]) for row in final_rows if row["experiment"] == "M3_tile_scattering" and row["metric"] == "access_amplification" and row["layout"] != "oracle_tile_packing_lower_bound" and row["tile_capacity"] == "16"]
    m3_oracle = [float(row["value"]) for row in final_rows if row["experiment"] == "M3_tile_scattering" and row["metric"] == "access_amplification" and row["layout"] == "oracle_tile_packing_lower_bound" and row["tile_capacity"] == "16"]
    check("m3_nonoracle_overfetch", bool(m3_nonoracle) and bool(m3_oracle) and min(m3_nonoracle) > m3_oracle[0], min(m3_nonoracle) if m3_nonoracle else None, f">{m3_oracle[0] if m3_oracle else 'oracle'}")
    s2_mismatch = [float(row["value"]) for row in final_rows if row["experiment"] == "S2_generic_content_maintenance" and row["metric"] == "lookup_mismatch_count"]
    check("s2_exact_lookup", bool(s2_mismatch) and max(s2_mismatch) == 0, max(s2_mismatch) if s2_mismatch else None, 0)
    check("s2_appendix_role", all(row["claim_role"] == "appendix_only" for row in final_rows if row["experiment"] == "S2_generic_content_maintenance"), sorted({row["claim_role"] for row in final_rows if row["experiment"] == "S2_generic_content_maintenance"}), "appendix_only")

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    validation_csv = ROOT / "aggregate/v5p1_validation.csv"
    with validation_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "observed", "expected", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    validation_json = ROOT / "aggregate/v5p1_validation.json"
    validation_json.write_text(json.dumps({"status": status, "checks": checks}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": len(checks), "failed": [row["check"] for row in checks if row["status"] != "pass"]}, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
