#!/usr/bin/env python3
"""Validate the closed motivation bundle and its claim guardrails."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    protocol = json.loads((ROOT / "configs" / "v5_protocol.json").read_text(encoding="utf-8"))
    guard = protocol["validation_guardrails"]
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})

    s1 = read_csv(ROOT / "aggregate" / "v5_s1_evidence_structure_per_query.csv")
    s1_summary = read_csv(ROOT / "aggregate" / "v5_s1_evidence_structure_summary.csv")
    s1_all = next(row for row in s1_summary if row["query_type"] == "all")
    s1_multi = next(row for row in s1_summary if row["query_type"] == "multi_hop_clinical_deduction")
    required_types = set(protocol["s1"]["required_query_types"])
    observed_types = {row["query_type"] for row in s1}
    multi_group_fraction = sum(int(float(row["evidence_group_count"]) > 1) for row in s1) / float(len(s1))
    check("s1_query_count", len(s1) >= int(protocol["s1"]["minimum_query_count"]), f"n={len(s1)}")
    check("s1_query_types", required_types.issubset(observed_types), f"missing={sorted(required_types - observed_types)}")
    check("s1_multi_group_presence", multi_group_fraction >= float(guard["s1_min_multi_group_fraction"]), f"fraction={multi_group_fraction:.4f}")
    check("s1_multi_hop_flat_limit", float(s1_multi["flat_all_groups_b8_rate"]) <= float(guard["s1_multi_hop_b8_all_group_rate_max"]), f"rate={s1_multi['flat_all_groups_b8_rate']}")
    check("s1_summary_consistency", int(s1_all["n"]) == len(s1), f"summary_n={s1_all['n']}")

    s2 = read_csv(ROOT / "aggregate" / "v5_s2_generic_content_maintenance.csv")
    s2_result = json.loads((ROOT / "aggregate" / "v5_s2_generic_content_maintenance_summary.json").read_text(encoding="utf-8"))
    at_20 = next(row for row in s2 if int(row["growth_percent_of_full_corpus"]) == 20)
    check("s2_growth_points", [int(row["growth_percent_of_full_corpus"]) for row in s2] == [1, 5, 10, 20], f"points={[row['growth_percent_of_full_corpus'] for row in s2]}")
    check("s2_exact_lookup", bool(s2_result["lookup_exactness"]) == bool(guard["s2_require_exact_lookup"]), f"exact={s2_result['lookup_exactness']}")
    check("s2_changed_posting_bound", float(at_20["changed_posting_fraction_of_full"]) <= float(guard["s2_max_changed_posting_fraction_at_20_percent"]), f"fraction={at_20['changed_posting_fraction_of_full']}")
    check("s2_rebuild_cost", all(float(row["full_to_incremental_time_ratio"]) >= float(guard["s2_min_full_to_incremental_time_ratio"]) for row in s2), f"ratios={[row['full_to_incremental_time_ratio'] for row in s2]}")

    m3 = read_csv(ROOT / "aggregate" / "v5_reused_m3_tile_summary.csv")
    oracle = {(int(row["tile_capacity"]), row["layout"]): float(row["activated_tiles_mean"]) for row in m3}
    non_oracle_ok = True
    details = []
    for capacity in sorted({key[0] for key in oracle}):
        lower = oracle[(capacity, "oracle_tile_packing_lower_bound")]
        for layout in ("chronological", "random_per_persona"):
            value = oracle[(capacity, layout)]
            non_oracle_ok &= value >= lower
            details.append(f"{capacity}/{layout}={value:.3f} vs oracle={lower:.3f}")
    check("m3_tile_lower_bound", non_oracle_ok, "; ".join(details))

    expected_figures = [
        "Fig_V5_M1_history_pressure.png", "Fig_V5_M2_flat_frontier_tradeoff.png",
        "Fig_V5_M3_tile_activation.png", "Fig_V5_S1_evidence_structure.png",
        "Fig_V5_S2_content_maintenance.png",
    ]
    check("figures", all((ROOT / "figures" / name).exists() for name in expected_figures), f"missing={[name for name in expected_figures if not (ROOT / 'figures' / name).exists()]}")
    check("claim_boundary", all(value is False for value in (protocol["claim_boundaries"]["stage_or_workflow_claim"], protocol["claim_boundaries"]["method_superiority_claim"], protocol["claim_boundaries"]["pim_hardware_claim"], protocol["claim_boundaries"]["causal_access_stream_claim"], protocol["claim_boundaries"]["dynamic_access_chain_claim"])), "stage/method/PIM/access claims disabled")

    output = ROOT / "aggregate" / "v5_validation.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(checks)
    result = {
        "schema_version": "motivation-v5-validation-1",
        "status": "passed" if all(row["status"] == "pass" for row in checks) else "failed",
        "checks": checks,
        "validation_csv": str(output.relative_to(ROOT)),
    }
    (ROOT / "aggregate" / "v5_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
