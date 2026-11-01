"""Run a no-device target-architecture closure over the frozen 200-question replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pim_backend.architecture_closure import (
    EvidencePimArchitectureConfig,
    project_rows,
    project_joint_retrieval_rows,
    sensitivity,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speedup-gate", type=float, default=1.05)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--pim-profile", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output required")
    raw = args.replay.read_bytes()
    source = json.loads(raw)
    if source.get("schema") != "evidence-state-route-replay-v1":
        raise ValueError("frozen evidence-state replay required")
    rows = source.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty replay rows required")
    config = EvidencePimArchitectureConfig()
    result = project_rows(rows, config, speedup_gate=args.speedup_gate)
    result["source_replay_sha256"] = hashlib.sha256(raw).hexdigest()
    result["sensitivity"] = sensitivity(rows, config, speedup_gate=args.speedup_gate)
    if args.index_root:
        vector_counts = {}
        for user_id in {str(row["person_id"]) for row in rows}:
            path = args.index_root / f"user_{user_id}" / "index.json"
            payload = json.loads(path.read_text())
            vector_counts[user_id] = int(payload["vector_count"])
        result["joint_retrieval_sensitivity"] = [
            project_joint_retrieval_rows(
                rows, vector_counts, config,
                retrieval_scan_fraction=fraction,
                speedup_gate=args.speedup_gate,
            )
            for fraction in (0.0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0)
        ]
        low, high = 0.0, 1.0
        for _ in range(24):
            mid = (low + high) / 2.0
            probe = project_joint_retrieval_rows(
                rows, vector_counts, config,
                retrieval_scan_fraction=mid,
                speedup_gate=args.speedup_gate,
            )
            if probe["positive_architecture_gate_pass"]:
                high = mid
            else:
                low = mid
        result["minimum_scan_fraction_for_positive_gate"] = high
    result["source_boundary"] = source.get("summary", {}).get("boundary")
    if args.pim_profile:
        profile_raw = args.pim_profile.read_bytes()
        profile = json.loads(profile_raw)
        if profile.get("schema") != "mobilemem-omni-main-pim-profile-v1":
            raise ValueError("main retrieval PIM profile required")
        profile_rows = profile.get("rows", [])
        if not profile_rows:
            raise ValueError("nonempty retrieval PIM profile required")
        replay_ids = {row["question_id"] for row in rows}
        profile_ids = {row["question_id"] for row in profile_rows}
        if replay_ids != profile_ids:
            raise ValueError("retrieval PIM profile and replay question identities differ")
        measured_scan = sum(float(row["fp32_host_score_and_topk_seconds_median"])
                            for row in profile_rows)
        modeled_pim = sum(float(row["pim_model_cycles_int8"])
                          for row in profile_rows) / float(
                              profile.get("model_config", {}).get("clock_hz", 1_000_000_000.0))
        retrieval_total = sum(float(row["event_seconds"]["query_parse_and_retrieval"])
                              for row in rows)
        result["retrieval_profile_alignment"] = {
            "profile_sha256": hashlib.sha256(profile_raw).hexdigest(),
            "profile_requests": int(profile["requests"]),
            "profile_query_vector_scans": int(profile["query_vector_scans"]),
            "measured_host_scan_seconds_sum_of_query_medians": measured_scan,
            "modeled_pim_scan_seconds": modeled_pim,
            "main_retrieval_seconds_total": retrieval_total,
            "scan_time_to_main_retrieval_ratio_proxy": measured_scan / retrieval_total,
            "modeled_scan_speedup": profile.get("modeled_warm_scan_speedup"),
            "quality_overlap_mean": profile.get("mean_topk_overlap_vs_fp32_host"),
            "boundary": (
                "same question identities and frozen vectors, but scan microbenchmark is "
                "not an in-service stage attribution; ratio is diagnostic, not E2E evidence"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "requests": result["requests"],
        "metrics": result["metrics"],
        "gates": result["gates"],
        "positive_architecture_gate_pass": result["positive_architecture_gate_pass"],
        "sensitivity": result["sensitivity"],
        "joint_retrieval_sensitivity": [
            {"fraction": item["retrieval_scan_fraction_assumed"],
             "metrics": item["metrics"],
             "gates": item["gates"],
             "positive_architecture_gate_pass": item["positive_architecture_gate_pass"]}
            for item in result.get("joint_retrieval_sensitivity", [])
        ],
        "minimum_scan_fraction_for_positive_gate": result.get(
            "minimum_scan_fraction_for_positive_gate"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
