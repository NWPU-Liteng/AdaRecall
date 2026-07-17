#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa import evidence_metrics, select_protected_frontier  # noqa: E402


def main() -> None:
    config = load_config("v21_qesa.json")
    data = load_config("v21_data.json")
    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load((VERSION_ROOT / data["memory"]["embedding_vectors"]).resolve(), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    helpers = runpy.run_path(str(VERSION_ROOT / "scripts" / "90_train_v21_qesa_oof.py"), run_name="v21_qesa_helpers")
    query_vectors = helpers["load_query_vectors"](data)
    rows_by_fold = helpers["load_rows"](data, query_vectors, int(config["outer_folds"]))
    run_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / config["run_id"]
    output_rows = []
    for seed in config["fixed_seeds"]:
        for fold in range(int(config["outer_folds"])):
            phase1_unit = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{seed}" / f"fold_{fold}"
            incidence = np.load(phase1_unit / "v21_uot_semantic_incidence.npy", allow_pickle=False)
            for row in rows_by_fold[fold]:
                group_affinities = []
                for group in row["groups"]:
                    member = incidence[[index_by_id[memory_id] for memory_id in group]]
                    affinity = member.max(axis=0)
                    affinity /= max(float(affinity.max()), 1e-12)
                    group_affinities.append(affinity)
                target = np.mean(group_affinities, axis=0)
                target /= max(float(target.sum()), 1e-12)
                eligible = [index_by_id[memory_id] for memory_id in row["eligible_memory_ids"]]
                eligible_array = np.asarray(eligible, dtype=np.int64)
                ann_scores = raw[eligible_array] @ row["query_embedding"]
                protected = eligible_array[np.argsort(-ann_scores, kind="stable")[:5]].tolist()
                eligible_scores = incidence[eligible_array] @ target
                score_by_index = np.full(len(ids), -np.inf, dtype=np.float32)
                score_by_index[eligible_array] = eligible_scores
                for budget in config["budgets"]:
                    selected = select_protected_frontier(score_by_index, eligible, protected, int(budget))
                    metrics = evidence_metrics([ids[position] for position in selected], row["groups"])
                    output_rows.append({
                        "query_id": row["query_id"],
                        "persona_id": row["persona_id"],
                        "query_session_id": row["query_session_id"],
                        "group_count": len(row["groups"]),
                        "seed": int(seed),
                        "fold": fold,
                        "budget": int(budget),
                        "oracle_type": "gold_evidence_prototype_target",
                        **metrics,
                    })

    source_metrics = VERSION_ROOT / "aggregate" / "v21_qesa_oof_metrics.csv"
    with source_metrics.open(encoding="utf-8", newline="") as handle:
        baseline_rows = list(csv.DictReader(handle))
    flat = {
        (int(row["seed"]), row["query_id"], int(row["budget"])): row
        for row in baseline_rows if row["method"] == "flat_cosine"
    }
    summary = {}
    for budget in map(int, config["budgets"]):
        selected = [row for row in output_rows if int(row["budget"]) == budget]
        multi = [row for row in selected if int(row["group_count"]) >= 2]
        all_delta = [
            float(row["group_coverage"])
            - float(flat[(int(row["seed"]), row["query_id"], budget)]["group_coverage"])
            for row in selected
        ]
        multi_delta = [
            float(row["group_coverage"])
            - float(flat[(int(row["seed"]), row["query_id"], budget)]["group_coverage"])
            for row in multi
        ]
        summary[str(budget)] = {
            "queries": len(selected),
            "multi_group_queries": len(multi),
            "oracle_mean_group_coverage": float(np.mean([row["group_coverage"] for row in selected])),
            "oracle_multi_group_coverage": float(np.mean([row["group_coverage"] for row in multi])),
            "oracle_multi_group_all_covered_rate": float(np.mean([row["all_groups_covered"] for row in multi])),
            "effect_oracle_minus_flat": float(np.mean(all_delta)),
            "multi_group_effect_oracle_minus_flat": float(np.mean(multi_delta)),
        }
    output_csv = VERSION_ROOT / "aggregate" / "v21_qesa_gold_prototype_headroom_r1.csv"
    write_csv_new(output_csv, output_rows)
    result = {
        "schema_version": "v21-qesa-gold-prototype-headroom-1",
        "oracle_type": "Oracle A: gold evidence coverage headroom",
        "deployable": False,
        "permitted_use": "representation headroom and error attribution only",
        "prohibited_use": "selector input, inference input, calibration, or formal-test model selection",
        "summary": summary,
        "headroom_above_flat_at_primary_budget": summary["16"]["effect_oracle_minus_flat"] > 0,
        "multi_group_headroom_above_flat_at_primary_budget": summary["16"]["multi_group_effect_oracle_minus_flat"] > 0,
        "rows_csv": str(output_csv.relative_to(VERSION_ROOT)),
        "hashes": {
            "rows": sha256_file(output_csv),
            "source_qesa_metrics": sha256_file(source_metrics),
        },
        "calibration_read": False,
        "formal_test_read": False,
    }
    output_json = VERSION_ROOT / "aggregate" / "v21_qesa_gold_prototype_headroom_r1.json"
    write_json_new(output_json, result)
    report = "# Method 1 v21 QESA Representation Headroom\n\n"
    report += "This is Oracle A, a gold evidence coverage oracle used only for train-OOF representation headroom and error attribution. It is not deployable and cannot be used by QESA inference, calibration, or formal test.\n\n"
    report += "| Budget | Oracle coverage | Oracle K>=2 coverage | Oracle minus flat | K>=2 oracle minus flat |\n"
    report += "| ---: | ---: | ---: | ---: | ---: |\n"
    for budget in map(int, config["budgets"]):
        row = summary[str(budget)]
        report += (
            f"| {budget} | {row['oracle_mean_group_coverage']:.6f} | {row['oracle_multi_group_coverage']:.6f} | "
            f"{row['effect_oracle_minus_flat']:+.6f} | {row['multi_group_effect_oracle_minus_flat']:+.6f} |\n"
        )
    report += "\nThe oracle reads heldout OOF evidence groups only to ask whether the frozen prototype incidence can express useful rankings. It does not establish learned generalization.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_QESA_REPRESENTATION_HEADROOM.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
