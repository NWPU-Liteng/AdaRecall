#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import V19_ROOT, V20R1_ROOT, load_config, read_json, resolve, sha256_file, write_json  # noqa: E402


def main() -> None:
    config_names = ["v21_data.json", "v21_uot_semantic.json", "v21_evidence_association.json", "v21_freeze.json"]
    report_names = [
        "METHOD1_V21_PIVOT_AND_DATASET_DECISION.md",
        "METHOD1_V21_HYPOTHESES_AND_GATES.md",
        "METHOD1_V21_EVIDENCE_SET_ASSOCIATION_SPEC.md",
        "METHOD1_V21_DATA_LEAKAGE_AUDIT.md",
        "METHOD1_V21_LACA_EXTENSION_DATA_AUDIT.md",
    ]
    for path in [*(VERSION_ROOT / "configs" / name for name in config_names), *(VERSION_ROOT / "reports" / name for name in report_names)]:
        if not path.is_file():
            raise RuntimeError(f"missing v21 preregistration input: {path}")
    data = load_config("v21_data.json")
    references = {
        "v19_stop_record": V19_ROOT / "aggregate/v19_stop_record.json",
        "v20r1_stop_record": V20R1_ROOT / "aggregate/v20r1_stop_record.json",
        "v20r1_final_freeze": V20R1_ROOT / "reports/METHOD1_V20R1_FINAL_FREEZE.md",
        "split_manifest": V19_ROOT / "artifacts/keypoint_basis_v6_schedule_layout_candidate/query_split/query_split_manifest.json",
        "train_plan": resolve(data["split"]["train_query_plan"]),
        "memory_embeddings": resolve(data["memory"]["embedding_vectors"]),
        "memory_embedding_ids": resolve(data["memory"]["embedding_ids"]),
        "train_query_embeddings": resolve(data["query_embeddings"]["vectors"]),
        "train_query_embedding_index": resolve(data["query_embeddings"]["index"]),
        "frozen_train_retrieval": resolve(data["frozen_baseline"]["train_retrieval"]),
    }
    immutable = {}
    for role, path in references.items():
        if not path.is_file():
            raise RuntimeError(f"missing immutable reference {role}: {path}")
        immutable[role] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    split = read_json(references["split_manifest"])
    expected = {"train": 1355, "calibration": 290, "test": 290}
    if split.get("actual_query_counts") != expected:
        raise RuntimeError("fixed split count mismatch")
    manifest = {
        "schema_version": "v21-manifest-1",
        "version_id": VERSION_ROOT.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "phase_0_preregistered_fsa_qesa_formal_test_locked",
        "method_scope": "fuzzy_semantic_association_query_conditioned_evidence_hypergraph",
        "excluded_scope": ["laca_oach_main_claim", "causal_access_or_reasoning_chain_claim", "stage_inputs", "manual_linear_weights", "dynamic_physical_layout", "pim"],
        "formal_test": {"locked": True, "runs": 0, "path_read": False},
        "calibration": {"locked": True, "path_read": False},
        "split_counts_from_metadata": split["actual_query_counts"],
        "query_id_sha256": split["query_id_sha256"],
        "immutable_references": immutable,
        "config_hashes": {name: sha256_file(VERSION_ROOT / "configs" / name) for name in config_names},
        "report_hashes": {name: sha256_file(VERSION_ROOT / "reports" / name) for name in report_names},
        "gates": {"A_fsa": "not_evaluated", "B_qesa": "not_run", "C_qhds": "not_run", "D_crc": "not_run", "E_dynamic": "not_run"},
    }
    write_json("aggregate/v21_manifest.json", manifest)
    for directory in ("artifacts/v21_projection", "artifacts/v21_uot", "artifacts/v21_qhds", "artifacts/v21_dynamic"):
        resolve(directory).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"status": manifest["status"], "formal_test": manifest["formal_test"]}, indent=2))


if __name__ == "__main__":
    main()
