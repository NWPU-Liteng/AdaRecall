#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_json_new  # noqa: E402
from v21.sparse_semantic_index import build_prototype_rank_index  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one immutable v21 sparse semantic inverted index.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-id", default="v21_sparse_semantic_index_r1_20260714")
    parser.add_argument("--attempt", default="attempt_001")
    args = parser.parse_args()
    config, data = load_config("v21_sparse_joint.json"), load_config("v21_data.json")
    if args.run_id != config["index_run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("run ID, seed, or fold outside sparse-index preregistration")
    if not ATTEMPT_RE.fullmatch(args.attempt):
        raise ValueError("attempt must have the form attempt_NNN")
    unit_dir = VERSION_ROOT / "artifacts" / "v21_sparse_semantic_index" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)
    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    incidence_path = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, allow_pickle=False)
    if incidence.shape[0] != len(ids):
        raise ValueError("incidence memory axis does not match frozen memory IDs")
    ranks = build_prototype_rank_index(incidence)
    rank_path = unit_dir / "v21_prototype_memory_rank_index.npy"
    np.save(rank_path, ranks, allow_pickle=False)
    manifest = {
        "schema_version": "v21-sparse-semantic-index-unit-1", "status": "completed", "revision_id": config["revision_id"],
        "run_id": args.run_id, "attempt": args.attempt, "seed": args.seed, "fold": args.fold,
        "incidence_shape": list(map(int, incidence.shape)), "rank_index_shape": list(map(int, ranks.shape)),
        "metadata_bytes": int(ranks.nbytes), "online_contract": config["online_lookup"],
        "artifacts": {"prototype_memory_rank_index": str(rank_path.relative_to(VERSION_ROOT))},
        "hashes": {"rank_index": sha256_file(rank_path), "config": sha256_file(VERSION_ROOT / "configs" / "v21_sparse_joint.json"), "incidence_array": array_sha256(incidence)},
        "calibration_read": False, "formal_test_read": False,
    }
    write_json_new(unit_dir / "v21_sparse_semantic_index_manifest.json", manifest)
    print(json.dumps({"seed": args.seed, "fold": args.fold, "attempt": args.attempt, "metadata_bytes": int(ranks.nbytes)}, sort_keys=True))


if __name__ == "__main__":
    main()
