from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

VERSION_ROOT = Path(__file__).resolve().parents[1]
V24_ROOT = VERSION_ROOT.parents[1] / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.incremental import IncrementalSemanticOverlay  # noqa: E402
from v24.io import iter_jsonl, write_json  # noqa: E402


def main() -> None:
    keyed = {}
    for path in sorted((VERSION_ROOT / "runs" / "v26_train_oof_qwen").glob("shard_*.jsonl")):
        for row in iter_jsonl(path):
            if row["method"] == "adaptive_rule":
                keyed[row["query_id"]] = row
    rows = list(keyed.values())
    if len(rows) != 1355:
        raise RuntimeError(f"Qwen OOF replay requires 1355 adaptive rows, got {len(rows)}")
    by_namespace: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_namespace[row["namespace_id"]].append(row)
    namespace_summary = {}
    for namespace, values in sorted(by_namespace.items()):
        namespace_summary[namespace] = {
            "query_count": len(values),
            "ndcg_at_10": float(np.mean([row["metrics"]["ndcg_at_10"] for row in values])),
            "candidate_recall_frontier": float(np.mean([row["metrics"]["candidate_recall_frontier"] for row in values])),
            "safety_violations": int(sum(sum(row["safety"].values()) for row in values)),
        }
    rng = np.random.default_rng(20260718)
    prototypes = rng.normal(size=(16, 32)).astype(np.float32)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
    base_embeddings = rng.normal(size=(32, 32)).astype(np.float32)
    base_incidence = sparse.csr_matrix((32, 16), dtype=np.float32)
    overlay = IncrementalSemanticOverlay([f"base-{i}" for i in range(32)], base_embeddings.copy(), base_incidence, prototypes)
    frozen_prefix = overlay.embeddings.copy()
    inserted_positions = []
    for index in range(100):
        inserted_positions.append(overlay.insert_memory(f"increment-{index}", rng.normal(size=32).astype(np.float32)))
    incremental = {
        "insert_count": 100,
        "positions_contiguous": inserted_positions == list(range(32, 132)),
        "old_embedding_rows_unchanged": bool(np.array_equal(frozen_prefix, overlay.embeddings[:32])),
        "new_incidence_rows_normalized": bool(np.allclose(np.asarray(overlay.incidence[32:].sum(axis=1)).reshape(-1), 1.0)),
        "duplicate_insert_rejected": False,
    }
    try:
        overlay.insert_memory("increment-0", rng.normal(size=32).astype(np.float32))
    except ValueError:
        incremental["duplicate_insert_rejected"] = True
    incremental["passed"] = all(incremental[key] for key in ("positions_contiguous", "old_embedding_rows_unchanged", "new_incidence_rows_normalized", "duplicate_insert_rejected"))
    replay = {
        "status": "PASS" if not any(value["safety_violations"] for value in namespace_summary.values()) else "FAIL",
        "source": "strict OOF adaptive-rule Qwen rows replayed in namespace streams",
        "native_agent_access_claim": False,
        "query_count": len(rows),
        "namespace_count": len(namespace_summary),
        "namespace_summary": namespace_summary,
        "safety_violations": int(sum(value["safety_violations"] for value in namespace_summary.values())),
    }
    write_json(VERSION_ROOT / "aggregate" / "agent_replay_summary.json", replay)
    write_json(VERSION_ROOT / "aggregate" / "incremental_update_summary.json", incremental)
    print(json.dumps({"replay": replay["status"], "incremental": incremental["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
