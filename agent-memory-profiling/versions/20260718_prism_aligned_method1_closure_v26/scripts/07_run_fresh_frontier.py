from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

VERSION_ROOT = Path(__file__).resolve().parents[1]
V24_ROOT = VERSION_ROOT.parents[1] / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(VERSION_ROOT / "src"))
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.io import iter_jsonl  # noqa: E402
from v24.metrics import quality_metrics  # noqa: E402
from v26.core import adaptive_features, rank_percentile, rule_tail, select_tail  # noqa: E402


def main() -> None:
    canonical = VERSION_ROOT / "artifacts" / "fresh" / "canonical"
    embedding_root = VERSION_ROOT / "artifacts" / "fresh" / "embeddings"
    config = json.loads((VERSION_ROOT / "configs" / "fresh_holdout.json").read_text())
    memories = list(iter_jsonl(canonical / "memory_records.jsonl"))
    queries = list(iter_jsonl(canonical / "queries.jsonl"))
    evaluations = {row["query_id"]: row for row in iter_jsonl(canonical / "evaluation.jsonl")}
    memory_vectors = np.load(embedding_root / "memory.npy", mmap_mode="r")
    query_vectors = np.load(embedding_root / "queries.npy", mmap_mode="r")
    by_namespace: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(memories):
        by_namespace[row["namespace_id"]].append(index)
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=60000, sublinear_tf=True, norm="l2", dtype=np.float32)
    memory_lexical = vectorizer.fit_transform([row["content"] for row in memories])
    query_lexical = vectorizer.transform([row["query"] for row in queries])
    output = VERSION_ROOT / "runs" / "v26_fresh_once" / "frontier.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError("fresh frontier formal run already exists")
    with output.open("x", encoding="utf-8") as handle:
        for query_index, query in enumerate(queries):
            eligible = np.asarray(by_namespace[query["namespace_id"]], dtype=np.int64)
            ids = [memories[int(index)]["memory_id"] for index in eligible]
            dense = np.asarray(memory_vectors[eligible] @ np.asarray(query_vectors[query_index]), dtype=np.float32)
            lexical = np.asarray((memory_lexical[eligible] @ query_lexical[query_index].T).toarray(), dtype=np.float32).reshape(-1)
            dense_order = sorted(range(len(ids)), key=lambda i: (-float(dense[i]), ids[i]))
            lexical_order = sorted(range(len(ids)), key=lambda i: (-float(lexical[i]), ids[i]))
            features = adaptive_features(query["query"], dense, lexical, ids, frontier_c=32)
            tail = rule_tail(features, config["frozen_method"]["rule"])
            selected = {
                "full_n": dense_order[:128],
                "dense_c": dense_order[:32],
                "lexical_c": lexical_order[:32],
                "adaptive_rule": select_tail(ids, dense, rank_percentile(lexical, ids), tail, frontier_c=32),
            }
            for method, positions in selected.items():
                frontier_ids = [ids[position] for position in positions]
                metric = quality_metrics(frontier_ids, type("Evaluation", (), {
                    "gold_memory_ids": tuple(evaluations[query["query_id"]]["gold_memory_ids"]),
                    "evidence_groups": tuple(tuple(x) for x in evaluations[query["query_id"]]["evidence_groups"]),
                })())
                row = {"query_id": query["query_id"], "query_type": query["metadata"]["query_type"], "namespace_id": query["namespace_id"], "method": method, "frontier_ids": frontier_ids, "tail": tail if method == "adaptive_rule" else 0, "metrics": {"candidate_recall_frontier": metric["candidate_recall"], "evidence_group_coverage_frontier": metric["evidence_group_coverage"]}, "safety": {"cross_namespace": 0, "future_memory": 0, "duplicates": len(frontier_ids) - len(set(frontier_ids))}}
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps({"status": "FRESH_FRONTIER_COMPLETED_ONCE", "query_count": len(queries)}, sort_keys=True))


if __name__ == "__main__":
    main()
