from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
V24_ROOT = PROJECT_ROOT / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.dataset import load_canonical  # noqa: E402
from v24.metrics import quality_metrics  # noqa: E402


def _partition(query_id: str) -> str:
    digest = hashlib.sha256(("v25-development-partition:" + query_id).encode("utf-8")).digest()
    return "tuning" if digest[0] % 2 == 0 else "validation"


def _rank(values: np.ndarray, ids: list[str]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), ids[index]))
    output = np.empty(len(values), dtype=np.float32)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        output[index] = float(rank) / float(denominator)
    return output


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _bootstrap_ci(values: list[float], seed: int, samples: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(samples)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _evaluate(features: list[dict[str, Any]], lexical_weight: float, cap: int = 32) -> dict[str, Any]:
    rows = []
    for feature in features:
        fused = (1.0 - lexical_weight) * feature["dense_rank"] + lexical_weight * feature["lexical_rank"]
        selected = sorted(
            range(len(fused)), key=lambda index: (-float(fused[index]), feature["ids"][index])
        )[:cap]
        ids = [feature["ids"][index] for index in selected]
        ours = quality_metrics(ids, feature["evaluation"])
        dense = quality_metrics(feature["dense_ids"], feature["evaluation"])
        rows.append(
            {
                "query_id": feature["query_id"],
                "candidate_recall_delta": float(ours["candidate_recall"] - dense["candidate_recall"]),
                "group_coverage_delta": float(ours["evidence_group_coverage"] - dense["evidence_group_coverage"]),
                "replacements": len(set(ids) - set(feature["dense_ids"])),
            }
        )
    return {
        "lexical_weight": float(lexical_weight),
        "query_count": len(rows),
        "candidate_recall_delta": _mean([row["candidate_recall_delta"] for row in rows]),
        "group_coverage_delta": _mean([row["group_coverage_delta"] for row in rows]),
        "win_rate": _mean(
            [float(row["candidate_recall_delta"] > 0.0 or row["group_coverage_delta"] > 0.0) for row in rows]
        ),
        "loss_rate": _mean(
            [float(row["candidate_recall_delta"] < 0.0 or row["group_coverage_delta"] < 0.0) for row in rows]
        ),
        "mean_replacements": _mean([float(row["replacements"]) for row in rows]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--weights", default="0.05,0.10,0.20,0.30,0.40,0.50")
    args = parser.parse_args()
    dataset = load_canonical(V24_ROOT / "inputs", split="development")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=int(args.max_features),
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    memory_matrix = vectorizer.fit_transform([memory.content for memory in dataset.memories])
    query_matrix = vectorizer.transform([query.query for query in dataset.queries["development"]])
    embeddings = np.array(dataset.embeddings, dtype=np.float32, copy=True)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    features = []
    for query_index, query in enumerate(dataset.queries["development"]):
        eligible = dataset.eligible_positions(query)
        query_vector = np.array(dataset.query_vector("development", query.query_id), dtype=np.float32, copy=True)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        dense = np.asarray(embeddings[eligible] @ query_vector, dtype=np.float32)
        lexical = np.asarray((memory_matrix[eligible] @ query_matrix[query_index].T).toarray()).reshape(-1)
        ids = [dataset.memories[int(position)].memory_id for position in eligible]
        dense_order = sorted(range(len(ids)), key=lambda index: (-float(dense[index]), ids[index]))
        features.append(
            {
                "query_id": query.query_id,
                "partition": _partition(query.query_id),
                "ids": ids,
                "dense_rank": _rank(dense, ids),
                "lexical_rank": _rank(lexical, ids),
                "dense_ids": [ids[index] for index in dense_order[:32]],
                "evaluation": dataset.evaluations["development"][query.query_id],
            }
        )

    tuning = [feature for feature in features if feature["partition"] == "tuning"]
    validation = [feature for feature in features if feature["partition"] == "validation"]
    weights = [float(value) for value in args.weights.split(",")]
    tuning_results = [_evaluate(tuning, weight) for weight in weights]
    tuning_results.sort(
        key=lambda row: (
            -float(row["candidate_recall_delta"] + row["group_coverage_delta"]),
            float(row["loss_rate"]),
            float(row["mean_replacements"]),
            float(row["lexical_weight"]),
        )
    )
    selected = tuning_results[0]
    validation_result = _evaluate(validation, float(selected["lexical_weight"]))
    validation_rows = validation_result.pop("rows")
    recall_values = [float(row["candidate_recall_delta"]) for row in validation_rows]
    coverage_values = [float(row["group_coverage_delta"]) for row in validation_rows]
    promoted = bool(
        selected["candidate_recall_delta"] > 0.0
        and validation_result["candidate_recall_delta"] >= 0.0
        and validation_result["group_coverage_delta"] >= 0.0
        and validation_result["loss_rate"] <= validation_result["win_rate"]
    )
    report = {
        "status": "PROMOTED_TO_DEVELOPMENT_RERANKER" if promoted else "NOT_PROMOTED",
        "scientific_status": "development_tuning_validation_no_test_read",
        "test_read": False,
        "model": {
            "kind": "tfidf_word_1_2",
            "vocabulary_size": len(vectorizer.vocabulary_),
            "max_features": int(args.max_features),
            "trained_on": "memory_text_without_labels",
        },
        "partition_counts": {"tuning": len(tuning), "validation": len(validation)},
        "selected_weight": float(selected["lexical_weight"]),
        "tuning_result": {key: value for key, value in selected.items() if key != "rows"},
        "validation_result": validation_result,
        "validation_ci": {
            "candidate_recall_delta_95": _bootstrap_ci(recall_values, 20260718),
            "group_coverage_delta_95": _bootstrap_ci(coverage_values, 20260719),
        },
        "promotion_passed": promoted,
    }
    output = VERSION_ROOT / "aggregate" / "lexical_safety_frontier_optimization.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
