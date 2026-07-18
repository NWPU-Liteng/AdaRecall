from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
V24_ROOT = PROJECT_ROOT / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(VERSION_ROOT / "src"))
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.dataset import load_canonical  # noqa: E402
from v24.io import iter_jsonl, write_json  # noqa: E402
from v24.metrics import quality_metrics  # noqa: E402
from v26.core import (  # noqa: E402
    HistoryEdge,
    TAIL_OPTIONS,
    adaptive_features,
    calibrated_edge_scores,
    calibrated_history_scores,
    oracle_tail_label,
    rank_percentile,
    rule_tail,
    safety_counts,
    select_tail,
)


METHODS = (
    "full_n",
    "dense_c",
    "lexical_c",
    "v25_fixed",
    "adaptive_rule",
    "adaptive_logistic",
    "semantic",
    "oof_history",
    "adaptive_all_graph",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(values: list[float], seed: int, samples: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    chunk = 500
    means = []
    for start in range(0, int(samples), chunk):
        count = min(chunk, int(samples) - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means.extend(np.asarray(array[indices].mean(axis=1), dtype=np.float64).tolist())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def fold_assignments(queries: list[Any], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray([str(query.metadata.get("query_type", "unknown")) for query in queries])
    groups = np.asarray([f"{query.namespace_id}:{query.cutoff_order}" for query in queries])
    indices = np.arange(len(queries), dtype=np.int64)
    fold = np.full(len(queries), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
    for fold_id, (_, heldout) in enumerate(splitter.split(indices, labels, groups)):
        fold[heldout] = int(fold_id)
    if np.any(fold < 0):
        raise RuntimeError("incomplete fold assignment")
    group_folds: dict[str, set[int]] = defaultdict(set)
    for group, value in zip(groups, fold):
        group_folds[str(group)].add(int(value))
    if any(len(values) != 1 for values in group_folds.values()):
        raise RuntimeError("correlated namespace/cutoff group crossed OOF folds")
    manifest = {
        "algorithm": "StratifiedGroupKFold(n_splits=5, shuffle=True)",
        "random_seed": int(seed),
        "group": "namespace_id:cutoff_order",
        "stratum": "query_type",
        "fold_counts": {str(i): int(np.sum(fold == i)) for i in range(5)},
        "group_count": len(group_folds),
        "group_cross_fold_violations": 0,
        "fold_query_ids": {
            str(i): [queries[index].query_id for index in np.flatnonzero(fold == i)] for i in range(5)
        },
    }
    return fold, manifest


def load_history() -> list[HistoryEdge]:
    index = V24_ROOT / "runs" / "v24_index_20260718" / "index"
    rows = read_json(index / "history_edges.json")
    vectors = np.load(index / "history_edge_vectors.npy", mmap_mode="r")
    return [
        HistoryEdge(
            query_id=str(row["query_id"]),
            namespace_id=str(row["namespace_id"]),
            arrival_order=int(row["arrival_order"]),
            members=tuple(str(value) for value in row["members"]),
            vector=np.asarray(vectors[index_value], dtype=np.float32),
        )
        for index_value, row in enumerate(rows)
    ]


def metric_row(ids: list[str], evaluation: Any, *, final_ids: list[str] | None = None) -> dict[str, float]:
    frontier = quality_metrics(ids, evaluation)
    final = quality_metrics(final_ids if final_ids is not None else ids[:5], evaluation, k=5)
    return {
        "candidate_recall_frontier": float(frontier["candidate_recall"]),
        "evidence_group_coverage_frontier": float(frontier["evidence_group_coverage"]),
        "candidate_recall_final": float(final["candidate_recall"]),
        "evidence_group_coverage_final": float(final["evidence_group_coverage"]),
        "precision_at_k": float(final["precision_at_k"]),
        "ndcg_at_10": float(final["ndcg_at_10"]),
    }


def paired(method_rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    keys = (
        "candidate_recall_frontier",
        "evidence_group_coverage_frontier",
        "candidate_recall_final",
        "evidence_group_coverage_final",
        "precision_at_k",
        "ndcg_at_10",
    )
    output = {}
    for offset, key in enumerate(keys):
        values = [float(row["metrics"][key]) - float(baseline[row["query_id"]]["metrics"][key]) for row in method_rows]
        output[key] = {
            "mean_delta": mean(values),
            "ci_95": bootstrap_ci(values, seed + offset, samples),
            "win_rate": mean([float(value > 0.0) for value in values]),
            "loss_rate": mean([float(value < 0.0) for value in values]),
        }
    return output


def main() -> None:
    config = read_json(VERSION_ROOT / "configs" / "v26.json")
    boundary = config["scientific_boundary"]
    if boundary["selection_split"] != "train_strict_5fold_oof" or set(boundary["forbidden_splits"]) != {"development", "test"}:
        raise RuntimeError("v26 OOF boundary is not fail-closed")

    started = perf_counter()
    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    queries = dataset.queries["train"]
    folds, fold_manifest = fold_assignments(queries, int(config["random_seed"]))
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "oof_fold_manifest.json", fold_manifest)
    heldout_ids = {fold: {queries[index].query_id for index in np.flatnonzero(folds == fold)} for fold in range(5)}
    all_history_ids = {str(row["query_id"]) for row in iter_jsonl(V24_ROOT / "inputs" / "history_events.jsonl")}
    exclusion_manifest = {
        "source_event_count": len(all_history_ids),
        "source_kind": "co_candidate_not_native_agent_access",
        "gold_used_to_build_history": False,
        "folds": {},
    }
    for fold in range(5):
        expected = heldout_ids[fold] & all_history_ids
        remaining = all_history_ids - heldout_ids[fold]
        exclusion_manifest["folds"][str(fold)] = {
            "heldout_query_count": len(heldout_ids[fold]),
            "excluded_matching_event_count": len(expected),
            "heldout_event_remaining_count": len(expected & remaining),
            "passed": not bool(expected & remaining),
        }
    if not all(row["passed"] for row in exclusion_manifest["folds"].values()):
        raise RuntimeError("held-out history exclusion failed")
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "history_exclusion_manifest.json", exclusion_manifest)

    lexical_config = config["lexical"]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=tuple(lexical_config["ngram_range"]),
        min_df=int(lexical_config["min_df"]),
        max_df=float(lexical_config["max_df"]),
        max_features=int(lexical_config["max_features"]),
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    lexical_started = perf_counter()
    memory_matrix = vectorizer.fit_transform([memory.content for memory in dataset.memories])
    query_matrix = vectorizer.transform([query.query for query in queries])
    lexical_build_ms = float((perf_counter() - lexical_started) * 1000.0)

    embeddings = np.asarray(dataset.embeddings, dtype=np.float32)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    index = V24_ROOT / "runs" / "v24_index_20260718" / "index"
    prototypes = np.load(index / "semantic_prototypes.npy", mmap_mode="r")
    incidence = sparse.load_npz(index / "semantic_incidence.npz").tocsr()
    history_edges = load_history()
    memory_by_id = {memory.memory_id: memory for memory in dataset.memories}

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    cached: list[dict[str, Any]] = []
    graph_diag = Counter()
    for query_index, query in enumerate(queries):
        eligible = dataset.eligible_positions(query)
        eligible_ids = [dataset.memories[int(position)].memory_id for position in eligible]
        query_vector = np.asarray(dataset.query_vector("train", query.query_id), dtype=np.float32)
        query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
        dense = np.asarray(embeddings[eligible] @ query_vector, dtype=np.float32)
        lexical = np.asarray((memory_matrix[eligible] @ query_matrix[query_index].T).toarray(), dtype=np.float32).reshape(-1)
        dense_order = sorted(range(len(eligible_ids)), key=lambda i: (-float(dense[i]), eligible_ids[i]))
        lexical_rank = rank_percentile(lexical, eligible_ids)
        dense_rank = rank_percentile(dense, eligible_ids)
        semantic = calibrated_edge_scores(
            query_vector,
            eligible[np.asarray(dense_order[:16], dtype=np.int64)],
            eligible,
            prototypes,
            incidence,
            minimum_seed_support=int(config["graph_repair"]["minimum_seed_support"]),
            edge_size_penalty_power=float(config["graph_repair"]["edge_size_penalty_power"]),
        )
        history, history_diag = calibrated_history_scores(
            query_vector,
            [eligible_ids[index_value] for index_value in dense_order[:16]],
            eligible_ids,
            history_edges,
            namespace_id=query.namespace_id,
            cutoff_order=query.cutoff_order,
            excluded_query_ids=heldout_ids[int(folds[query_index])],
            minimum_seed_support=int(config["graph_repair"]["minimum_seed_support"]),
        )
        graph_diag.update(history_diag)
        semantic_rank = rank_percentile(semantic, eligible_ids) if np.any(semantic) else np.zeros(len(eligible_ids), dtype=np.float32)
        history_rank = rank_percentile(history, eligible_ids) if np.any(history) else np.zeros(len(eligible_ids), dtype=np.float32)
        hybrid = (
            lexical_rank
            + float(config["graph_repair"]["hybrid_semantic_weight"]) * semantic_rank
            + float(config["graph_repair"]["hybrid_history_weight"]) * history_rank
        )
        features = adaptive_features(query.query, dense, lexical, eligible_ids, frontier_c=32)
        evaluation = dataset.evaluations["train"][query.query_id]
        option_ids: dict[int, list[str]] = {}
        option_metrics: dict[int, dict[str, float]] = {}
        for tail in TAIL_OPTIONS:
            selected = select_tail(eligible_ids, dense, lexical_rank, tail, frontier_c=32)
            ids = [eligible_ids[index_value] for index_value in selected]
            option_ids[int(tail)] = ids
            metrics = quality_metrics(ids, evaluation)
            option_metrics[int(tail)] = {
                "candidate_recall": float(metrics["candidate_recall"]),
                "evidence_group_coverage": float(metrics["evidence_group_coverage"]),
            }
        label = oracle_tail_label(option_metrics)
        feature_rows.append(features)
        labels.append(label)
        cached.append(
            {
                "eligible": eligible,
                "ids": eligible_ids,
                "dense": dense,
                "dense_rank": dense_rank,
                "lexical": lexical,
                "lexical_rank": lexical_rank,
                "semantic": semantic,
                "history": history,
                "hybrid": hybrid,
                "features": features,
                "option_ids": option_ids,
            }
        )
        if (query_index + 1) % 200 == 0:
            print(json.dumps({"stage": "features", "completed": query_index + 1, "total": len(queries)}), flush=True)

    feature_matrix = np.vstack(feature_rows)
    labels_array = np.asarray(labels, dtype=np.int64)
    logistic_predictions = np.zeros(len(queries), dtype=np.int64)
    model_manifest = {"kind": "multinomial_logistic_regression", "features": 10, "folds": {}}
    for fold in range(5):
        train_mask = folds != fold
        heldout_mask = folds == fold
        classes = np.unique(labels_array[train_mask])
        if len(classes) == 1:
            logistic_predictions[heldout_mask] = int(classes[0])
            model_kind = "constant_single_training_class"
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(config["random_seed"]) + fold,
                    solver="lbfgs",
                ),
            )
            model.fit(feature_matrix[train_mask], labels_array[train_mask])
            logistic_predictions[heldout_mask] = model.predict(feature_matrix[heldout_mask]).astype(np.int64)
            model_kind = "standardized_logistic"
        model_manifest["folds"][str(fold)] = {
            "training_query_count": int(train_mask.sum()),
            "heldout_query_count": int(heldout_mask.sum()),
            "training_label_counts": {str(key): int(value) for key, value in Counter(labels_array[train_mask]).items()},
            "heldout_ids_seen_in_training": 0,
            "model": model_kind,
        }
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "oof_logistic_manifest.json", model_manifest)

    row_path = VERSION_ROOT / "runs" / "v26_train_oof_frontier" / "query_results.jsonl"
    if row_path.exists():
        row_path.unlink()
    all_rows: list[dict[str, Any]] = []
    tail_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for query_index, (query, cache) in enumerate(zip(queries, cached)):
        ids = cache["ids"]
        dense = cache["dense"]
        dense_order = sorted(range(len(ids)), key=lambda i: (-float(dense[i]), ids[i]))
        lexical_order = sorted(range(len(ids)), key=lambda i: (-float(cache["lexical"][i]), ids[i]))
        full_selected = dense_order[:128]
        dense_selected = dense_order[:32]
        lexical_selected = lexical_order[:32]
        fixed_score = (1.0 - float(lexical_config["v25_fixed_weight"])) * cache["dense_rank"] + float(lexical_config["v25_fixed_weight"]) * cache["lexical_rank"]
        fixed_selected = sorted(range(len(ids)), key=lambda i: (-float(fixed_score[i]), ids[i]))[:32]
        rule_value = rule_tail(cache["features"], config["adaptive_rule"])
        logistic_value = int(logistic_predictions[query_index])
        selected_by_method = {
            "full_n": full_selected,
            "dense_c": dense_selected,
            "lexical_c": lexical_selected,
            "v25_fixed": fixed_selected,
            "adaptive_rule": select_tail(ids, dense, cache["lexical_rank"], rule_value, frontier_c=32),
            "adaptive_logistic": select_tail(ids, dense, cache["lexical_rank"], logistic_value, frontier_c=32),
            "semantic": select_tail(ids, dense, cache["semantic"], int(config["graph_repair"]["semantic_tail"]), frontier_c=32),
            "oof_history": select_tail(ids, dense, cache["history"], int(config["graph_repair"]["history_tail"]), frontier_c=32),
            "adaptive_all_graph": select_tail(ids, dense, cache["hybrid"], logistic_value, frontier_c=32),
        }
        tail_counts["adaptive_rule"][rule_value] += 1
        tail_counts["adaptive_logistic"][logistic_value] += 1
        evaluation = dataset.evaluations["train"][query.query_id]
        query_type = str(query.metadata.get("query_type", "unknown"))
        for method in METHODS:
            selected = selected_by_method[method]
            frontier_ids = [ids[index_value] for index_value in selected]
            final_selected = sorted(selected, key=lambda i: (-float(dense[i]), ids[i]))[:5]
            final_ids = [ids[index_value] for index_value in final_selected]
            safety = safety_counts(
                frontier_ids,
                memory_by_id,
                namespace_id=query.namespace_id,
                cutoff_order=query.cutoff_order,
            )
            row = {
                "query_id": query.query_id,
                "query_type": query_type,
                "namespace_id": query.namespace_id,
                "fold": int(folds[query_index]),
                "method": method,
                "frontier_ids": frontier_ids,
                "dense_reference_final_ids": final_ids,
                "metrics": metric_row(frontier_ids, evaluation, final_ids=final_ids),
                "safety": safety,
                "tail": int(
                    rule_value if method == "adaptive_rule" else logistic_value if method in {"adaptive_logistic", "adaptive_all_graph"} else 0
                ),
                "scientific_scope": "train_strict_5fold_oof",
            }
            append_jsonl(row_path, row)
            all_rows.append(row)

    by_method = {method: [row for row in all_rows if row["method"] == method] for method in METHODS}
    baselines = {
        method: {row["query_id"]: row for row in by_method[method]} for method in ("dense_c", "full_n")
    }
    samples = int(config["gate"]["bootstrap_samples"])
    summary_methods: dict[str, Any] = {}
    for method_index, method in enumerate(METHODS):
        rows = by_method[method]
        absolute = {
            key: mean([float(row["metrics"][key]) for row in rows])
            for key in rows[0]["metrics"]
        }
        fold_delta = {}
        for fold in range(5):
            fold_rows = [row for row in rows if row["fold"] == fold]
            fold_delta[str(fold)] = mean(
                [
                    float(row["metrics"]["candidate_recall_frontier"])
                    - float(baselines["dense_c"][row["query_id"]]["metrics"]["candidate_recall_frontier"])
                    for row in fold_rows
                ]
            )
        type_delta = {}
        for query_type in sorted({row["query_type"] for row in rows}):
            type_rows = [row for row in rows if row["query_type"] == query_type]
            type_delta[query_type] = mean(
                [
                    float(row["metrics"]["candidate_recall_frontier"])
                    - float(baselines["dense_c"][row["query_id"]]["metrics"]["candidate_recall_frontier"])
                    for row in type_rows
                ]
            )
        summary_methods[method] = {
            "absolute": absolute,
            "paired_vs_dense": paired(rows, baselines["dense_c"], samples, int(config["random_seed"]) + 100 * method_index),
            "paired_vs_full": paired(rows, baselines["full_n"], samples, int(config["random_seed"]) + 100 * method_index + 50),
            "candidate_delta_by_fold": fold_delta,
            "candidate_delta_by_query_type": type_delta,
            "nonnegative_fold_count": sum(value >= 0.0 for value in fold_delta.values()),
            "safety_violations": {
                key: int(sum(row["safety"][key] for row in rows)) for key in ("cross_namespace", "future_memory", "duplicates")
            },
        }

    promotable = []
    for method in ("adaptive_rule", "adaptive_logistic", "adaptive_all_graph"):
        result = summary_methods[method]
        candidate = result["paired_vs_dense"]["candidate_recall_frontier"]
        coverage = result["paired_vs_dense"]["evidence_group_coverage_frontier"]
        if (
            candidate["mean_delta"] >= 0.0
            and candidate["ci_95"][0] >= 0.0
            and coverage["mean_delta"] > 0.0
            and candidate["loss_rate"] <= candidate["win_rate"]
            and result["nonnegative_fold_count"] >= int(config["gate"]["nonnegative_fold_count_min"])
            and not any(result["safety_violations"].values())
        ):
            promotable.append(method)
    selected_method = max(
        promotable,
        key=lambda method: (
            summary_methods[method]["paired_vs_dense"]["evidence_group_coverage_frontier"]["mean_delta"],
            summary_methods[method]["paired_vs_dense"]["candidate_recall_frontier"]["mean_delta"],
            -METHODS.index(method),
        ),
        default=None,
    )
    graph_ablation = {
        "semantic_vs_dense_candidate_delta": summary_methods["semantic"]["paired_vs_dense"]["candidate_recall_frontier"],
        "history_vs_dense_candidate_delta": summary_methods["oof_history"]["paired_vs_dense"]["candidate_recall_frontier"],
        "all_graph_vs_adaptive_logistic_candidate_delta": paired(
            by_method["adaptive_all_graph"],
            {row["query_id"]: row for row in by_method["adaptive_logistic"]},
            samples,
            int(config["random_seed"]) + 999,
        )["candidate_recall_frontier"],
        "history_is_native_agent_access": False,
    }
    summary = {
        "status": "OOF_PROMOTED_TO_QWEN" if selected_method else "OOF_NO_CONFIGURATION_PROMOTED",
        "scientific_scope": "train strict five-fold OOF; v24 test and v25 development not read",
        "query_count": len(queries),
        "fold_count": 5,
        "methods": summary_methods,
        "promotable_methods": promotable,
        "selected_method": selected_method,
        "tail_counts": {method: {str(key): int(value) for key, value in sorted(counts.items())} for method, counts in tail_counts.items()},
        "oracle_label_counts": {str(key): int(value) for key, value in sorted(Counter(labels).items())},
        "graph_ablation": graph_ablation,
        "history_diagnostics": dict(graph_diag),
        "cost": {
            "lexical_offline_build_ms": lexical_build_ms,
            "frontier_candidate_count": 32,
            "full_candidate_count": 128,
            "candidate_reduction_vs_full": 0.75,
            "qwen_not_run_in_this_stage": True,
        },
        "runtime_seconds": float(perf_counter() - started),
        "forbidden_split_reads": 0,
    }
    write_json(VERSION_ROOT / "aggregate" / "oof_frontier_summary.json", summary)
    write_json(
        VERSION_ROOT / "artifacts" / "selected_oof_frontier.json",
        {
            "status": summary["status"],
            "selected_method": selected_method,
            "promotable_methods": promotable,
            "selection_source": "train_strict_5fold_oof",
            "tail_options": list(TAIL_OPTIONS),
            "v24_test_read": False,
            "v25_development_read": False,
        },
    )
    print(json.dumps({"status": summary["status"], "selected_method": selected_method, "runtime_seconds": summary["runtime_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
