from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
V28 = ROOT.parent / "20260718_prism_aligned_method1_graph_only_v28"
V24_INPUTS = ROOT.parent / "20260718_prism_aligned_method1_closure_v24" / "inputs"
sys.path.insert(0, str(ROOT))

from src.posterior import posterior_rank  # noqa: E402

METHODS = ("dense_hnsw_32", "standard_ivf_32", "lexical_bm25_32", "semantic_graph_only_32", "dual_graph_only_32")
GRAPH_METHODS = {"semantic_graph_only_32", "dual_graph_only_32"}
METRICS = ("ndcg_at_10", "precision_at_5", "recall_at_5", "group_coverage_at_5")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def metrics(ids: list[str], evaluation: dict) -> dict[str, float]:
    gold = set(evaluation["gold_memory_ids"])
    top5 = ids[:5]
    top10 = ids[:10]
    hits10 = [1.0 if value in gold else 0.0 for value in top10]
    ideal = [1.0] * min(len(gold), len(top10)) + [0.0] * max(0, len(top10) - len(gold))
    dcg = lambda values: sum(value / math.log2(index + 2.0) for index, value in enumerate(values))
    groups = [set(group) for group in evaluation["evidence_groups"] if group]
    return {
        "ndcg_at_10": float(dcg(hits10) / max(dcg(ideal), 1e-12)),
        "precision_at_5": float(len(set(top5) & gold) / max(len(top5), 1)),
        "recall_at_5": float(len(set(top5) & gold) / max(len(gold), 1)),
        "group_coverage_at_5": float(np.mean([bool(set(top5) & group) for group in groups])) if groups else 0.0,
    }


def bootstrap(values: list[float], seed: int, samples: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[start : start + count] = array[indices].mean(axis=1)
    return {"mean": float(array.mean()), "ci_95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]}


def objective(metric: dict, weights: dict) -> float:
    return float(sum(float(weights[key]) * float(metric[key]) for key in weights))


def main() -> None:
    config = json.loads((ROOT / "configs" / "posterior.json").read_text(encoding="utf-8"))
    receipt_path = ROOT / "artifacts" / "freeze_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("v29 freeze receipt required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["config_sha256"] != sha(ROOT / "configs" / "posterior.json"):
        raise RuntimeError("v29 config changed after freeze")
    frontier_path = V28 / "runs" / "v28_frontier_oof_20260719_attempt003_913cd3f" / "per_query_frontier.jsonl"
    qwen_path = V28 / "runs" / "v28_qwen_oof_20260719_fullranking" / "per_query_qwen.jsonl"
    if sha(frontier_path) != receipt["v28_frontier_rows_sha256"] or sha(qwen_path) != receipt["v28_qwen_rows_sha256"]:
        raise RuntimeError("v28 frozen input lineage changed")
    run_dir = ROOT / "runs" / "v29_crossfit_oof_20260719"
    run_dir.mkdir(parents=True, exist_ok=False)
    frontier_rows = read_jsonl(frontier_path)
    qwen_rows = read_jsonl(qwen_path)
    evaluations = {row["query_id"]: row for row in read_jsonl(V24_INPUTS / "train_evaluation.jsonl")}
    frontier = {(row["query_id"], row["method"]): row for row in frontier_rows}
    qwen = {(row["query_id"], row["method"]): row for row in qwen_rows}
    query_ids = sorted({row["query_id"] for row in qwen_rows})
    fold_by_id = {row["query_id"]: int(row["fold"]) for row in qwen_rows}
    selected = defaultdict(dict)
    output_rows = []
    for method in METHODS:
        prior_kinds = config["graph_prior_kinds"] if method in GRAPH_METHODS else [config["baseline_prior_kind"]]
        for heldout_fold in range(5):
            tuning_ids = [query_id for query_id in query_ids if fold_by_id[query_id] != heldout_fold]
            candidates = []
            for prior_kind in prior_kinds:
                for alpha in config["alpha_qwen_grid"]:
                    values = []
                    for query_id in tuning_ids:
                        ranked = posterior_rank(
                            qwen[(query_id, method)],
                            frontier[(query_id, method)],
                            alpha_qwen=float(alpha),
                            prior_kind=prior_kind,
                            graph_method=method in GRAPH_METHODS,
                        )
                        values.append(objective(metrics(ranked, evaluations[query_id]), config["objective"]))
                    candidates.append((float(np.mean(values)), float(alpha), str(prior_kind)))
            best_objective, best_alpha, best_prior = max(candidates, key=lambda row: (row[0], row[1], row[2]))
            selected[method][str(heldout_fold)] = {
                "alpha_qwen": best_alpha,
                "prior_kind": best_prior,
                "inner_objective": best_objective,
                "tuning_query_count": len(tuning_ids),
                "heldout_query_count": sum(fold_by_id[query_id] == heldout_fold for query_id in query_ids),
                "heldout_gold_used_for_selection": False,
            }
        for query_id in query_ids:
            fold = fold_by_id[query_id]
            choice = selected[method][str(fold)]
            ranked = posterior_rank(
                qwen[(query_id, method)],
                frontier[(query_id, method)],
                alpha_qwen=choice["alpha_qwen"],
                prior_kind=choice["prior_kind"],
                graph_method=method in GRAPH_METHODS,
            )
            output_rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    "fold": fold,
                    "posterior_ids": ranked,
                    "top10": ranked[:10],
                    "top5": ranked[:5],
                    "metrics": metrics(ranked, evaluations[query_id]),
                    "selection": choice,
                    "candidate_identity_preserved": set(ranked) == set(qwen[(query_id, method)]["frontier_ids"]),
                    "dense_signal_for_graph": False if method in GRAPH_METHODS else None,
                }
            )
    if not all(row["candidate_identity_preserved"] for row in output_rows):
        raise RuntimeError("v29 changed a frozen candidate set")
    write_jsonl(run_dir / "per_query_posterior.jsonl", output_rows)
    by_method = {method: {row["query_id"]: row for row in output_rows if row["method"] == method} for method in METHODS}
    dense_qwen = {row["query_id"]: row for row in qwen_rows if row["method"] == "dense_hnsw_32"}
    dense_posterior = by_method["dense_hnsw_32"]
    summary = {}
    for method in METHODS:
        rows = list(by_method[method].values())
        result = {"absolute": {metric: float(np.mean([row["metrics"][metric] for row in rows])) for metric in METRICS}}
        result["paired_vs_dense_qwen"] = {}
        result["paired_vs_dense_posterior"] = {}
        for metric in METRICS:
            result["paired_vs_dense_qwen"][metric] = bootstrap(
                [row["metrics"][metric] - dense_qwen[row["query_id"]]["metrics"][metric] for row in rows],
                int(config["random_seed"]),
                int(config["bootstrap_samples"]),
            )
            result["paired_vs_dense_posterior"][metric] = bootstrap(
                [row["metrics"][metric] - dense_posterior[row["query_id"]]["metrics"][metric] for row in rows],
                int(config["random_seed"]),
                int(config["bootstrap_samples"]),
            )
        if method in GRAPH_METHODS:
            result["gate_d_vs_dense_qwen"] = all(
                result["paired_vs_dense_qwen"][metric]["ci_95"][0] >= float(config["gate_d_ci_lower"][metric]) for metric in METRICS
            )
            result["gate_d_vs_dense_posterior"] = all(
                result["paired_vs_dense_posterior"][metric]["ci_95"][0] >= float(config["gate_d_ci_lower"][metric]) for metric in METRICS
            )
        summary[method] = result
    aggregate = {
        "schema_version": "method1-v29-crossfit-oof-1",
        "status": "CROSSFIT_OOF_COMPLETE",
        "query_count": len(query_ids),
        "row_count": len(output_rows),
        "config_sha256": sha(ROOT / "configs" / "posterior.json"),
        "freeze_receipt_sha256": sha(receipt_path),
        "v28_frontier_rows_sha256": sha(frontier_path),
        "v28_qwen_rows_sha256": sha(qwen_path),
        "selected_by_fold": selected,
        "methods": summary,
        "gate_d_promoted_vs_dense_qwen": [method for method in GRAPH_METHODS if summary[method]["gate_d_vs_dense_qwen"]],
        "gate_d_promoted_vs_dense_posterior": [method for method in GRAPH_METHODS if summary[method]["gate_d_vs_dense_posterior"]],
        "candidate_identity_violations": 0,
        "dense_signal_for_graph": False,
        "scientific_scope": "strict five-fold cross-fitted development evidence after v28 Gate-D failure",
    }
    aggregate["rows_sha256"] = sha(run_dir / "per_query_posterior.jsonl")
    write_json(run_dir / "aggregate_posterior.json", aggregate)
    print(json.dumps({"status": aggregate["status"], "promoted_vs_dense_qwen": aggregate["gate_d_promoted_vs_dense_qwen"], "promoted_vs_dense_posterior": aggregate["gate_d_promoted_vs_dense_posterior"]}, indent=2))


if __name__ == "__main__":
    main()
