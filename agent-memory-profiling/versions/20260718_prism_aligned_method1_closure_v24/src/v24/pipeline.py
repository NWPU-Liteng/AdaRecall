from __future__ import annotations

import json
import resource
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .ann import NamespaceANN
from .contracts import QueryRecord, QueryRunResult
from .dataset import Dataset
from .diffusion import diffuse
from .frontier_selector import fuse_relevance, materialize_pool, select_frontier
from .history_hypergraph import HistoryHypergraph
from .metrics import quality_metrics
from .reranker import DenseReferenceReranker, load_reranker
from .semantic_hypergraph import SemanticHypergraph


def _peak_cpu_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if value < 10**9 else value


def _peak_gpu_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
    except Exception:
        pass
    return 0


class Method1Runner:
    def __init__(self, dataset: Dataset, artifact_dir: str | Path, config: Mapping[str, Any], *, reranker_backend: str | None = None) -> None:
        self.dataset = dataset
        self.artifact_dir = Path(artifact_dir)
        self.config = dict(config)
        self.memory_ids = dataset.memory_ids
        self.id_by_position = self.memory_ids
        self.position_by_id = dataset.memory_index
        self.embeddings = np.array(dataset.embeddings, dtype=np.float32, copy=True)
        self.embeddings /= np.maximum(np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-12)
        self.ann = NamespaceANN(self.embeddings, [memory.namespace_id for memory in dataset.memories])
        self.semantic = SemanticHypergraph(
            np.load(self.artifact_dir / "semantic_prototypes.npy", mmap_mode="r"),
            __import__("scipy.sparse", fromlist=["load_npz"]).load_npz(self.artifact_dir / "semantic_incidence.npz"),
            self.memory_ids,
        )
        self.history = HistoryHypergraph(
            json.loads((self.artifact_dir / "history_edges.json").read_text(encoding="utf-8")),
            np.load(self.artifact_dir / "history_edge_vectors.npy", mmap_mode="r"),
            json.loads((self.artifact_dir / "history_inverted.json").read_text(encoding="utf-8")),
        )
        reranker_config = dict(self.config.get("reranker", {}))
        if reranker_backend is not None:
            reranker_config["backend"] = reranker_backend
        self.reranker = load_reranker(reranker_config)

    def _dense_scores(self, query_vector: np.ndarray, positions: np.ndarray) -> np.ndarray:
        query = np.array(query_vector, dtype=np.float32, copy=True).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        return np.asarray(self.embeddings[np.asarray(positions, dtype=np.int64)] @ query, dtype=np.float32)

    def _rerank(self, query: QueryRecord, positions: np.ndarray, dense_scores: np.ndarray, final_k: int) -> tuple[list[str], float, int, list[dict[str, Any]]]:
        documents = [self.dataset.memories[int(position)].content for position in positions]
        started = perf_counter()
        if isinstance(self.reranker, DenseReferenceReranker):
            output_scores = [float(value) for value in dense_scores]
            input_tokens = 0
        else:
            output = self.reranker.score_pairs(query.query, documents)
            output_scores = output.scores
            input_tokens = int(output.input_tokens)
        rows = [
            {
                "memory_id": self.memory_ids[int(position)],
                "rerank_score": float(score),
                "dense_score": float(dense),
            }
            for position, score, dense in zip(positions, output_scores, dense_scores)
        ]
        rows.sort(key=lambda row: (-float(row["rerank_score"]), -float(row["dense_score"]), str(row["memory_id"])))
        return [row["memory_id"] for row in rows[: int(final_k)]], float((perf_counter() - started) * 1000.0), input_tokens, rows

    def run_query(
        self,
        query: QueryRecord,
        *,
        method: str,
        run_id: str,
        full_n: int,
        seed_s: int,
        frontier_c: int,
        final_k: int,
    ) -> QueryRunResult:
        started = perf_counter()
        query_vector = self.dataset.query_vector(query.split, query.query_id)
        eligible = self.dataset.eligible_positions(query)
        if method == "full_n":
            seed_k = int(full_n)
        elif method == "dense_c":
            seed_k = int(frontier_c)
        elif method == "ours_c":
            seed_k = int(seed_s)
        else:
            raise ValueError(f"unknown method: {method}")
        ann_result = self.ann.search(query.namespace_id, query_vector, k=seed_k, eligible=eligible)
        seed_positions = ann_result.positions
        seed_ids = [self.memory_ids[int(position)] for position in seed_positions]
        seed_scores = ann_result.scores
        edge_ms = 0.0
        diffusion_ms = 0.0
        finalize_ms = 0.0
        active_semantic: list[dict[str, Any]] = []
        active_history: list[dict[str, Any]] = []
        expanded_positions = seed_positions.copy()
        frontier_positions = seed_positions.copy()
        component_values: dict[str, np.ndarray] = {}
        trace: dict[str, Any] = {"ann_backend": ann_result.backend, "eligible_count": int(len(eligible))}
        dense_scores = self._dense_scores(query_vector, frontier_positions)

        if method == "ours_c":
            activation_started = perf_counter()
            active_semantic, semantic_ms = self.semantic.active_edges(
                query_vector, seed_positions, seed_scores,
                top_edges=int(self.config.get("semantic_hypergraph", {}).get("top_active_edges", 16)),
                tau=float(self.config.get("semantic_hypergraph", {}).get("activation_temperature", 0.10)),
            )
            active_history, history_ms = self.history.active_edges(
                query_vector, seed_ids, seed_scores,
                top_edges=int(self.config.get("history_hypergraph", {}).get("top_active_edges", 32)),
                tau=float(self.config.get("history_hypergraph", {}).get("activation_temperature", 0.10)),
            )
            edge_ms = float((perf_counter() - activation_started) * 1000.0)
            candidate_cap = int(self.config.get("budget", {}).get("expanded_pool_cap", frontier_c * 8))
            dense_cache = {int(position): float(value) for position, value in zip(eligible, self._dense_scores(query_vector, eligible))}
            history_edges_for_pool = []
            for edge in active_history:
                positions = [self.position_by_id[memory_id] for memory_id in edge["members"] if memory_id in self.position_by_id]
                if positions:
                    history_edges_for_pool.append({**edge, "members": np.asarray(positions, dtype=np.int64)})
            pool_positions = materialize_pool(
                eligible, seed_positions,
                active_semantic + history_edges_for_pool,
                cap=candidate_cap, dense_scores_by_position=dense_cache,
            )
            if len(pool_positions) == 0:
                pool_positions = seed_positions.copy()
            expanded_positions = pool_positions
            local_index = {int(position): index for index, position in enumerate(pool_positions)}
            local_ids = [self.memory_ids[int(position)] for position in pool_positions]
            local_dense = self._dense_scores(query_vector, pool_positions)
            x0 = np.zeros(len(pool_positions), dtype=np.float32)
            for position, score in zip(seed_positions, seed_scores):
                if int(position) in local_index:
                    x0[local_index[int(position)]] = max(float(score), 0.0)
            if not np.any(x0):
                x0[: min(len(x0), len(seed_positions))] = 1.0
            semantic_members = []
            semantic_weights = []
            for edge in active_semantic:
                members = np.asarray([local_index[int(position)] for position in edge["members"] if int(position) in local_index], dtype=np.int64)
                if len(members):
                    semantic_members.append(members)
                    semantic_weights.append(float(edge["activation"]))
            history_members = []
            history_weights = []
            for edge in history_edges_for_pool:
                members = np.asarray([local_index[int(position)] for position in edge["members"] if int(position) in local_index], dtype=np.int64)
                if len(members):
                    history_members.append(members)
                    history_weights.append(float(edge["activation"]))
            diffusion_started = perf_counter()
            diffusion_cfg = self.config.get("diffusion", {})
            semantic_support = diffuse(
                len(pool_positions), semantic_members, semantic_weights, x0,
                alpha=float(diffusion_cfg.get("alpha_restart", 0.8)), steps=int(diffusion_cfg.get("steps", 2)),
            ) if semantic_members else np.zeros(len(pool_positions), dtype=np.float32)
            history_support = diffuse(
                len(pool_positions), history_members, history_weights, x0,
                alpha=float(diffusion_cfg.get("alpha_restart", 0.8)), steps=int(diffusion_cfg.get("steps", 2)),
            ) if history_members else np.zeros(len(pool_positions), dtype=np.float32)
            diffusion_ms = float((perf_counter() - diffusion_started) * 1000.0)
            beta = float(diffusion_cfg.get("beta_dense", 0.7))
            history_weight = float(diffusion_cfg.get("history_weight", 0.25))
            fused, component_values = fuse_relevance(
                local_ids, local_dense, semantic_support, history_support,
                beta_dense=beta, history_weight=history_weight,
            )
            finalize_started = perf_counter()
            selected_local, selected_rows = select_frontier(
                local_ids, fused, cap=int(frontier_c),
                components={"dense_score": local_dense, "semantic_support": semantic_support, "history_support": history_support},
            )
            frontier_positions = pool_positions[np.asarray(selected_local, dtype=np.int64)]
            dense_scores = local_dense[np.asarray(selected_local, dtype=np.int64)]
            finalize_ms = float((perf_counter() - finalize_started) * 1000.0)
            trace.update({
                "active_semantic_edges": [
                    {key: value for key, value in edge.items() if key != "members"}
                    for edge in active_semantic
                ],
                "active_history_edges": [
                    {key: value for key, value in edge.items() if key != "members"}
                    for edge in active_history
                ],
                "frontier_score_rows": selected_rows,
                "semantic_activation_ms": float(semantic_ms),
                "history_activation_ms": float(history_ms),
            })
        else:
            frontier_positions = seed_positions[: (int(full_n) if method == "full_n" else int(frontier_c))]
            expanded_positions = frontier_positions.copy()
            dense_scores = self._dense_scores(query_vector, frontier_positions)

        final_ids, rerank_ms, input_tokens, reranked_rows = self._rerank(query, frontier_positions, dense_scores, int(final_k))
        evaluation = self.dataset.evaluations.get(query.split, {}).get(query.query_id)
        frontier_ids = [self.memory_ids[int(position)] for position in frontier_positions]
        metrics = {
            "eligible_memory_count": int(len(eligible)),
            "expanded_pool_count": int(len(expanded_positions)),
            "frontier_count": int(len(frontier_positions)),
            "reranker_candidate_count": int(len(frontier_positions)),
            "seed_count": int(len(seed_positions)),
            "candidate_recall_frontier": quality_metrics(frontier_ids, evaluation)["candidate_recall"],
            "evidence_group_coverage_frontier": quality_metrics(frontier_ids, evaluation)["evidence_group_coverage"],
            "candidate_recall_final": quality_metrics(final_ids, evaluation, k=final_k)["candidate_recall"],
            "evidence_group_coverage_final": quality_metrics(final_ids, evaluation, k=final_k)["evidence_group_coverage"],
            "precision_at_k": quality_metrics(final_ids, evaluation, k=final_k)["precision_at_k"],
            "recall_at_k": quality_metrics(final_ids, evaluation, k=final_k)["recall_at_k"],
            "ndcg_at_10": quality_metrics(final_ids, evaluation, k=10)["ndcg_at_10"],
            "mrr_at_10": quality_metrics(final_ids, evaluation, k=10)["mrr_at_10"],
            "cross_namespace_violation": bool(any(self.dataset.memories[int(position)].namespace_id != query.namespace_id for position in expanded_positions)),
            "future_memory_violation": bool(any(self.dataset.memories[int(position)].arrival_order > query.cutoff_order for position in expanded_positions)),
            "duplicate_frontier_ids": bool(len(frontier_ids) != len(set(frontier_ids))),
            "candidate_pool_source": "ann_only" if method != "ours_c" else "query_conditioned_dual_view",
            "ann_backend": ann_result.backend,
        }
        total_ms = float((perf_counter() - started) * 1000.0)
        semantic_selection_ms = float(ann_result.latency_ms + edge_ms + diffusion_ms + finalize_ms + rerank_ms)
        return QueryRunResult(
            run_id=run_id, query_id=query.query_id, namespace_id=query.namespace_id, method=method,
            seed_ids=seed_ids, expanded_pool_ids=[self.memory_ids[int(position)] for position in expanded_positions],
            frontier_ids=frontier_ids, reranked_ids=final_ids,
            seed_ms=float(ann_result.latency_ms), edge_activation_ms=edge_ms, diffusion_ms=diffusion_ms,
            frontier_finalize_ms=finalize_ms, rerank_ms=rerank_ms, semantic_selection_ms=semantic_selection_ms,
            peak_cpu_bytes=_peak_cpu_bytes(), peak_gpu_bytes=_peak_gpu_bytes(),
            reranker_pairs=int(len(frontier_positions)), reranker_input_tokens=int(input_tokens),
            metrics={**metrics, "total_wall_ms": total_ms},
            trace={**trace, "reranked_candidates": reranked_rows, "total_wall_ms": total_ms},
        )


def run_query_matrix(
    runner: Method1Runner,
    queries: list[QueryRecord],
    *,
    run_id: str,
    full_n: int,
    seed_s: int,
    frontier_c: int,
    final_k: int,
    methods: tuple[str, ...] = ("full_n", "dense_c", "ours_c"),
) -> list[QueryRunResult]:
    results = []
    for query in queries:
        for method in methods:
            results.append(runner.run_query(
                query, method=method, run_id=run_id, full_n=full_n, seed_s=seed_s,
                frontier_c=frontier_c, final_k=final_k,
            ))
    return results
