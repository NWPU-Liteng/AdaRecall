from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from v24.contracts import EvaluationRecord, HistoryEvent, MemoryRecord, QueryRecord
from v24.dataset import Dataset
from v24.history_hypergraph import build_history_edges
from v24.pipeline import Method1Runner
from v24.semantic_hypergraph import fit_spherical_soft


def test_three_way_pipeline_preserves_causal_visibility(tmp_path) -> None:
    rng = np.random.default_rng(9)
    embeddings = rng.normal(size=(6, 8)).astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    memories = [MemoryRecord(f"m{i}", "a", f"memory {i}", 1 if i < 3 else 2) for i in range(6)]
    query = QueryRecord("q", "a", "memory question", 1, "development")
    evaluation = EvaluationRecord("q", (("m0",),), ("m0",), ("answer",))
    data = Dataset(
        memories, embeddings, [memory.memory_id for memory in memories],
        {"development": [query]}, {"development": {"q": evaluation}},
        {"development": {"q": embeddings[0]}},
    )
    semantic = fit_spherical_soft(embeddings, n_prototypes=3, top_memberships=2, temperature=0.2, seed=9)
    artifact = tmp_path / "index"
    artifact.mkdir()
    np.save(artifact / "semantic_prototypes.npy", semantic.prototypes)
    sparse.save_npz(artifact / "semantic_incidence.npz", semantic.incidence)
    (artifact / "memory_ids.json").write_text(json.dumps(data.memory_ids), encoding="utf-8")
    history = build_history_edges(
        [HistoryEvent("e", "a", "q", "q", ("m0", "m1"), "co_candidate", None, 1)],
        set(data.memory_ids), {"q": embeddings[0]}, max_event_size=4,
    )
    (artifact / "history_edges.json").write_text(json.dumps(history.edges), encoding="utf-8")
    np.save(artifact / "history_edge_vectors.npy", history.vectors)
    (artifact / "history_inverted.json").write_text(json.dumps(history.inverted), encoding="utf-8")
    config = {
        "budget": {"expanded_pool_cap": 8},
        "semantic_hypergraph": {"top_active_edges": 4, "activation_temperature": 0.1},
        "history_hypergraph": {"top_active_edges": 4, "activation_temperature": 0.1},
        "diffusion": {"alpha_restart": 0.8, "steps": 2, "beta_dense": 0.7, "history_weight": 0.25},
        "reranker": {"backend": "dense_reference"},
    }
    runner = Method1Runner(data, artifact, config)
    for method in ("full_n", "dense_c", "ours_c"):
        result = runner.run_query(query, method=method, run_id="test", full_n=3, seed_s=1, frontier_c=2, final_k=1)
        assert len(result.frontier_ids) == (3 if method == "full_n" else 2)
        assert len(set(result.frontier_ids)) == (3 if method == "full_n" else 2)
        assert result.metrics["cross_namespace_violation"] is False
        assert result.metrics["future_memory_violation"] is False
        assert result.reranker_pairs == (3 if method == "full_n" else 2)
