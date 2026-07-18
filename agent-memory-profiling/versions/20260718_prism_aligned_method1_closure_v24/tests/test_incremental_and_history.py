from __future__ import annotations

import numpy as np
from scipy import sparse

from v24.contracts import HistoryEvent
from v24.history_hypergraph import build_history_edges
from v24.incremental import IncrementalSemanticOverlay


def test_incremental_insert_only_appends_one_embedding_and_incidence_row() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    incidence = sparse.csr_matrix(np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    overlay = IncrementalSemanticOverlay(["m0", "m1"], np.eye(2, 3, dtype=np.float32), incidence, prototypes)
    position = overlay.insert_memory("m2", np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    assert position == 2
    assert overlay.embeddings.shape == (3, 3)
    assert overlay.incidence.shape == (3, 3)
    assert overlay.incidence.getrow(2).indices.tolist() == [0, 1, 2]
    assert np.isclose(overlay.incidence.getrow(2).toarray().sum(), 1.0)


def test_history_builder_preserves_co_candidate_semantics() -> None:
    events = [HistoryEvent("e1", "a", "q1", "q1", ("m1", "m2", "missing"), "co_candidate", None, 2)]
    result = build_history_edges(events, {"m1", "m2"}, {"q1": np.asarray([1.0, 0.0], dtype=np.float32)}, max_event_size=4)
    assert len(result.edges) == 1
    assert result.edges[0]["members"] == ["m1", "m2"]
    assert result.diagnostics["source_semantics"] == "co_candidate_only"
    assert result.diagnostics["task_success_available"] is False
