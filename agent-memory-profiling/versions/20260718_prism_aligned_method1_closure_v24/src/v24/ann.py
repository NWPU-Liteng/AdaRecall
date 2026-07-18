from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class ANNResult:
    positions: np.ndarray
    scores: np.ndarray
    latency_ms: float
    backend: str


class NamespaceANN:
    """Namespace-isolated dense retriever with a deterministic exact fallback."""

    def __init__(self, embeddings: np.ndarray, namespaces: list[str] | np.ndarray) -> None:
        self.values = np.array(embeddings, dtype=np.float32, copy=True)
        self.values /= np.maximum(np.linalg.norm(self.values, axis=1, keepdims=True), 1e-12)
        self.namespaces = np.asarray([str(value) for value in namespaces], dtype=object)
        self.positions = {
            namespace: np.flatnonzero(self.namespaces == namespace).astype(np.int64)
            for namespace in sorted(set(self.namespaces.tolist()))
        }
        self.backend = "numpy_exact"

    def search(
        self,
        namespace: str,
        query_vector: np.ndarray,
        *,
        k: int,
        eligible: np.ndarray,
    ) -> ANNResult:
        started = perf_counter()
        visible = np.asarray(eligible, dtype=np.int64)
        if visible.size == 0 or k <= 0:
            return ANNResult(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), 0.0, self.backend)
        query = np.array(query_vector, dtype=np.float32, copy=True).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = np.asarray(self.values[visible] @ query, dtype=np.float32)
        take = min(int(k), int(visible.size))
        if take < visible.size:
            candidate_indices = np.argpartition(-scores, take - 1)[:take]
        else:
            candidate_indices = np.arange(visible.size, dtype=np.int64)
        ordered = sorted(
            (int(index) for index in candidate_indices),
            key=lambda index: (-float(scores[index]), str(visible[index])),
        )
        selected = np.asarray([visible[index] for index in ordered], dtype=np.int64)
        selected_scores = np.asarray([scores[index] for index in ordered], dtype=np.float32)
        return ANNResult(selected, selected_scores, float((perf_counter() - started) * 1000.0), self.backend)

    def summary(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "metric": "inner_product_on_l2_normalized_vectors",
            "dimension": int(self.values.shape[1]),
            "namespace_count": int(len(self.positions)),
            "entries": int(self.values.shape[0]),
        }
