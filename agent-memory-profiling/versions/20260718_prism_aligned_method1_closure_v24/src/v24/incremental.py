from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class IncrementalSemanticOverlay:
    memory_ids: list[str]
    embeddings: np.ndarray
    incidence: sparse.csr_matrix
    prototypes: np.ndarray

    def insert_memory(self, memory_id: str, embedding: np.ndarray) -> int:
        if memory_id in self.memory_ids:
            raise ValueError(f"duplicate memory ID: {memory_id}")
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
        logits = vector @ self.prototypes.T
        top = min(4, logits.shape[1])
        ids = np.argpartition(-logits[0], top - 1)[:top]
        selected = logits[0, ids] - float(logits[0, ids].max())
        weights = np.exp(selected)
        weights /= max(float(weights.sum()), 1e-12)
        row = sparse.csr_matrix((weights.astype(np.float32), (np.zeros(top, dtype=np.int64), ids)), shape=(1, self.incidence.shape[1]))
        self.memory_ids.append(str(memory_id))
        self.embeddings = np.vstack([np.asarray(self.embeddings), vector])
        self.incidence = sparse.vstack([self.incidence, row], format="csr")
        return len(self.memory_ids) - 1
