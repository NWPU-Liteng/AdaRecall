from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    faiss = None
    FAISS_IMPORT_ERROR = exc
else:
    FAISS_IMPORT_ERROR = None


def search_flat_ip(vectors: np.ndarray, query: np.ndarray, top_r: int) -> tuple[np.ndarray, np.ndarray]:
    if faiss is None:
        raise RuntimeError(f"faiss unavailable: {FAISS_IMPORT_ERROR}")
    vectors = np.ascontiguousarray(vectors.astype("float32"))
    query = np.ascontiguousarray(query.reshape(1, -1).astype("float32"))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, idx = index.search(query, min(top_r, len(vectors)))
    return scores[0], idx[0]


def brute_force_ip(vectors: np.ndarray, query: np.ndarray, top_r: int) -> np.ndarray:
    scores = vectors.astype("float32") @ query.astype("float32")
    order = np.argsort(-scores, kind="stable")
    return order[: min(top_r, len(order))]


def validate_flat_ip_top_r(
    vectors: np.ndarray,
    query: np.ndarray,
    faiss_indices: list[int],
    top_r: int,
    *,
    epsilon: float = 1e-6,
) -> bool:
    """Validate exact flat-IP retrieval while tolerating float32 boundary ties."""
    r_eff = min(top_r, len(vectors))
    if len(faiss_indices) != r_eff:
        return False
    if len(set(faiss_indices)) != len(faiss_indices):
        return False
    if any(i < 0 or i >= len(vectors) for i in faiss_indices):
        return False
    if r_eff == 0:
        return True

    scores = vectors.astype("float32") @ query.astype("float32")
    brute_order = np.argsort(-scores, kind="stable")
    boundary = float(scores[brute_order[r_eff - 1]])
    return all(float(scores[i]) >= boundary - epsilon for i in faiss_indices)
