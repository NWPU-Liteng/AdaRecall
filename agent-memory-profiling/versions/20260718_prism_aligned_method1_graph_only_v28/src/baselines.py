from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import CausalGraphSnapshot


@dataclass(frozen=True)
class BaselineResult:
    memory_ids: list[str]
    local_positions: list[int]
    query_memory_similarity_calls: int
    ann_calls: int


class HNSWSnapshotIndex:
    def __init__(self, snapshot: CausalGraphSnapshot, ef_search: int = 128) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("formal Dense-HNSW baseline requires faiss") from exc
        dimension = snapshot.embeddings.shape[1]
        self.snapshot = snapshot
        self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = int(ef_search)
        self.index.add(np.ascontiguousarray(snapshot.embeddings, dtype=np.float32))

    def search(self, query_embedding: np.ndarray, budget: int) -> BaselineResult:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1).copy()
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        _, indices = self.index.search(query, min(int(budget), len(self.snapshot.positions)))
        rows = [int(value) for value in indices[0] if int(value) >= 0]
        return BaselineResult([self.snapshot.memory_ids[i] for i in rows], rows, 0, 1)


def exact_dense(snapshot: CausalGraphSnapshot, query_embedding: np.ndarray, budget: int) -> BaselineResult:
    query = np.asarray(query_embedding, dtype=np.float32).copy()
    query /= max(float(np.linalg.norm(query)), 1e-12)
    scores = snapshot.embeddings @ query
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), snapshot.memory_ids[i]))[:budget]
    return BaselineResult([snapshot.memory_ids[i] for i in order], order, len(scores), 0)


def hnsw_dense(snapshot: CausalGraphSnapshot, query_embedding: np.ndarray, budget: int, ef_search: int = 128) -> BaselineResult:
    return HNSWSnapshotIndex(snapshot, ef_search=ef_search).search(query_embedding, budget)


def standard_ivf_prototype(
    snapshot: CausalGraphSnapshot,
    query_embedding: np.ndarray,
    *,
    active_cells: int,
    pool_cap: int,
    budget: int,
) -> BaselineResult:
    query = np.asarray(query_embedding, dtype=np.float32).copy()
    query /= max(float(np.linalg.norm(query)), 1e-12)
    centroids = np.vstack([edge.centroid for edge in snapshot.edges[: snapshot.coarse_edge_count]])
    cell_order = np.argsort(-(centroids @ query), kind="stable")[:active_cells]
    hard_assignments = np.asarray(snapshot.incidence[:, : snapshot.coarse_edge_count].argmax(axis=1)).reshape(-1)
    pool = np.flatnonzero(np.isin(hard_assignments, cell_order))
    # A conventional IVF baseline uses local query-memory similarity.
    scores = snapshot.embeddings[pool] @ query
    local_order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), snapshot.memory_ids[int(pool[i])]))
    capped = [int(pool[i]) for i in local_order[:pool_cap]]
    selected = capped[:budget]
    return BaselineResult([snapshot.memory_ids[i] for i in selected], selected, len(pool), 0)


def bm25_lexical(documents: Sequence[str], memory_ids: Sequence[str], query: str, budget: int) -> BaselineResult:
    import re
    from collections import Counter

    tokenize = lambda value: re.findall(r"[a-z0-9]+", value.lower())
    tokenized = [tokenize(value) for value in documents]
    query_terms = tokenize(query)
    frequencies = [Counter(row) for row in tokenized]
    lengths = np.asarray([len(row) for row in tokenized], dtype=np.float64)
    average_length = max(float(lengths.mean()), 1.0)
    document_frequency = Counter(term for row in tokenized for term in set(row))
    scores = np.zeros(len(documents), dtype=np.float64)
    k1, b = 1.5, 0.75
    for term in query_terms:
        df = document_frequency.get(term, 0)
        idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
        tf = np.asarray([row.get(term, 0) for row in frequencies], dtype=np.float64)
        scores += idf * tf * (k1 + 1.0) / np.maximum(tf + k1 * (1.0 - b + b * lengths / average_length), 1e-12)
    order = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), str(memory_ids[i])))[:budget]
    return BaselineResult([str(memory_ids[i]) for i in order], order, 0, 0)


class BM25SnapshotIndex:
    def __init__(self, documents: Sequence[str], memory_ids: Sequence[str]) -> None:
        import re
        from collections import Counter

        self.memory_ids = tuple(str(value) for value in memory_ids)
        self._tokenize = lambda value: re.findall(r"[a-z0-9]+", value.lower())
        tokenized = [self._tokenize(value) for value in documents]
        self.frequencies = [Counter(row) for row in tokenized]
        self.lengths = np.asarray([len(row) for row in tokenized], dtype=np.float64)
        self.average_length = max(float(self.lengths.mean()), 1.0)
        self.document_frequency = Counter(term for row in tokenized for term in set(row))

    def search(self, query: str, budget: int) -> BaselineResult:
        scores = np.zeros(len(self.memory_ids), dtype=np.float64)
        k1, b = 1.5, 0.75
        for term in self._tokenize(query):
            df = self.document_frequency.get(term, 0)
            idf = math.log(1.0 + (len(self.memory_ids) - df + 0.5) / (df + 0.5))
            tf = np.asarray([row.get(term, 0) for row in self.frequencies], dtype=np.float64)
            scores += idf * tf * (k1 + 1.0) / np.maximum(tf + k1 * (1.0 - b + b * self.lengths / self.average_length), 1e-12)
        order = sorted(range(len(self.memory_ids)), key=lambda i: (-float(scores[i]), self.memory_ids[i]))[:budget]
        return BaselineResult([self.memory_ids[i] for i in order], order, 0, 0)
