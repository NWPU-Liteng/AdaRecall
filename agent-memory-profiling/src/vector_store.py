import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover
    QdrantClient = None
    qmodels = None


class NumpyVectorStore:
    def __init__(self, vector_path: str, metadata_path: str):
        self.vector_path = Path(vector_path)
        self.metadata_path = Path(metadata_path)
        self.mode = "numpy_fallback"
        self.vectors = None
        self.metadata = None

    def build(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        self.vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.vector_path, vectors.astype(np.float32))
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            for row in metadata:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.vectors = vectors.astype(np.float32)
        self.metadata = metadata

    def load(self) -> None:
        self.vectors = np.load(self.vector_path).astype(np.float32)
        self.metadata = []
        with open(self.metadata_path, "r", encoding="utf-8") as handle:
            for line in handle:
                self.metadata.append(json.loads(line))

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[dict, float]]:
        if self.vectors is None or self.metadata is None:
            self.load()
        scores = self.vectors @ query_vector.astype(np.float32)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.metadata[i], float(scores[i])) for i in top_indices]


class AnnNumpyVectorStore(NumpyVectorStore):
    def __init__(
        self,
        vector_path: str,
        metadata_path: str,
        ann_index_path: str,
        cluster_count: int | None = None,
        probe_count: int | None = None,
    ):
        super().__init__(vector_path=vector_path, metadata_path=metadata_path)
        self.ann_index_path = Path(ann_index_path)
        self.cluster_count = cluster_count
        self.probe_count = probe_count
        self.mode = "ann_numpy"
        self.centroids: np.ndarray | None = None
        self.cluster_offsets: np.ndarray | None = None
        self.sorted_indices: np.ndarray | None = None
        self.last_search_stats: Dict[str, int] = {}

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-9, None)
        return (vectors / norms).astype(np.float32, copy=False)

    @staticmethod
    def _default_cluster_count(size: int) -> int:
        if size <= 256:
            return 1
        return max(4, min(128, int(math.sqrt(size))))

    def _resolve_cluster_count(self, size: int) -> int:
        requested = self.cluster_count or self._default_cluster_count(size)
        return max(1, min(requested, size))

    def _resolve_probe_count(self, cluster_count: int) -> int:
        requested = self.probe_count or max(1, int(math.sqrt(cluster_count)))
        return max(1, min(requested, cluster_count))

    @staticmethod
    def _pack_assignments(assignments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(assignments, kind="stable")
        counts = np.bincount(assignments, minlength=int(assignments.max()) + 1)
        offsets = np.zeros(len(counts) + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts, dtype=np.int32)
        return offsets, order.astype(np.int32, copy=False)

    def _build_ann_index(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = self._normalize_rows(vectors.astype(np.float32, copy=False))
        size = normalized.shape[0]
        cluster_count = self._resolve_cluster_count(size)
        if cluster_count == 1:
            centroids = normalized[:1].copy()
            assignments = np.zeros(size, dtype=np.int32)
            offsets, order = self._pack_assignments(assignments)
            return centroids, offsets, order

        seed_positions = np.linspace(0, size - 1, num=cluster_count, dtype=np.int32)
        centroids = normalized[seed_positions].copy()

        for _ in range(4):
            similarities = normalized @ centroids.T
            assignments = similarities.argmax(axis=1).astype(np.int32)
            new_centroids = centroids.copy()
            for cluster_id in range(cluster_count):
                members = normalized[assignments == cluster_id]
                if len(members) == 0:
                    replacement_idx = int((cluster_id * size) / cluster_count)
                    new_centroids[cluster_id] = normalized[replacement_idx]
                    continue
                mean_vector = members.mean(axis=0)
                norm = np.linalg.norm(mean_vector)
                if norm <= 1e-9:
                    new_centroids[cluster_id] = centroids[cluster_id]
                else:
                    new_centroids[cluster_id] = (mean_vector / norm).astype(np.float32, copy=False)
            centroids = new_centroids.astype(np.float32, copy=False)

        similarities = normalized @ centroids.T
        assignments = similarities.argmax(axis=1).astype(np.int32)
        offsets, order = self._pack_assignments(assignments)
        return centroids, offsets, order

    def build(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        super().build(vectors, metadata)
        centroids, cluster_offsets, sorted_indices = self._build_ann_index(self.vectors)
        self.ann_index_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.ann_index_path,
            centroids=centroids.astype(np.float32, copy=False),
            cluster_offsets=cluster_offsets.astype(np.int32, copy=False),
            sorted_indices=sorted_indices.astype(np.int32, copy=False),
        )
        self.centroids = centroids
        self.cluster_offsets = cluster_offsets
        self.sorted_indices = sorted_indices

    def load(self) -> None:
        super().load()
        if not self.ann_index_path.exists():
            raise FileNotFoundError(f"Missing ANN index: {self.ann_index_path}")
        archive = np.load(self.ann_index_path)
        self.centroids = archive["centroids"].astype(np.float32, copy=False)
        self.cluster_offsets = archive["cluster_offsets"].astype(np.int32, copy=False)
        self.sorted_indices = archive["sorted_indices"].astype(np.int32, copy=False)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[dict, float]]:
        if (
            self.vectors is None
            or self.metadata is None
            or self.centroids is None
            or self.cluster_offsets is None
            or self.sorted_indices is None
        ):
            self.load()

        query = query_vector.astype(np.float32, copy=False)
        cluster_scores = self.centroids @ query
        probe_count = self._resolve_probe_count(len(self.centroids))
        probe_clusters = np.argsort(cluster_scores)[-probe_count:][::-1]

        candidate_slices = []
        candidate_total = 0
        for cluster_id in probe_clusters:
            start = int(self.cluster_offsets[cluster_id])
            end = int(self.cluster_offsets[cluster_id + 1])
            if end <= start:
                continue
            candidate_slices.append(self.sorted_indices[start:end])
            candidate_total += end - start

        if not candidate_slices or candidate_total < top_k:
            scores = self.vectors @ query
            top_indices = np.argsort(scores)[-top_k:][::-1]
            self.last_search_stats = {
                "probe_clusters": int(probe_count),
                "candidate_vectors": int(len(self.vectors)),
                "used_fallback_scan": 1,
            }
            return [(self.metadata[i], float(scores[i])) for i in top_indices]

        candidate_indices = np.concatenate(candidate_slices)
        candidate_scores = self.vectors[candidate_indices] @ query
        local_top = np.argsort(candidate_scores)[-top_k:][::-1]
        top_indices = candidate_indices[local_top]
        self.last_search_stats = {
            "probe_clusters": int(probe_count),
            "candidate_vectors": int(candidate_total),
            "used_fallback_scan": 0,
        }
        return [(self.metadata[i], float(candidate_scores[j])) for j, i in zip(local_top, top_indices)]


class QdrantVectorStore:
    def __init__(self, collection_name: str, dim: int, host: str = "127.0.0.1", port: int = 6333):
        if QdrantClient is None:
            raise RuntimeError("qdrant-client not installed")
        self.collection_name = collection_name
        self.dim = dim
        self.client = QdrantClient(host=host, port=port, timeout=5.0)
        self.mode = "qdrant"

    def build(self, vectors: np.ndarray, metadata: List[dict]) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
        )
        points = []
        for idx, (vector, payload) in enumerate(zip(vectors, metadata)):
            points.append(qmodels.PointStruct(id=idx, vector=vector.tolist(), payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[dict, float]]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [(hit.payload, float(hit.score)) for hit in hits]


def get_vector_store(
    prefer_qdrant: bool,
    size: int,
    dim: int,
    base_dir: str,
    retrieval_backend: str = "auto",
):
    vector_path = str(Path(base_dir) / f"index_vectors_{size}.npy")
    metadata_path = str(Path(base_dir) / f"index_metadata_{size}.jsonl")
    ann_index_path = str(Path(base_dir) / f"index_ann_{size}.npz")

    if retrieval_backend == "ann_numpy":
        return AnnNumpyVectorStore(
            vector_path=vector_path,
            metadata_path=metadata_path,
            ann_index_path=ann_index_path,
        )

    if retrieval_backend == "numpy":
        return NumpyVectorStore(vector_path=vector_path, metadata_path=metadata_path)

    if retrieval_backend == "qdrant":
        return QdrantVectorStore(collection_name=f"agent_memory_{size}", dim=dim)

    if prefer_qdrant:
        try:
            store = QdrantVectorStore(collection_name=f"agent_memory_{size}", dim=dim)
            store.client.get_collections()
            return store
        except Exception:
            pass
    return NumpyVectorStore(vector_path=vector_path, metadata_path=metadata_path)
