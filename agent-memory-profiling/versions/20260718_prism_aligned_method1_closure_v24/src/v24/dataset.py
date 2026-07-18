from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import EvaluationRecord, MemoryRecord, QueryRecord
from .io import iter_jsonl, read_json


@dataclass
class Dataset:
    memories: list[MemoryRecord]
    embeddings: np.ndarray
    embedding_ids: list[str]
    queries: dict[str, list[QueryRecord]]
    evaluations: dict[str, dict[str, EvaluationRecord]]
    query_vectors: dict[str, dict[str, np.ndarray]]

    def __post_init__(self) -> None:
        if len(self.memories) != int(self.embeddings.shape[0]):
            raise ValueError("memory and embedding row counts differ")
        expected = [memory.memory_id for memory in self.memories]
        if self.embedding_ids != expected:
            raise ValueError("memory embedding IDs do not match canonical memory order")
        self.memory_index = {memory.memory_id: index for index, memory in enumerate(self.memories)}
        self.positions_by_namespace: dict[str, np.ndarray] = {}
        positions: dict[str, list[int]] = defaultdict(list)
        for index, memory in enumerate(self.memories):
            positions[memory.namespace_id].append(index)
        self.positions_by_namespace = {key: np.asarray(value, dtype=np.int64) for key, value in positions.items()}

    @property
    def memory_ids(self) -> list[str]:
        return [memory.memory_id for memory in self.memories]

    def eligible_positions(self, query: QueryRecord) -> np.ndarray:
        values = self.positions_by_namespace.get(query.namespace_id, np.asarray([], dtype=np.int64))
        if values.size == 0:
            return values.copy()
        arrival = np.asarray([self.memories[int(position)].arrival_order for position in values], dtype=np.int64)
        start = np.asarray([self.memories[int(position)].eligible_from_order for position in values], dtype=np.int64)
        keep = (arrival <= int(query.cutoff_order)) & (start <= int(query.cutoff_order))
        return values[keep]

    def query_vector(self, split: str, query_id: str) -> np.ndarray:
        try:
            return np.asarray(self.query_vectors[split][query_id], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(f"missing query vector for {split}/{query_id}") from exc


def load_canonical(root: str | Path, *, split: str | None = None) -> Dataset:
    root = Path(root)
    memories = [MemoryRecord.from_mapping(row) for row in iter_jsonl(root / "memory_records.jsonl")]
    embeddings = np.load(root / "embeddings" / "memory.npy", mmap_mode="r")
    embedding_ids = [str(value) for value in read_json(root / "embeddings" / "memory_ids.json")]
    splits = [split] if split else ["train", "development", "test"]
    queries: dict[str, list[QueryRecord]] = {}
    evaluations: dict[str, dict[str, EvaluationRecord]] = {}
    query_vectors: dict[str, dict[str, np.ndarray]] = {}
    for name in splits:
        queries[name] = [QueryRecord.from_mapping(row) for row in iter_jsonl(root / f"{name}_queries.jsonl")]
        evaluations[name] = {record.query_id: record for record in (EvaluationRecord.from_mapping(row) for row in iter_jsonl(root / f"{name}_evaluation.jsonl"))}
        vectors = np.load(root / "embeddings" / f"queries_{name}.npy", mmap_mode="r")
        index_rows = read_json(root / "embeddings" / f"queries_{name}_index.json")["records"]
        query_vectors[name] = {str(row["query_id"]): np.asarray(vectors[index], dtype=np.float32) for index, row in enumerate(index_rows)}
    return Dataset(memories, embeddings, embedding_ids, queries, evaluations, query_vectors)


def validate_no_cross_split_ids(query_sets: dict[str, Iterable[QueryRecord]]) -> dict[str, object]:
    ids = {split: {query.query_id for query in rows} for split, rows in query_sets.items()}
    overlap = {}
    names = sorted(ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            values = sorted(ids[left] & ids[right])
            if values:
                overlap[f"{left}:{right}"] = values
    return {"split_counts": {key: len(value) for key, value in ids.items()}, "overlap": overlap, "passed": not overlap}
