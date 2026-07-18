from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    namespace_id: str
    content: str
    arrival_order: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    eligible_from_order: int = 0

    @classmethod
    def from_source(cls, row: Mapping[str, Any]) -> "MemoryRecord":
        namespace = row.get("namespace_id", row.get("persona_id"))
        if namespace is None:
            raise ValueError(f"memory has no namespace: {row.get('memory_id')}")
        arrival = row.get("arrival_order", row.get("session_id", row.get("chronological_rank", 0)))
        content = row.get("content", row.get("text", ""))
        metadata = {
            "source_event_id": row.get("event_id"),
            "source_session_id": row.get("session_id"),
            "source_chronological_rank": row.get("chronological_rank"),
            "memory_kind": row.get("memory_kind"),
            "payload_bytes": int(row.get("payload_bytes", 0) or 0),
        }
        return cls(
            memory_id=str(row["memory_id"]),
            namespace_id=str(namespace),
            content=str(content or ""),
            arrival_order=int(arrival or 0),
            metadata=metadata,
            eligible_from_order=int(row.get("eligible_from_order", 0) or 0),
        )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MemoryRecord":
        return cls(
            memory_id=str(row["memory_id"]),
            namespace_id=str(row["namespace_id"]),
            content=str(row.get("content", "")),
            arrival_order=int(row.get("arrival_order", 0)),
            metadata=dict(row.get("metadata", {})),
            eligible_from_order=int(row.get("eligible_from_order", 0)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    namespace_id: str
    query: str
    cutoff_order: int
    split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_source(cls, row: Mapping[str, Any], split: str) -> "QueryRecord":
        namespace = row.get("namespace_id", row.get("persona_id"))
        cutoff = row.get("cutoff_order", row.get("query_session_id", 0))
        return cls(
            query_id=str(row["query_id"]),
            namespace_id=str(namespace),
            query=str(row.get("query", row.get("query_text", row.get("text", ""))) or ""),
            cutoff_order=int(cutoff or 0),
            split=str(split),
            metadata={
                "query_type": str(row.get("query_type", "unknown")),
                "session_id": row.get("query_session_id"),
                "candidate_split": row.get("candidate_split"),
                "confirmation_split": row.get("confirmation_split"),
            },
        )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "QueryRecord":
        return cls(
            query_id=str(row["query_id"]),
            namespace_id=str(row["namespace_id"]),
            query=str(row.get("query", "")),
            cutoff_order=int(row.get("cutoff_order", 0)),
            split=str(row.get("split", "unknown")),
            metadata=dict(row.get("metadata", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRecord:
    query_id: str
    evidence_groups: tuple[tuple[str, ...], ...]
    gold_memory_ids: tuple[str, ...]
    gold_answers: tuple[str, ...]

    @classmethod
    def from_source(cls, row: Mapping[str, Any]) -> "EvaluationRecord":
        groups = []
        for group in row.get("evidence_groups", []) or []:
            if isinstance(group, Mapping):
                values = group.get("acceptable_memory_ids", group.get("memory_ids", []))
            else:
                values = group
            groups.append(tuple(str(value) for value in (values or [])))
        answers = row.get("gold_answers", row.get("gold_answer", [])) or []
        if isinstance(answers, str):
            answers = [answers]
        return cls(
            query_id=str(row["query_id"]),
            evidence_groups=tuple(groups),
            gold_memory_ids=tuple(str(value) for value in row.get("gold_memory_ids", []) or []),
            gold_answers=tuple(str(value) for value in answers),
        )

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EvaluationRecord":
        return cls(
            query_id=str(row["query_id"]),
            evidence_groups=tuple(tuple(str(x) for x in group) for group in row.get("evidence_groups", [])),
            gold_memory_ids=tuple(str(x) for x in row.get("gold_memory_ids", [])),
            gold_answers=tuple(str(x) for x in row.get("gold_answers", [])),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "evidence_groups": [list(group) for group in self.evidence_groups],
            "gold_memory_ids": list(self.gold_memory_ids),
            "gold_answers": list(self.gold_answers),
        }


@dataclass(frozen=True)
class HistoryEvent:
    event_id: str
    namespace_id: str
    query_id: str
    query_embedding_id: str
    selected_memory_ids: tuple[str, ...]
    event_kind: str
    task_success: float | None
    arrival_order: int

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_memory_ids"] = list(self.selected_memory_ids)
        return value


@dataclass
class QueryRunResult:
    run_id: str
    query_id: str
    namespace_id: str
    method: str
    seed_ids: list[str]
    expanded_pool_ids: list[str]
    frontier_ids: list[str]
    reranked_ids: list[str]
    seed_ms: float
    edge_activation_ms: float
    diffusion_ms: float
    frontier_finalize_ms: float
    rerank_ms: float
    semantic_selection_ms: float
    peak_cpu_bytes: int
    peak_gpu_bytes: int
    reranker_pairs: int
    reranker_input_tokens: int
    metrics: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)
