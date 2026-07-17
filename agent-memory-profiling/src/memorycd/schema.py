from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DOMAINS = [
    "Beauty_and_Personal_Care",
    "Books",
    "Electronics",
    "Home_and_Kitchen",
]

TASKS = [
    "rating_prediction",
    "item_ranking",
    "review_generation",
    "review_summarization",
]


@dataclass(frozen=True)
class MemoryCDMemory:
    memory_id: str
    tenant_id: str
    domain: str
    timestamp: int
    parent_asin: str
    rating: float | None
    review_title: str
    review_text: str
    memory_text: str
    text_bytes: int

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "tenant_id": self.tenant_id,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "parent_asin": self.parent_asin,
            "memory_text": self.memory_text,
            "text": self.memory_text,
            "text_bytes": self.text_bytes,
        }


@dataclass(frozen=True)
class MemoryCDTaskEvent:
    event_id: str
    tenant_id: str
    protocol: str
    task: str
    setting: str
    target_domain: str
    source_domains: tuple[str, ...]
    target_timestamp: int
    query_text: str
    candidate_item_metadata: tuple[str, ...]
    runtime_memory_ids: tuple[str, ...]
    runtime_memory_scope_size: int

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "protocol": self.protocol,
            "task": self.task,
            "setting": self.setting,
            "target_domain": self.target_domain,
            "source_domains": list(self.source_domains),
            "target_timestamp": self.target_timestamp,
            "query_text": self.query_text,
            "candidate_item_metadata": list(self.candidate_item_metadata),
            "runtime_memory_ids": list(self.runtime_memory_ids),
            "runtime_memory_scope_size": self.runtime_memory_scope_size,
        }

