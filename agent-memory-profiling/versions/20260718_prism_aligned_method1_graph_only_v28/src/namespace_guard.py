from __future__ import annotations

from typing import Sequence


ALLOWED_NAMESPACE_SOURCES = {"authenticated", "dataset_provided", "routed"}


def validate_namespace_source(source: str) -> None:
    if source not in ALLOWED_NAMESPACE_SOURCES:
        raise ValueError(f"unsupported namespace_source: {source}")


def assert_candidate_scope(
    selected_ids: Sequence[str],
    memory_namespace_by_id: dict[str, str],
    memory_arrival_by_id: dict[str, int],
    *,
    namespace_id: str,
    cutoff_order: int,
) -> dict[str, int]:
    return {
        "cross_namespace": sum(memory_namespace_by_id[value] != str(namespace_id) for value in selected_ids),
        "future_memory": sum(memory_arrival_by_id[value] > int(cutoff_order) for value in selected_ids),
        "duplicates": len(selected_ids) - len(set(selected_ids)),
    }
