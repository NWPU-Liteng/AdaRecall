from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


FORBIDDEN_INFERENCE_FIELDS = frozenset({
    "acceptable_member_ids",
    "acceptable_memory_ids",
    "evidence_group_id",
    "evidence_groups",
    "gold_answer",
    "gold_answers",
    "gold_coverage",
    "gold_memory_ids",
    "gold_memory_ids_original",
    "legacy_gold_memory_ids",
    "source_key_points",
})


def session_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(float(text)):012d}")
    except (TypeError, ValueError):
        return (1, text)


@dataclass(frozen=True)
class EligibilityQuery:
    persona_id: str
    query_session_id: Any

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "EligibilityQuery":
        # Deliberately access only fields needed by the label-independent rule.
        return cls(persona_id=str(row["persona_id"]), query_session_id=row["query_session_id"])


@dataclass(frozen=True)
class InferenceEligibilityQuery:
    query_id: str
    persona_id: str
    query_session_id: Any
    eligible_memory_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "InferenceEligibilityQuery":
        forbidden = sorted(FORBIDDEN_INFERENCE_FIELDS.intersection(row.keys()))
        if forbidden:
            raise ValueError(f"inference input contains supervision-only fields: {forbidden}")
        return cls(
            query_id=str(row["query_id"]),
            persona_id=str(row["persona_id"]),
            query_session_id=row["query_session_id"],
            eligible_memory_ids=tuple(map(str, row["eligible_memory_ids"])),
        )


def generate_eligible_memory_ids(
    memories: Iterable[Mapping[str, Any]],
    query: Mapping[str, Any] | EligibilityQuery,
    *,
    include_noise: bool,
    allow_current_session_context: bool,
) -> list[str]:
    """Apply the deployable persona/session eligibility rule without labels."""
    runtime_query = query if isinstance(query, EligibilityQuery) else EligibilityQuery.from_mapping(query)
    query_session = session_sort_key(runtime_query.query_session_id)
    eligible: list[Mapping[str, Any]] = []
    for memory in memories:
        if str(memory["persona_id"]) != runtime_query.persona_id:
            continue
        memory_session = session_sort_key(memory["session_id"])
        valid_time = memory_session <= query_session if allow_current_session_context else memory_session < query_session
        if not valid_time:
            continue
        if not include_noise and str(memory.get("session_type", "")).startswith("noise"):
            continue
        eligible.append(memory)
    eligible.sort(key=lambda row: (int(row.get("chronological_rank", 0)), str(row["memory_id"])))
    return [str(memory["memory_id"]) for memory in eligible]


def assert_train_only_incidence(
    incidence_rows: Iterable[Mapping[str, Any]],
    train_query_ids: set[str],
    *,
    source_split: str,
) -> dict[str, int]:
    if source_split != "train":
        raise ValueError(f"incidence source split must be train, got {source_split!r}")
    observed = {str(row["query_id"]) for row in incidence_rows}
    unexpected = observed - train_query_ids
    missing = train_query_ids - observed
    if unexpected or missing:
        raise ValueError(
            f"train incidence query mismatch: unexpected={sorted(unexpected)[:5]} missing={sorted(missing)[:5]}"
        )
    return {
        "train_query_ids": len(train_query_ids),
        "incidence_query_ids": len(observed),
        "unexpected_query_ids": len(unexpected),
        "missing_query_ids": len(missing),
    }
