from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Iterable

from .io import iter_jsonl


_MEMORY_SESSION = re.compile(r"_s(\d+)_x")


@dataclass(frozen=True)
class QueryEvidenceSet:
    query_id: str
    persona_id: str
    query_session_id: int
    group_id: str
    canonical_memory_id: str
    acceptable_memory_ids: tuple[str, ...]
    eligible_memory_count: int


def fold_for(persona_id: str, query_session_id: int, folds: int) -> int:
    return (int(persona_id) * 131 + int(query_session_id)) % folds


def load_query_evidence_sets(path: str, *, folds: int) -> tuple[list[QueryEvidenceSet], dict[str, int]]:
    """Load only causal, train-visible query-to-evidence-set incidences."""
    records: list[QueryEvidenceSet] = []
    query_group_counts: dict[str, int] = {}
    for row in iter_jsonl(path):
        query_id = str(row["query_id"])
        eligible = set(map(str, row["eligible_memory_ids"]))
        groups = row.get("evidence_groups", [])
        query_group_counts[query_id] = len(groups)
        seen_group_ids: set[str] = set()
        for position, group in enumerate(groups):
            group_id = str(group.get("group_id") or f"{query_id}:{position}")
            if group_id in seen_group_ids:
                raise ValueError(f"duplicate evidence group id within query: {group_id}")
            seen_group_ids.add(group_id)
            members = tuple(dict.fromkeys(map(str, group.get("acceptable_memory_ids", []))))
            canonical = str(group.get("canonical_memory_id", ""))
            if not members:
                raise ValueError(f"empty evidence set: {group_id}")
            if canonical not in members:
                raise ValueError(f"canonical member is not acceptable: {group_id}")
            unseen = set(members) - eligible
            if unseen:
                raise ValueError(f"future or unavailable evidence member in {group_id}: {sorted(unseen)}")
            records.append(QueryEvidenceSet(
                query_id=query_id,
                persona_id=str(row["persona_id"]),
                query_session_id=int(row["query_session_id"]),
                group_id=group_id,
                canonical_memory_id=canonical,
                acceptable_memory_ids=members,
                eligible_memory_count=len(eligible),
            ))
    if not records:
        raise ValueError("no train evidence sets found")
    return records, query_group_counts


def incidence_rows(records: Iterable[QueryEvidenceSet], *, folds: int) -> list[dict]:
    rows = []
    for record in records:
        source_sessions = []
        for memory_id in record.acceptable_memory_ids:
            match = _MEMORY_SESSION.search(memory_id)
            if match:
                source_sessions.append(int(match.group(1)))
        rows.append({
            "query_id": record.query_id,
            "fold": fold_for(record.persona_id, record.query_session_id, folds),
            "persona_id": record.persona_id,
            "query_session_id": record.query_session_id,
            "evidence_group_id": record.group_id,
            "canonical_memory_id": record.canonical_memory_id,
            "acceptable_memory_ids": list(record.acceptable_memory_ids),
            "eligible_memory_count": record.eligible_memory_count,
            "source_session_lags": [record.query_session_id - value for value in source_sessions],
        })
    return rows


def summarize(records: Iterable[QueryEvidenceSet], query_group_counts: dict[str, int]) -> dict:
    records = list(records)
    members_per_group = Counter(len(record.acceptable_memory_ids) for record in records)
    groups_per_query = Counter(query_group_counts.values())
    lags = []
    for record in records:
        for memory_id in record.acceptable_memory_ids:
            match = _MEMORY_SESSION.search(memory_id)
            if match:
                lags.append(record.query_session_id - int(match.group(1)))
    by_fold: dict[int, int] = defaultdict(int)
    for record in records:
        by_fold[fold_for(record.persona_id, record.query_session_id, 5)] += 1
    return {
        "queries": len(query_group_counts),
        "evidence_groups": len(records),
        "evidence_member_incidences": sum(members_per_group.elements()),
        "members_per_group": dict(sorted(members_per_group.items())),
        "groups_per_query": dict(sorted(groups_per_query.items())),
        "queries_with_two_or_more_groups": int(sum(count for group_count, count in groups_per_query.items() if group_count >= 2)),
        "fold_group_counts": dict(sorted(by_fold.items())),
        "source_session_lags": {
            "count": len(lags),
            "min": min(lags),
            "p50": sorted(lags)[len(lags) // 2],
            "max": max(lags),
            "same_session": int(sum(value == 0 for value in lags)),
        },
        "causal_availability": "all accepted members were checked against each query's eligible_memory_ids",
    }
