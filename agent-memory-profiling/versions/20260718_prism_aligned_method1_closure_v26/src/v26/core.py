from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TAIL_OPTIONS = (0, 2, 4, 8, 12)
MEDICAL_TERMS = frozenset(
    "allergy asthma blood cancer cholesterol copd diabetes dose drug glucose heart hypertension "
    "insulin kidney laboratory medication medicine pain pressure symptom test treatment".split()
)
DATE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:\s?(?:mg|ml|mmhg|kg|%))?\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_-]+", re.IGNORECASE)


def stable_fold(query_id: str, group_id: str, groups: Mapping[str, int] | None = None) -> int:
    """Return a deterministic fold, optionally using a precomputed group map."""
    if groups is not None and group_id in groups:
        return int(groups[group_id])
    digest = hashlib.sha256(("v26-oof:" + group_id + ":" + query_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 5


def rank_percentile(values: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), str(ids[index])))
    output = np.empty(len(values), dtype=np.float32)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        output[index] = float(rank) / float(denominator)
    return output


def query_flags(text: str) -> tuple[float, float, float]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return (
        float(bool(DATE_PATTERN.search(text))),
        float(bool(NUMBER_PATTERN.search(text))),
        float(sum(token in MEDICAL_TERMS for token in tokens)),
    )


def adaptive_features(
    query: str,
    dense_scores: np.ndarray,
    lexical_scores: np.ndarray,
    ids: Sequence[str],
    *,
    frontier_c: int = 32,
) -> np.ndarray:
    dense_scores = np.asarray(dense_scores, dtype=np.float64)
    lexical_scores = np.asarray(lexical_scores, dtype=np.float64)
    dense_order = sorted(range(len(ids)), key=lambda i: (-float(dense_scores[i]), str(ids[i])))
    lexical_order = sorted(range(len(ids)), key=lambda i: (-float(lexical_scores[i]), str(ids[i])))
    dense_top = set(dense_order[:frontier_c])
    lexical_top = set(lexical_order[:frontier_c])
    novel = [index for index in lexical_order if index not in dense_top]
    top = float(lexical_scores[lexical_order[0]]) if lexical_order else 0.0
    second = float(lexical_scores[lexical_order[1]]) if len(lexical_order) > 1 else 0.0
    novel_top = float(lexical_scores[novel[0]]) if novel else 0.0
    top8_mass = float(sum(max(float(lexical_scores[index]), 0.0) for index in lexical_order[:8]))
    date, numeric, medical_count = query_flags(query)
    agreement = float(len(dense_top & lexical_top) / max(frontier_c, 1))
    top16_agreement = float(len(set(dense_order[:16]) & set(lexical_order[:16])) / 16.0)
    return np.asarray(
        [
            top,
            top - second,
            novel_top,
            top8_mass,
            agreement,
            top16_agreement,
            date,
            numeric,
            min(medical_count, 5.0) / 5.0,
            float(len(ids) > frontier_c),
        ],
        dtype=np.float64,
    )


def rule_tail(features: np.ndarray, config: Mapping[str, Any]) -> int:
    novel, agreement = float(features[2]), float(features[4])
    exact_signal = bool(features[6] or features[7] or features[8] >= 0.2)
    if novel >= float(config["high_confidence"]) and agreement <= float(config["low_agreement"]) and exact_signal:
        return 12
    if novel >= float(config["medium_confidence"]) and agreement <= float(config["medium_agreement"]) and exact_signal:
        return 8
    if novel >= float(config["medium_confidence"]):
        return 4
    if novel >= float(config["low_confidence"]) and exact_signal:
        return 2
    return 0


def select_tail(
    ids: Sequence[str],
    dense_scores: np.ndarray,
    auxiliary_scores: np.ndarray,
    tail: int,
    *,
    frontier_c: int = 32,
) -> list[int]:
    """Protect the dense prefix and replace only its tail with novel evidence."""
    cap = min(int(frontier_c), len(ids))
    tail = min(max(int(tail), 0), cap)
    dense_order = sorted(range(len(ids)), key=lambda i: (-float(dense_scores[i]), str(ids[i])))
    dense_frontier = dense_order[:cap]
    dense_set = set(dense_frontier)
    auxiliary_order = sorted(range(len(ids)), key=lambda i: (-float(auxiliary_scores[i]), str(ids[i])))
    additions = [index for index in auxiliary_order if index not in dense_set][:tail]
    selected = dense_frontier[: cap - len(additions)] + additions
    if len(selected) < cap:
        selected.extend(index for index in dense_frontier if index not in selected)
    return selected[:cap]


def oracle_tail_label(
    option_metrics: Mapping[int, Mapping[str, float]], options: Sequence[int] = TAIL_OPTIONS
) -> int:
    """Training-fold-only target. Prefer recall, coverage, then fewer replacements."""
    return int(
        max(
            options,
            key=lambda tail: (
                float(option_metrics[int(tail)]["candidate_recall"]),
                float(option_metrics[int(tail)]["evidence_group_coverage"]),
                -int(tail),
            ),
        )
    )


def calibrated_edge_scores(
    query_vector: np.ndarray,
    seed_positions: Sequence[int],
    eligible_positions: Sequence[int],
    prototypes: np.ndarray,
    incidence: Any,
    *,
    minimum_seed_support: int = 2,
    edge_size_penalty_power: float = 1.0,
) -> np.ndarray:
    """Degree-calibrated semantic support with centered query activation."""
    eligible = np.asarray(eligible_positions, dtype=np.int64)
    output = np.zeros(len(eligible), dtype=np.float32)
    if not len(eligible):
        return output
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1).copy()
    query /= max(float(np.linalg.norm(query)), 1e-12)
    proto = np.asarray(prototypes, dtype=np.float32)
    proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    raw = np.asarray(proto @ query, dtype=np.float64)
    median = float(np.median(raw))
    mad = float(np.median(np.abs(raw - median)))
    centered = np.maximum((raw - median) / max(1.4826 * mad, 1e-4), 0.0)
    degrees = np.asarray(incidence.sum(axis=0)).reshape(-1).astype(np.float64)
    seed_support = np.asarray((incidence[np.asarray(seed_positions, dtype=np.int64)] > 0).sum(axis=0)).reshape(-1)
    active = seed_support >= int(minimum_seed_support)
    weights = centered * active / np.maximum(np.log1p(degrees), 1.0) ** float(edge_size_penalty_power)
    output[:] = np.asarray(incidence[eligible] @ weights, dtype=np.float32).reshape(-1)
    return output


@dataclass(frozen=True)
class HistoryEdge:
    query_id: str
    namespace_id: str
    arrival_order: int
    members: tuple[str, ...]
    vector: np.ndarray


def calibrated_history_scores(
    query_vector: np.ndarray,
    seed_ids: Sequence[str],
    eligible_ids: Sequence[str],
    edges: Iterable[HistoryEdge],
    *,
    namespace_id: str,
    cutoff_order: int,
    excluded_query_ids: set[str],
    minimum_seed_support: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return OOF-safe history support and an auditable exclusion count."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1).copy()
    query /= max(float(np.linalg.norm(query)), 1e-12)
    seed_set = set(str(value) for value in seed_ids)
    id_to_local = {str(value): index for index, value in enumerate(eligible_ids)}
    output = np.zeros(len(eligible_ids), dtype=np.float32)
    usable: list[tuple[HistoryEdge, float, int]] = []
    excluded = 0
    future = 0
    for edge in edges:
        if edge.query_id in excluded_query_ids:
            excluded += 1
            continue
        if edge.namespace_id != str(namespace_id):
            continue
        if int(edge.arrival_order) > int(cutoff_order):
            future += 1
            continue
        overlap = len(seed_set & set(edge.members))
        if overlap < int(minimum_seed_support):
            continue
        vector = np.asarray(edge.vector, dtype=np.float32).reshape(-1)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        usable.append((edge, float(vector @ query), overlap))
    if usable:
        values = np.asarray([row[1] for row in usable], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        for edge, similarity, overlap in usable:
            activation = max((similarity - median) / max(1.4826 * mad, 1e-4), 0.0)
            activation *= float(overlap) / max(float(len(edge.members)), 1.0)
            for memory_id in edge.members:
                local = id_to_local.get(memory_id)
                if local is not None:
                    output[local] += float(activation)
    return output, {"excluded_heldout_events": excluded, "excluded_future_events": future, "usable_edges": len(usable)}


def safety_counts(
    selected_ids: Sequence[str],
    memory_by_id: Mapping[str, Any],
    *,
    namespace_id: str,
    cutoff_order: int,
) -> dict[str, int]:
    return {
        "cross_namespace": int(sum(memory_by_id[value].namespace_id != str(namespace_id) for value in selected_ids)),
        "future_memory": int(sum(memory_by_id[value].arrival_order > int(cutoff_order) for value in selected_ids)),
        "duplicates": int(len(selected_ids) - len(set(selected_ids))),
    }
