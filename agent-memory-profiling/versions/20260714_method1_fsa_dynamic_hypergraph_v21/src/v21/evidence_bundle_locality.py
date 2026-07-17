from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eligibility import FORBIDDEN_INFERENCE_FIELDS


@dataclass(frozen=True)
class BundleCandidate:
    """A deployable candidate whose positions come from distinct latent aspects."""

    memory_ids: tuple[str, ...]
    semantic_aspect_ids: tuple[int, ...]
    member_features: np.ndarray

    def __post_init__(self) -> None:
        if not self.memory_ids or len(self.memory_ids) != len(self.semantic_aspect_ids):
            raise ValueError("bundle members and semantic aspects must have the same nonzero cardinality")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("a bundle cannot contain a memory twice")
        if len(set(self.semantic_aspect_ids)) != len(self.semantic_aspect_ids):
            raise ValueError("a bundle must use distinct activated semantic aspects")
        values = np.asarray(self.member_features, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(self.memory_ids) or not np.isfinite(values).all():
            raise ValueError("bundle features must be finite and align with members")


@dataclass(frozen=True)
class BundleInferenceExample:
    query_embedding: np.ndarray
    eligible_memory_ids: tuple[str, ...]
    predicted_aspect_count: int
    candidates: tuple[BundleCandidate, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "BundleInferenceExample":
        forbidden = sorted(FORBIDDEN_INFERENCE_FIELDS.intersection(row.keys()))
        extra_forbidden = sorted({"query_id", "bundle_id", "gold_bundle"}.intersection(row.keys()))
        if forbidden or extra_forbidden:
            raise ValueError(f"bundle inference contains supervision-only fields: {forbidden + extra_forbidden}")
        embedding = np.asarray(row["query_embedding"], dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError("query_embedding must be one finite vector")
        eligible = tuple(dict.fromkeys(map(str, row["eligible_memory_ids"])))
        cardinality = int(row["predicted_aspect_count"])
        candidates = tuple(row["candidates"])
        if cardinality < 1 or not candidates:
            raise ValueError("inference needs a positive predicted cardinality and candidates")
        for candidate in candidates:
            if not isinstance(candidate, BundleCandidate):
                raise TypeError("candidates must be BundleCandidate instances")
            if len(candidate.memory_ids) != cardinality:
                raise ValueError("bundle cardinality must equal predicted evidence-aspect count")
            if not set(candidate.memory_ids).issubset(eligible):
                raise ValueError("bundle contains ineligible memory")
        return cls(embedding, eligible, cardinality, candidates)


class PermutationInvariantBundleScorer(nn.Module):
    """DeepSets scorer: changing member order leaves every score unchanged."""

    def __init__(self, query_dimension: int, member_feature_dimension: int = 3, hidden_dimension: int = 48):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(query_dimension, hidden_dimension), nn.GELU())
        self.member = nn.Sequential(
            nn.Linear(member_feature_dimension + hidden_dimension, hidden_dimension), nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dimension * 3, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, 1))

    def forward(self, query: torch.Tensor, member_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if query.ndim != 2 or member_features.ndim != 3 or len(query) != len(member_features):
            raise ValueError("expected [batch, query_dim] and [batch, members, feature_dim]")
        if mask is None:
            mask = torch.ones(member_features.shape[:2], dtype=torch.bool, device=member_features.device)
        if mask.shape != member_features.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("each bundle needs at least one valid member")
        context = self.query(query)[:, None, :].expand(-1, member_features.shape[1], -1)
        encoded = self.member(torch.cat((member_features, context), dim=-1))
        weight = mask[..., None].to(encoded.dtype)
        mean = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        summed = (encoded * weight).sum(dim=1)
        return self.head(torch.cat((mean, maximum, summed), dim=1)).squeeze(1)


class PairwiseOnlyBundleScorer(nn.Module):
    """Strong ordinary-graph comparator using only unordered member pairs."""

    def __init__(self, query_dimension: int, member_feature_dimension: int = 3, hidden_dimension: int = 48):
        super().__init__()
        self.query = nn.Sequential(nn.Linear(query_dimension, hidden_dimension), nn.GELU())
        self.pair = nn.Sequential(
            nn.Linear(member_feature_dimension * 2 + hidden_dimension, hidden_dimension), nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dimension * 2, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, 1))

    def forward(self, query: torch.Tensor, member_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if query.ndim != 2 or member_features.ndim != 3 or len(query) != len(member_features):
            raise ValueError("expected [batch, query_dim] and [batch, members, feature_dim]")
        if mask is None:
            mask = torch.ones(member_features.shape[:2], dtype=torch.bool, device=member_features.device)
        pairs, pair_masks = [], []
        for left in range(member_features.shape[1]):
            for right in range(left + 1, member_features.shape[1]):
                pairs.append(torch.cat((member_features[:, left], member_features[:, right]), dim=1))
                pair_masks.append(mask[:, left] & mask[:, right])
        if not pairs:
            # A singleton has no ordinary graph edge. Returning a deterministic
            # neutral score preserves the predicted cardinality instead of
            # covertly expanding it only to make the comparator executable.
            return query.new_zeros(len(query))
        pair_values = torch.stack(pairs, dim=1)
        pair_mask = torch.stack(pair_masks, dim=1)
        context = self.query(query)[:, None, :].expand(-1, len(pairs), -1)
        encoded = self.pair(torch.cat((pair_values, context), dim=-1))
        weight = pair_mask[..., None].to(encoded.dtype)
        mean = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~pair_mask[..., None], -torch.inf).amax(dim=1)
        return self.head(torch.cat((mean, maximum), dim=1)).squeeze(1)


class EvidenceAspectCountPredictor(nn.Module):
    def __init__(self, query_dimension: int, hidden_dimension: int, maximum_count: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(query_dimension, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, maximum_count),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.network(query)


def member_feature_matrix(
    query_embedding: np.ndarray,
    memory_vectors: np.ndarray,
    incidence: np.ndarray,
    aspect_prediction: np.ndarray,
) -> np.ndarray:
    """Label-free features available to bundle construction and scoring."""
    cosine = memory_vectors @ query_embedding
    association = incidence @ aspect_prediction
    peak_incidence = incidence.max(axis=1)
    return np.stack((cosine, association, peak_incidence), axis=1).astype(np.float32)


def build_semantic_bundle_candidates(
    *,
    eligible_memory_ids: Sequence[str],
    eligible_member_features: np.ndarray,
    eligible_incidence: np.ndarray,
    aspect_prediction: np.ndarray,
    predicted_aspect_count: int,
    activated_aspects: int,
    members_per_aspect: int,
    beam_width: int,
) -> tuple[list[BundleCandidate], dict[str, int]]:
    """Build a bounded Cartesian beam without evidence-group or gold-bundle access."""
    if predicted_aspect_count < 1 or predicted_aspect_count > activated_aspects:
        raise ValueError("predicted cardinality must fit the activated semantic-aspect budget")
    if len(eligible_memory_ids) != len(eligible_member_features) or len(eligible_memory_ids) != len(eligible_incidence):
        raise ValueError("eligible ids, features, and incidence must align")
    aspect_order = np.argsort(-np.asarray(aspect_prediction), kind="stable")[:activated_aspects]
    active = tuple(map(int, aspect_order[:predicted_aspect_count]))
    options: list[list[int]] = []
    for aspect in active:
        semantic = eligible_incidence[:, aspect] * float(aspect_prediction[aspect])
        score = eligible_member_features[:, 0] + eligible_member_features[:, 1] + semantic
        positions = np.argsort(-score, kind="stable")[: min(members_per_aspect, len(score))]
        options.append(list(map(int, positions)))
    generated = 0
    duplicates = 0
    candidates: list[tuple[float, BundleCandidate]] = []
    seen: set[tuple[str, ...]] = set()
    for positions in product(*options):
        generated += 1
        if len(set(positions)) != len(positions):
            duplicates += 1
            continue
        ids = tuple(str(eligible_memory_ids[position]) for position in positions)
        key = tuple(sorted(ids))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        features = np.asarray([eligible_member_features[position] for position in positions], dtype=np.float32)
        heuristic = float(features[:, :2].sum())
        candidates.append((heuristic, BundleCandidate(ids, active, features)))
    candidates.sort(key=lambda item: (-item[0], item[1].memory_ids))
    selected = [candidate for _score, candidate in candidates[:beam_width]]
    return selected, {
        "activated_semantic_edges": len(active),
        "bundle_hyperedges_generated": generated,
        "duplicate_bundle_expansions": duplicates,
        "bundle_hyperedges_retained": len(selected),
        "unique_expanded_memories": len({memory for candidate in selected for memory in candidate.memory_ids}),
    }


def bundle_outcome(candidate: BundleCandidate, groups: Sequence[Sequence[str]]) -> dict[str, float]:
    selected = set(candidate.memory_ids)
    matched = [bool(selected.intersection(group)) for group in groups]
    counts = [len(selected.intersection(group)) for group in groups]
    return {
        "all_groups_covered": float(all(matched)),
        "bundle_recall": float(np.mean(matched)) if matched else 0.0,
        "bundle_member_redundancy": float(sum(max(0, count - 1) for count in counts) / max(1, len(candidate.memory_ids))),
    }


def ranked_bundle_metrics(ranked: Sequence[BundleCandidate], groups: Sequence[Sequence[str]], evaluation_beam: int) -> dict[str, float]:
    if not ranked:
        return {
            "gold_compatible_bundle_hit_rate": 0.0, "all_groups_covered_rate": 0.0,
            "bundle_recall": 0.0, "bundle_member_redundancy": 0.0,
        }
    top = bundle_outcome(ranked[0], groups)
    explored = [bundle_outcome(candidate, groups) for candidate in ranked[:evaluation_beam]]
    return {
        "gold_compatible_bundle_hit_rate": float(any(item["all_groups_covered"] for item in explored)),
        "all_groups_covered_rate": top["all_groups_covered"],
        "bundle_recall": top["bundle_recall"],
        "bundle_member_redundancy": top["bundle_member_redundancy"],
    }


def binary_bundle_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape != labels.shape:
        raise ValueError("bundle logits and labels must align")
    return F.binary_cross_entropy_with_logits(logits, labels)
