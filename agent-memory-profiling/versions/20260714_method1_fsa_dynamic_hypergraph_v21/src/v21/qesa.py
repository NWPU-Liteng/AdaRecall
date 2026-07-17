from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eligibility import FORBIDDEN_INFERENCE_FIELDS


@dataclass(frozen=True)
class QESAInferenceExample:
    query_embedding: np.ndarray
    protected_ann_seed_ids: tuple[str, ...]
    eligible_memory_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "QESAInferenceExample":
        forbidden = sorted(FORBIDDEN_INFERENCE_FIELDS.intersection(row.keys()))
        if forbidden:
            raise ValueError(f"QESA inference input contains supervision-only fields: {forbidden}")
        embedding = np.asarray(row["query_embedding"], dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError("query_embedding must be one finite vector")
        protected = tuple(dict.fromkeys(map(str, row["protected_ann_seed_ids"])))
        eligible = tuple(dict.fromkeys(map(str, row["eligible_memory_ids"])))
        if not set(protected).issubset(eligible):
            raise ValueError("protected ANN seeds must be eligible")
        return cls(embedding, protected, eligible)


class LatentAspectQESA(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, latent_aspects: int, prototypes: int):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, latent_aspects),
        )
        self.aspect_prototype_logits = nn.Parameter(torch.empty(latent_aspects, prototypes))
        nn.init.xavier_uniform_(self.aspect_prototype_logits)

    def forward(self, query_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.softmax(self.query_encoder(query_embeddings), dim=-1)
        aspect_to_prototype = torch.softmax(self.aspect_prototype_logits, dim=-1)
        prototype = latent @ aspect_to_prototype
        return latent, aspect_to_prototype, prototype


class LinearPrototypeProbe(nn.Module):
    def __init__(self, input_dimension: int, prototypes: int):
        super().__init__()
        self.linear = nn.Linear(input_dimension, prototypes)

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(query_embeddings), dim=-1)


class FlatMLPRanker(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, input_dimension),
        )

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(query_embeddings), dim=-1, eps=1e-12)


def alternative_group_affinity(member_incidence: torch.Tensor) -> torch.Tensor:
    """OR semantics: each prototype receives the best acceptable-member affinity."""
    if member_incidence.ndim != 2 or len(member_incidence) < 1:
        raise ValueError("one or more acceptable members are required")
    if not torch.isfinite(member_incidence).all() or (member_incidence < 0).any():
        raise ValueError("member incidence must be finite and nonnegative")
    affinity = member_incidence.max(dim=0).values
    maximum = affinity.max().clamp_min(1e-12)
    return affinity / maximum


def group_coverage_loss(
    prototype_probabilities: torch.Tensor,
    group_affinities: torch.Tensor,
    group_query_indices: torch.Tensor,
) -> torch.Tensor:
    if prototype_probabilities.ndim != 2 or group_affinities.ndim != 2:
        raise ValueError("probabilities and affinities must be matrices")
    if prototype_probabilities.shape[1] != group_affinities.shape[1]:
        raise ValueError("prototype dimension mismatch")
    selected = prototype_probabilities[group_query_indices]
    coverage = (selected * group_affinities).sum(dim=1).clamp(1e-8, 1.0)
    return -coverage.log().mean()


def qesa_training_loss(
    model: LatentAspectQESA,
    query_embeddings: torch.Tensor,
    group_affinities: torch.Tensor,
    group_query_indices: torch.Tensor,
    *,
    diversity_weight: float,
    entropy_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent, aspect_map, prototype = model(query_embeddings)
    coverage = group_coverage_loss(prototype, group_affinities, group_query_indices)
    normalized_map = F.normalize(aspect_map, dim=1, eps=1e-12)
    cosine = normalized_map @ normalized_map.T
    off_diagonal = cosine[~torch.eye(len(cosine), dtype=torch.bool, device=cosine.device)]
    diversity = off_diagonal.square().mean() if len(off_diagonal) else coverage.new_zeros(())
    entropy = -(latent * latent.clamp_min(1e-12).log()).sum(dim=1).mean()
    loss = coverage + diversity_weight * diversity - entropy_weight * entropy
    return loss, {
        "coverage_loss": float(coverage.detach().item()),
        "aspect_diversity_penalty": float(diversity.detach().item()),
        "prediction_entropy": float(entropy.detach().item()),
    }


def select_protected_frontier(
    scores: np.ndarray,
    eligible_indices: Sequence[int],
    protected_indices: Sequence[int],
    budget: int,
) -> list[int]:
    if budget < len(dict.fromkeys(protected_indices)):
        raise ValueError("budget cannot be smaller than the protected seed count")
    eligible = set(map(int, eligible_indices))
    protected = list(dict.fromkeys(map(int, protected_indices)))
    if not set(protected).issubset(eligible):
        raise ValueError("protected indices must be eligible")
    selected = list(protected)
    remaining = sorted((index for index in eligible if index not in set(selected)), key=lambda index: (-float(scores[index]), index))
    selected.extend(remaining[: max(0, budget - len(selected))])
    return selected


def evidence_metrics(selected_ids: Sequence[str], groups: Sequence[Sequence[str]]) -> dict[str, float]:
    selected = set(map(str, selected_ids))
    covered = [bool(selected.intersection(map(str, group))) for group in groups]
    both = [len(selected.intersection(map(str, group))) >= 2 for group in groups]
    return {
        "group_coverage": float(np.mean(covered)) if covered else 0.0,
        "all_groups_covered": float(all(covered)) if covered else 0.0,
        "both_alternatives_selected_rate": float(np.mean(both)) if both else 0.0,
    }
