from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eligibility import FORBIDDEN_INFERENCE_FIELDS


@dataclass(frozen=True)
class QHDSInferenceExample:
    query_embedding: np.ndarray
    protected_ann_seed_ids: tuple[str, ...]
    eligible_memory_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "QHDSInferenceExample":
        forbidden = sorted(FORBIDDEN_INFERENCE_FIELDS.intersection(row.keys()))
        if forbidden:
            raise ValueError(f"QHDS inference input contains supervision-only fields: {forbidden}")
        embedding = np.asarray(row["query_embedding"], dtype=np.float32)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError("query_embedding must be one finite vector")
        protected = tuple(dict.fromkeys(map(str, row["protected_ann_seed_ids"])))
        eligible = tuple(dict.fromkeys(map(str, row["eligible_memory_ids"])))
        if not set(protected).issubset(eligible):
            raise ValueError("protected ANN seeds must be eligible")
        return cls(embedding, protected, eligible)


class MonotoneSubmodularQHDS(nn.Module):
    def __init__(self, feature_dimension: int = 3, hidden_dimension: int = 16):
        super().__init__()
        self.relevance_network = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.structural_scale_logit = nn.Parameter(torch.tensor(0.0))
        self.modular_scale_logit = nn.Parameter(torch.tensor(0.0))

    def relevance(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.relevance_network(features).squeeze(-1)) + 1e-8

    def positive_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.structural_scale_logit), F.softplus(self.modular_scale_logit)

    def utility(
        self,
        selected_mask: torch.Tensor,
        aspect_probabilities: torch.Tensor,
        incidence: torch.Tensor,
        relevance: torch.Tensor,
    ) -> torch.Tensor:
        structural_scale, modular_scale = self.positive_scales()
        selected = selected_mask.to(relevance.dtype)
        covered = (incidence * (selected * relevance)[:, None]).sum(dim=0)
        structural = (aspect_probabilities * torch.log1p(covered)).sum()
        modular = (selected * relevance).sum()
        return structural_scale * structural + modular_scale * modular

    def marginal_gains(
        self,
        selected_mask: torch.Tensor,
        aspect_probabilities: torch.Tensor,
        incidence: torch.Tensor,
        relevance: torch.Tensor,
    ) -> torch.Tensor:
        structural_scale, modular_scale = self.positive_scales()
        selected = selected_mask.to(relevance.dtype)
        covered = (incidence * (selected * relevance)[:, None]).sum(dim=0)
        added = incidence * relevance[:, None]
        structural_delta = ((torch.log1p(covered[None, :] + added) - torch.log1p(covered)[None, :]) * aspect_probabilities[None, :]).sum(dim=1)
        gains = structural_scale * structural_delta + modular_scale * relevance
        return gains.masked_fill(selected_mask, -torch.inf)


def build_features(cosine: torch.Tensor, association: torch.Tensor) -> torch.Tensor:
    if cosine.shape != association.shape:
        raise ValueError("cosine and association vectors must have the same shape")
    return torch.stack((cosine, association, cosine * association), dim=-1)


def group_contrastive_loss(
    model: MonotoneSubmodularQHDS,
    features: torch.Tensor,
    incidence: torch.Tensor,
    aspect_probabilities: torch.Tensor,
    selected_mask: torch.Tensor,
    positive_masks: Sequence[torch.Tensor],
    temperature: float,
    scale_regularization: float,
) -> torch.Tensor:
    relevance = model.relevance(features)
    gains = model.marginal_gains(selected_mask, aspect_probabilities, incidence, relevance)
    available = ~selected_mask
    denominator = torch.logsumexp(gains[available] / temperature, dim=0)
    losses = []
    for positive_mask in positive_masks:
        valid_positive = positive_mask & available
        if valid_positive.any():
            numerator = torch.logsumexp(gains[valid_positive] / temperature, dim=0)
            losses.append(denominator - numerator)
    if not losses:
        return gains[available].sum() * 0.0
    structural_scale, modular_scale = model.positive_scales()
    regularizer = scale_regularization * (structural_scale.square() + modular_scale.square())
    return torch.stack(losses).mean() + regularizer


@torch.no_grad()
def greedy_select(
    model: MonotoneSubmodularQHDS,
    features: torch.Tensor,
    incidence: torch.Tensor,
    aspect_probabilities: torch.Tensor,
    protected_positions: Sequence[int],
    budget: int,
    *,
    structural: bool = True,
) -> list[int]:
    protected = list(dict.fromkeys(map(int, protected_positions)))
    if budget < len(protected) or budget > len(features):
        raise ValueError("budget must contain all protected positions and fit the candidate set")
    selected = torch.zeros(len(features), dtype=torch.bool, device=features.device)
    selected[protected] = True
    relevance = model.relevance(features)
    addition_order = []
    while int(selected.sum().item()) < budget:
        if structural:
            gains = model.marginal_gains(selected, aspect_probabilities, incidence, relevance)
        else:
            gains = relevance.masked_fill(selected, -torch.inf)
        position = int(torch.argmax(gains).item())
        selected[position] = True
        addition_order.append(position)
    return protected + addition_order
