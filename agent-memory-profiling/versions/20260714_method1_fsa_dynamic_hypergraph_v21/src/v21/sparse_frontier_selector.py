"""Candidate utility model trained on the actual sparse deployment frontier."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SparseFrontierUtilityMLP(nn.Module):
    def __init__(self, feature_dimension: int = 4, hidden_dimension: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2:
            raise ValueError("candidate features require a final feature dimension")
        return self.network(features).squeeze(-1)


class StructureAwareMarginalSelector(nn.Module):
    """Score unary evidence and stateful hypergraph marginals separately."""

    def __init__(self, unary_dimension: int = 4, structural_dimension: int = 5, hidden_dimension: int = 32) -> None:
        super().__init__()
        self.unary_network = nn.Sequential(
            nn.Linear(unary_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.structural_network = nn.Sequential(
            nn.Linear(structural_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, unary_features: torch.Tensor, structural_features: torch.Tensor) -> torch.Tensor:
        if unary_features.shape[:-1] != structural_features.shape[:-1]:
            raise ValueError("unary and structural candidate axes must align")
        unary = self.unary_network(unary_features).squeeze(-1)
        structural = self.structural_network(structural_features).squeeze(-1)
        return unary + structural


def structure_marginal_features(
    selected_mask: np.ndarray,
    bundle_incidence: np.ndarray,
    bundle_weights: np.ndarray,
    prototype_incidence: np.ndarray,
    prototype_weights: np.ndarray,
    candidate_similarity: np.ndarray,
) -> np.ndarray:
    """Return stateful bundle, prototype-coverage, and diversity marginals."""
    selected = np.asarray(selected_mask, dtype=bool)
    bundles = np.asarray(bundle_incidence, dtype=bool)
    bundle_weight = np.asarray(bundle_weights, dtype=np.float32)
    prototypes = np.asarray(prototype_incidence, dtype=bool)
    prototype_weight = np.asarray(prototype_weights, dtype=np.float32)
    similarity = np.asarray(candidate_similarity, dtype=np.float32)
    candidates = len(selected)
    if bundles.ndim != 2 or bundles.shape[1] != candidates or bundle_weight.shape != (len(bundles),):
        raise ValueError("bundle incidence and weights must align with candidates")
    if prototypes.ndim != 2 or prototypes.shape[1] != candidates or prototype_weight.shape != (len(prototypes),):
        raise ValueError("prototype incidence and weights must align with candidates")
    if similarity.shape != (candidates, candidates):
        raise ValueError("candidate similarity must be a square candidate matrix")

    if len(bundles):
        bundle_sizes = bundles.sum(axis=1).astype(np.float32)
        selected_per_bundle = bundles[:, selected].sum(axis=1).astype(np.float32)
        remaining = np.maximum(bundle_sizes - selected_per_bundle, 1.0)
        incomplete = selected_per_bundle < bundle_sizes
        bundle_progress = (bundles * (bundle_weight * incomplete / remaining)[:, None]).sum(axis=0)
        completion = incomplete & (selected_per_bundle == bundle_sizes - 1)
        bundle_completion = (bundles * (bundle_weight * completion)[:, None]).sum(axis=0)
    else:
        bundle_progress = np.zeros(candidates, dtype=np.float32)
        bundle_completion = np.zeros(candidates, dtype=np.float32)

    if len(prototypes):
        covered = prototypes[:, selected].any(axis=1)
        prototype_novelty = (prototypes * (prototype_weight * ~covered)[:, None]).sum(axis=0)
        prototype_covered = (prototypes * (prototype_weight * covered)[:, None]).sum(axis=0)
    else:
        prototype_novelty = np.zeros(candidates, dtype=np.float32)
        prototype_covered = np.zeros(candidates, dtype=np.float32)

    if selected.any():
        diversity = 1.0 - similarity[:, selected].max(axis=1)
    else:
        diversity = np.ones(candidates, dtype=np.float32)
    output = np.stack(
        (bundle_completion, bundle_progress, prototype_novelty, prototype_covered, diversity),
        axis=1,
    ).astype(np.float32)
    output[selected] = 0.0
    return output


def group_target_cross_entropy(
    scores: torch.Tensor,
    target_distribution: torch.Tensor,
    protected_mask: torch.Tensor,
) -> torch.Tensor:
    if scores.shape != target_distribution.shape or scores.shape != protected_mask.shape:
        raise ValueError("scores, targets, and protected mask must align")
    if protected_mask.dtype != torch.bool:
        raise ValueError("protected mask must be boolean")
    masked = scores.masked_fill(protected_mask, -torch.inf)
    log_probabilities = torch.log_softmax(masked, dim=-1)
    valid = target_distribution.sum(dim=-1) > 0
    if not valid.any():
        return scores.sum() * 0.0
    losses = -(target_distribution[valid] * log_probabilities[valid].nan_to_num()).sum(dim=-1)
    return losses.mean()
