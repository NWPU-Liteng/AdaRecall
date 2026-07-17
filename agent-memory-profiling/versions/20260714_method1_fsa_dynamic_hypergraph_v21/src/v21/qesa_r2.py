from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qesa import group_coverage_loss


class LatentAspectQESAR2(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        latent_aspects: int,
        prototypes: int,
        aspect_temperature: float,
    ):
        super().__init__()
        if aspect_temperature <= 0:
            raise ValueError("aspect_temperature must be positive")
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, latent_aspects),
        )
        self.aspect_prototype_logits = nn.Parameter(torch.empty(latent_aspects, prototypes))
        self.aspect_temperature = float(aspect_temperature)
        nn.init.xavier_uniform_(self.aspect_prototype_logits)

    def forward(self, query_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.softmax(self.query_encoder(query_embeddings), dim=-1)
        aspect_map = torch.softmax(self.aspect_prototype_logits / self.aspect_temperature, dim=-1)
        prototype = latent @ aspect_map
        return latent, aspect_map, prototype


def qesa_r2_training_loss(
    model: LatentAspectQESAR2,
    query_embeddings: torch.Tensor,
    group_affinities: torch.Tensor,
    group_query_indices: torch.Tensor,
    query_targets: torch.Tensor,
    *,
    coverage_weight: float,
    target_ce_weight: float,
    diversity_weight: float,
    map_entropy_weight: float,
    aspect_usage_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent, aspect_map, prototype = model(query_embeddings)
    coverage = group_coverage_loss(prototype, group_affinities, group_query_indices)
    target_ce = -(query_targets * prototype.clamp_min(1e-12).log()).sum(dim=1).mean()
    normalized_map = F.normalize(aspect_map, dim=1, eps=1e-12)
    cosine = normalized_map @ normalized_map.T
    off_diagonal = cosine[~torch.eye(len(cosine), dtype=torch.bool, device=cosine.device)]
    diversity = off_diagonal.square().mean() if len(off_diagonal) else coverage.new_zeros(())
    map_entropy = -(aspect_map * aspect_map.clamp_min(1e-12).log()).sum(dim=1).mean()
    usage = latent.mean(dim=0)
    usage_kl = (usage * (usage.clamp_min(1e-12) * len(usage)).log()).sum()
    loss = (
        coverage_weight * coverage
        + target_ce_weight * target_ce
        + diversity_weight * diversity
        + map_entropy_weight * map_entropy
        + aspect_usage_weight * usage_kl
    )
    return loss, {
        "coverage_loss": float(coverage.detach().item()),
        "target_cross_entropy": float(target_ce.detach().item()),
        "aspect_diversity_penalty": float(diversity.detach().item()),
        "aspect_map_entropy": float(map_entropy.detach().item()),
        "aspect_usage_kl": float(usage_kl.detach().item()),
    }
