from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualProjectionAdapter(nn.Module):
    def __init__(self, dimension: int = 768, bottleneck: int = 128):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dimension, bias=False)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = values + self.up(F.gelu(self.down(values)))
        return F.normalize(projected, dim=-1, eps=1e-12)


def pairwise_contrastive_loss(adapter: ResidualProjectionAdapter, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor = adapter(anchors)
    positive = adapter(positives)
    negative = adapter(negatives)
    logits = torch.stack([(anchor * positive).sum(dim=-1), (anchor * negative).sum(dim=-1)], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


def multi_positive_info_nce(
    adapter: ResidualProjectionAdapter,
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss with an explicit multi-positive numerator."""
    if positive_mask.dtype != torch.bool or positive_mask.shape != (len(anchors), len(candidates)):
        raise ValueError("positive mask shape/type mismatch")
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("every anchor requires at least one positive")
    scores = adapter(anchors) @ adapter(candidates).T / temperature
    numerator = torch.logsumexp(scores.masked_fill(~positive_mask, float("-inf")), dim=1)
    denominator = torch.logsumexp(scores, dim=1)
    return (denominator - numerator).mean()


@torch.no_grad()
def project_matrix(adapter: ResidualProjectionAdapter, values: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    adapter.eval()
    return torch.cat([adapter(values[start : start + batch_size]) for start in range(0, len(values), batch_size)], dim=0)
