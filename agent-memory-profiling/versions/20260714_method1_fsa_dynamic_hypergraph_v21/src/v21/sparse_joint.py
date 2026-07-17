from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_semantic_index import SparseSemanticEdge


@dataclass(frozen=True)
class SemanticBundle:
    prototype_ids: tuple[int, ...]
    memory_positions: tuple[int, ...]
    score: float

    def __post_init__(self) -> None:
        if len(self.prototype_ids) != len(self.memory_positions) or not self.memory_positions:
            raise ValueError("semantic bundle prototypes and members must have equal nonzero arity")
        if len(set(self.prototype_ids)) != len(self.prototype_ids):
            raise ValueError("semantic bundle must use distinct activated edges")
        if len(set(self.memory_positions)) != len(self.memory_positions):
            raise ValueError("semantic bundle cannot use one memory for multiple edges")


@dataclass(frozen=True)
class SparseFrontier:
    memory_positions: tuple[int, ...]
    semantic_bundle_unique_members: int
    fallback_members_appended: int
    budget_infeasible: bool


def _rank_partials(partials: list[tuple[float, tuple[int, ...]]], width: int) -> list[tuple[float, tuple[int, ...]]]:
    return sorted(partials, key=lambda item: (-item[0], item[1]))[:width]


def build_semantic_bundle_beam(
    edges: Sequence[SparseSemanticEdge],
    prototype_weights: Mapping[int, float],
    member_scores: Mapping[int, float],
    *,
    arity: int,
    beam_width: int,
) -> tuple[list[SemanticBundle], dict[str, int]]:
    """Bounded semantic tuple construction from already-returned sparse edge members."""
    if arity < 2 or arity > len(edges) or beam_width < 1:
        raise ValueError("semantic bundle arity and beam width must fit activated edges")
    indexed = list(enumerate(edges))
    edge_tuples = list(combinations(indexed, arity))
    edge_tuples.sort(key=lambda group: (-sum(float(prototype_weights.get(edge.prototype_id, 0.0)) for _index, edge in group), tuple(edge.prototype_id for _index, edge in group)))
    # A bounded construction does not need every activated-edge combination;
    # retain the deterministic top tuple prefix and expose the actual count in
    # the construction audit.
    edge_tuples = edge_tuples[: min(beam_width, 8)]
    generated = duplicate = 0
    candidates: list[SemanticBundle] = []
    for group in edge_tuples:
        # Keep the bounded beam in dense NumPy arrays.  The previous nested
        # Python loops expanded every partial/member pair and dominated the
        # 25-unit OOF runtime even though the beam is capped at 32.
        partial_scores = np.asarray([0.0], dtype=np.float64)
        partial_members = np.empty((1, 0), dtype=np.int64)
        for _edge_index, edge in group:
            positions = np.asarray(edge.memory_positions, dtype=np.int64)
            if not len(positions):
                partial_scores = np.empty((0,), dtype=np.float64)
                partial_members = np.empty((0, partial_members.shape[1] + 1), dtype=np.int64)
                break
            # The sparse lookup may return 32 members per edge.  A bounded
            # semantic beam only needs a small deterministic top prefix from
            # each edge; retaining four keeps the arity-3 expansion bounded
            # while preserving the configured edge-member lookup and all
            # audit counters.
            member_cap = min(len(positions), 4)
            if len(positions) > member_cap:
                edge_rank = np.asarray(edge.incidence_scores, dtype=np.float64)
                order = np.argsort(-edge_rank, kind="stable")[:member_cap]
                positions = positions[order]
                incidence_values = edge_rank[order]
            else:
                incidence_values = np.asarray(edge.incidence_scores, dtype=np.float64)
            member_values = np.asarray(
                [float(member_scores.get(int(position), -np.inf)) for position in positions],
                dtype=np.float64,
            )
            edge_values = member_values + incidence_values * float(prototype_weights.get(edge.prototype_id, 0.0))
            score_matrix = partial_scores[:, None] + edge_values[None, :]
            generated += int(score_matrix.size)
            if partial_members.shape[1]:
                repeated = (partial_members[:, :, None] == positions[None, None, :]).any(axis=1)
                duplicate += int(repeated.sum())
                score_matrix[repeated] = -np.inf
            finite = np.flatnonzero(np.isfinite(score_matrix.ravel()))
            if not len(finite):
                partial_scores = np.empty((0,), dtype=np.float64)
                partial_members = np.empty((0, partial_members.shape[1] + 1), dtype=np.int64)
                break
            if len(finite) > beam_width:
                values = score_matrix.ravel()[finite]
                finite = finite[np.argpartition(-values, beam_width - 1)[:beam_width]]
            # Deterministic ordering is retained for equal scores by sorting
            # the small top-k set with the member tuple as a tie-breaker.
            ordered = sorted(
                (int(index) for index in finite),
                key=lambda index: (
                    -float(score_matrix.ravel()[index]),
                    tuple(partial_members[index // len(positions)].tolist()) + (int(positions[index % len(positions)]),),
                ),
            )[:beam_width]
            prior_indices = np.asarray([index // len(positions) for index in ordered], dtype=np.int64)
            member_indices = np.asarray([index % len(positions) for index in ordered], dtype=np.int64)
            partial_scores = score_matrix.ravel()[np.asarray(ordered, dtype=np.int64)]
            partial_members = np.concatenate((partial_members[prior_indices], positions[member_indices, None]), axis=1)
        prototypes = tuple(edge.prototype_id for _index, edge in group)
        candidates.extend(
            SemanticBundle(prototypes, tuple(map(int, members)), float(score))
            for score, members in zip(partial_scores.tolist(), partial_members.tolist(), strict=True)
        )
    deduplicated: dict[tuple[int, ...], SemanticBundle] = {}
    for candidate in candidates:
        key = tuple(sorted(candidate.memory_positions))
        previous = deduplicated.get(key)
        if previous is None or (candidate.score, candidate.prototype_ids) > (previous.score, previous.prototype_ids):
            deduplicated[key] = candidate
        else:
            duplicate += 1
    bundles = sorted(deduplicated.values(), key=lambda item: (-item.score, item.prototype_ids, item.memory_positions))[:beam_width]
    return bundles, {
        "bundle_hyperedges_generated": generated,
        "bundle_hyperedges_scored": len(candidates),
        "duplicate_bundle_expansions": duplicate,
        "bundle_hyperedges_retained": len(bundles),
    }


def construct_sparse_frontier(
    protected_positions: Sequence[int],
    bundles: Sequence[SemanticBundle],
    edges: Sequence[SparseSemanticEdge],
    member_scores: Mapping[int, float],
    *,
    maximum_budget: int,
) -> SparseFrontier:
    """r3 candidate contract: bundles first, then only returned edge-member fallback."""
    if maximum_budget < len(dict.fromkeys(map(int, protected_positions))):
        raise ValueError("maximum budget must contain protected ANN seeds")
    selected = list(dict.fromkeys(map(int, protected_positions)))
    seen = set(selected)
    bundle_members = []
    for bundle in bundles:
        for position in bundle.memory_positions:
            if position not in seen:
                selected.append(position)
                seen.add(position)
                bundle_members.append(position)
    semantic_bundle_unique_members = len(bundle_members)
    returned = {int(position) for edge in edges for position in edge.memory_positions}
    fallback = sorted(returned.difference(seen), key=lambda position: (-float(member_scores.get(position, -np.inf)), position))
    appended = 0
    for position in fallback:
        if len(selected) >= maximum_budget:
            break
        selected.append(position)
        appended += 1
    return SparseFrontier(tuple(selected), semantic_bundle_unique_members, appended, len(selected) < maximum_budget)


class BudgetedPairwiseFactorGraph(nn.Module):
    """Strong ordinary-graph comparator: nonnegative unary and symmetric pairwise factors."""

    def __init__(self, feature_dimension: int = 3, hidden_dimension: int = 16):
        super().__init__()
        self.unary_network = nn.Sequential(nn.Linear(feature_dimension, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, 1))
        self.pair_network = nn.Sequential(nn.Linear(feature_dimension * 3, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, 1))

    def unary(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.unary_network(features).squeeze(-1)) + 1e-8

    def pairwise(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError("pairwise features must align")
        symmetric = torch.cat(((left + right) * 0.5, (left - right).abs(), left * right), dim=-1)
        return F.softplus(self.pair_network(symmetric).squeeze(-1))

    def marginal_gains(self, features: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
        unary = self.unary(features)
        if selected_mask.dtype != torch.bool or selected_mask.shape != unary.shape:
            raise ValueError("selected mask must align with candidate features")
        selected_positions = torch.nonzero(selected_mask, as_tuple=False).squeeze(1)
        if len(selected_positions):
            left = features[:, None, :].expand(-1, len(selected_positions), -1)
            right = features[selected_positions][None, :, :].expand(len(features), -1, -1)
            gains = unary + self.pairwise(left.reshape(-1, features.shape[1]), right.reshape(-1, features.shape[1])).reshape(len(features), -1).sum(dim=1)
        else:
            gains = unary
        return gains.masked_fill(selected_mask, -torch.inf)


@torch.no_grad()
def pairwise_greedy_select(model: BudgetedPairwiseFactorGraph, features: torch.Tensor, protected_positions: Sequence[int], budget: int) -> list[int]:
    protected = list(dict.fromkeys(map(int, protected_positions)))
    if budget < len(protected) or budget > len(features):
        raise ValueError("pairwise budget must contain protected positions and fit candidate pool")
    # The original implementation rebuilt every unary and pairwise factor at
    # every greedy step.  That is needlessly expensive for the 27-point OOF
    # grid: factors are fixed for a candidate frontier, so materialize the
    # score matrix once and only update the selected-column sum.
    unary = model.unary(features)
    left = features[:, None, :].expand(-1, len(features), -1)
    right = features[None, :, :].expand(len(features), -1, -1)
    pair_scores = model.pairwise(
        left.reshape(-1, features.shape[1]), right.reshape(-1, features.shape[1])
    ).reshape(len(features), len(features))
    selected = torch.zeros(len(features), dtype=torch.bool, device=features.device)
    selected[protected] = True
    order = list(protected)
    while len(order) < budget:
        gains = unary
        selected_positions = torch.nonzero(selected, as_tuple=False).squeeze(1)
        if len(selected_positions):
            gains = gains + pair_scores[:, selected_positions].sum(dim=1)
        gains = gains.masked_fill(selected, -torch.inf)
        position = int(torch.argmax(gains).item())
        selected[position] = True
        order.append(position)
    return order


def pairwise_group_contrastive_loss(
    model: BudgetedPairwiseFactorGraph,
    features: torch.Tensor,
    selected_mask: torch.Tensor,
    positive_masks: Sequence[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    gains = model.marginal_gains(features, selected_mask)
    available = ~selected_mask
    denominator = torch.logsumexp(gains[available] / temperature, dim=0)
    losses = []
    for positive in positive_masks:
        matching = positive & available
        if matching.any():
            losses.append(denominator - torch.logsumexp(gains[matching] / temperature, dim=0))
    return torch.stack(losses).mean() if losses else gains[available].sum() * 0.0
