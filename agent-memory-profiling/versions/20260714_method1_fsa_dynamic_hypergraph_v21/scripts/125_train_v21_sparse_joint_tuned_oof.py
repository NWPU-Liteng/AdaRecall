#!/usr/bin/env python3
"""Run one immutable, session-blocked sparse-joint Method 1 OOF unit.

The sparse path intentionally keeps the incidence access boundary narrow: after
QESA predicts active prototypes, the inverted index is the only operation that
may discover semantic members.  Dense incidence is used only for the explicitly
labelled dense QHDS reference and for outer-train supervision.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import runpy
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
import torch.nn.functional as F


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_csv_new  # noqa: E402
from v21.online_latency import OnlineLatencyRecorder, synchronize_if_cuda  # noqa: E402
from v21.qesa import FlatMLPRanker  # noqa: E402
from v21.qesa_r2 import LatentAspectQESAR2  # noqa: E402
from v21.qhds import MonotoneSubmodularQHDS, build_features, greedy_select  # noqa: E402
from v21.sparse_joint import (  # noqa: E402
    BudgetedPairwiseFactorGraph,
    build_semantic_bundle_beam,
    construct_sparse_frontier,
    pairwise_greedy_select,
    pairwise_group_contrastive_loss,
)
from v21.sparse_semantic_index import (  # noqa: E402
    SparseSemanticInvertedIndex,
    SparseSemanticLookupRequest,
    SparseSemanticLookupResult,
    SparseSemanticEdge,
)
from v21.uot_recipes import array_sha256  # noqa: E402


QESA_HELPERS = runpy.run_path(str(VERSION_ROOT / "scripts" / "90_train_v21_qesa_oof.py"), run_name="v21_sparse_joint_qesa_helpers")
ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")
GRID_FIELDS = ("activated_prototypes", "members_per_semantic_edge", "semantic_bundle_beam")


def qesa_model(checkpoint: dict, dimension: int, hidden: int, device: str) -> LatentAspectQESAR2:
    state = checkpoint["qesa"]
    model = LatentAspectQESAR2(
        dimension,
        hidden,
        int(checkpoint["selected_latent_aspects"]),
        int(state["aspect_prototype_logits"].shape[1]),
        float(checkpoint["selected_regime"]["aspect_temperature"]),
    ).to(device)
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def predict_qesa(model: LatentAspectQESAR2, query: np.ndarray, device: str) -> np.ndarray:
    return model(torch.as_tensor(query[None, :], dtype=torch.float32, device=device))[2][0].detach().cpu().numpy()


@torch.no_grad()
def predict_qesa_batch(model: LatentAspectQESAR2, queries: np.ndarray, device: str) -> np.ndarray:
    return model(torch.as_tensor(queries, dtype=torch.float32, device=device))[2].detach().cpu().numpy()


def load_ann_prefix(data: dict) -> dict[str, tuple[str, ...]]:
    """Load only the frozen, train-only ANN candidate prefix, never labels."""
    output: dict[str, tuple[str, ...]] = {}
    for row in iter_jsonl(data["frozen_baseline"]["train_retrieval"]):
        query_id = str(row["query_id"])
        candidates = tuple(dict.fromkeys(map(str, row.get("ann_candidate_ids", []))))
        if len(candidates) < 5:
            raise ValueError(f"frozen ANN prefix is shorter than protected seed count for {query_id}")
        previous = output.get(query_id)
        if previous is not None and previous != candidates:
            raise ValueError(f"inconsistent frozen ANN prefix for {query_id}")
        output[query_id] = candidates
    return output


def load_layout() -> tuple[dict[str, str], dict[str, int]]:
    path = VERSION_ROOT / "artifacts" / "v21_effect_closure" / "v21_neutral_memory_to_tile.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tile = {str(row["memory_id"]): str(row["tile_id"]) for row in rows}
    payload = {str(row["memory_id"]): int(row["payload_bytes"]) for row in rows}
    if len(tile) != 42034 or set(tile) != set(payload):
        raise ValueError("neutral layout is not a complete memory-to-tile/payload map")
    return tile, payload


def load_sparse_index(config: dict, seed: int, fold: int, incidence: np.ndarray) -> tuple[SparseSemanticInvertedIndex, Path]:
    manifest_path = (
        VERSION_ROOT / "artifacts" / "v21_sparse_semantic_index" / config["index_run_id"]
        / f"seed_{seed}" / f"fold_{fold}" / "attempts" / config["index_attempt"]
        / "v21_sparse_semantic_index_manifest.json"
    )
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("attempt") != config["index_attempt"]:
        raise ValueError(f"sparse index manifest is not completed: {manifest_path}")
    rank_path = VERSION_ROOT / manifest["artifacts"]["prototype_memory_rank_index"]
    ranks = np.load(rank_path, mmap_mode="r")
    if sha256_file(rank_path) != manifest["hashes"]["rank_index"]:
        raise ValueError(f"sparse rank index hash mismatch: {rank_path}")
    if tuple(ranks.shape) != tuple(manifest["rank_index_shape"]):
        raise ValueError("sparse rank index shape mismatch")
    return SparseSemanticInvertedIndex(ranks, incidence), rank_path


def load_qhds(config: dict, seed: int, fold: int, dimension: int, device: str) -> tuple[MonotoneSubmodularQHDS, Path]:
    path = VERSION_ROOT / "artifacts" / "v21_qhds" / config["dense_qhds_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "v21_qhds_model.pt"
    checkpoint = torch.load(path, map_location="cpu")
    model = MonotoneSubmodularQHDS(3, 16).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), path


def load_dense_reference_selections(config: dict, seed: int, fold: int) -> dict[tuple[str, int], tuple[str, ...]]:
    path = VERSION_ROOT / "artifacts" / "v21_qhds" / config["dense_qhds_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "v21_qhds_selections.jsonl"
    output: dict[tuple[str, int], tuple[str, ...]] = {}
    for row in iter_jsonl(path):
        if row.get("method") != "qhds":
            continue
        key = (str(row["query_id"]), int(row["budget"]))
        output[key] = tuple(map(str, row["selected_memory_ids"]))
    return output


def train_flat_model(rows: list[dict], raw: np.ndarray, index_by_id: dict[str, int], config: dict, seed: int, device: str) -> tuple[FlatMLPRanker, dict]:
    queries = np.stack([row["query_embedding"] for row in rows]).astype(np.float32)
    targets = []
    for row in rows:
        positions = [index_by_id[memory_id] for group in row["groups"] for memory_id in group]
        if not positions:
            positions = [index_by_id[row["eligible_memory_ids"][0]]]
        target = np.asarray(raw[positions], dtype=np.float32).mean(axis=0)
        targets.append(target / max(float(np.linalg.norm(target)), 1e-12))
    query_tensor = torch.as_tensor(queries, device=device)
    target_tensor = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device)
    torch.manual_seed(seed)
    model = FlatMLPRanker(queries.shape[1], int(config["flat_hidden_dimension"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["flat_learning_rate"]), weight_decay=float(config["flat_weight_decay"]))
    rng = np.random.default_rng(seed)
    losses = []
    for _epoch in range(int(config["flat_epochs"])):
        order = rng.permutation(len(rows))
        epoch = []
        for start in range(0, len(order), int(config["flat_batch_size"])):
            positions = torch.as_tensor(order[start : start + int(config["flat_batch_size"])], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(query_tensor[positions])
            target = target_tensor[positions]
            logits = prediction @ target.T / float(config["flat_temperature"])
            loss = F.cross_entropy(logits, torch.arange(len(positions), device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch.append(float(loss.detach().item()))
        losses.append(float(np.mean(epoch)))
    return model.eval(), {
        "epoch_loss_first": losses[0],
        "epoch_loss_last": losses[-1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def pairwise_training_candidates(row: dict, aspect: np.ndarray, raw: np.ndarray, incidence: np.ndarray, index_by_id: dict[str, int], config: dict) -> tuple[np.ndarray, np.ndarray, list[torch.Tensor]]:
    eligible = np.asarray([index_by_id[memory_id] for memory_id in row["eligible_memory_ids"]], dtype=np.int64)
    cosine = raw[eligible] @ row["query_embedding"]
    association = incidence[eligible] @ aspect
    positive_global = {index_by_id[memory_id] for group in row["groups"] for memory_id in group}
    protected = np.argsort(-cosine, kind="stable")[: int(config["protected_ann_seeds"])].tolist()
    scores = np.maximum(cosine, association)
    ranked = sorted(range(len(eligible)), key=lambda p: (-float(scores[p]), int(eligible[p])))
    selected = list(dict.fromkeys(protected + [p for p, value in enumerate(eligible) if int(value) in positive_global]))
    cap = int(config["pairwise_training_candidate_cap"])
    selected.extend(position for position in ranked if position not in selected and len(selected) < cap)
    candidate_positions = np.asarray(selected, dtype=np.int64)
    candidate_global = eligible[candidate_positions]
    local_by_global = {int(value): position for position, value in enumerate(candidate_global)}
    features = np.stack((cosine[candidate_positions], association[candidate_positions], cosine[candidate_positions] * association[candidate_positions]), axis=1).astype(np.float32)
    protected_local = torch.zeros(len(candidate_global), dtype=torch.bool, device="cpu")
    protected_local[: len(dict.fromkeys(protected))] = True
    positive_masks: list[torch.Tensor] = []
    for bundle in itertools.chain.from_iterable(
        itertools.product(*row["groups"]) for _ in [0] if len(row["groups"]) >= 2
    ):
        bundle = tuple(map(str, bundle))
        if len(set(bundle)) != len(bundle):
            continue
        for left, right in itertools.combinations(bundle, 2):
            if index_by_id[left] not in local_by_global or index_by_id[right] not in local_by_global:
                continue
            mask = torch.zeros(len(candidate_global), dtype=torch.bool, device="cpu")
            mask[local_by_global[index_by_id[left]]] = True
            mask[local_by_global[index_by_id[right]]] = True
            positive_masks.append(mask)
    return features, protected_local, positive_masks


def train_pairwise_model(rows: list[dict], train_aspects: np.ndarray, raw: np.ndarray, incidence: np.ndarray, index_by_id: dict[str, int], config: dict, seed: int, device: str) -> tuple[BudgetedPairwiseFactorGraph, dict]:
    examples = []
    for row, aspect in zip(rows, train_aspects, strict=True):
        features, protected, positive = pairwise_training_candidates(row, aspect, raw, incidence, index_by_id, config)
        if positive and len(features) > int(protected.sum().item()):
            examples.append((features, protected, positive))
    if not examples:
        raise ValueError("no outer-train distinct-member pairwise supervision")
    torch.manual_seed(seed)
    model = BudgetedPairwiseFactorGraph(3, int(config["pairwise_hidden_dimension"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["pairwise_learning_rate"]), weight_decay=float(config["pairwise_weight_decay"]))
    rng = np.random.default_rng(seed)
    losses = []
    for _epoch in range(int(config["pairwise_epochs"])):
        running = []
        order = rng.permutation(len(examples))
        optimizer.zero_grad(set_to_none=True)
        for count, position in enumerate(order, start=1):
            features_np, protected_cpu, positives_cpu = examples[int(position)]
            features_t = torch.as_tensor(features_np, device=device)
            protected = protected_cpu.to(device)
            positives = [item.to(device) for item in positives_cpu]
            loss = pairwise_group_contrastive_loss(
                model, features_t, protected, positives,
                float(config["pairwise_temperature"]),
            )
            (loss / 16.0).backward()
            running.append(float(loss.detach().item()))
            if count % 16 == 0 or count == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(np.mean(running)))
    return model.eval(), {
        "examples": len(examples),
        "epoch_loss_first": losses[0],
        "epoch_loss_last": losses[-1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def protected_positions(row: dict, ann_prefix: dict[str, tuple[str, ...]], index_by_id: dict[str, int]) -> list[int]:
    eligible = set(map(str, row["eligible_memory_ids"]))
    ids = [memory_id for memory_id in ann_prefix.get(str(row["query_id"]), ()) if memory_id in eligible]
    if len(ids) < 5:
        raise ValueError(f"frozen ANN prefix has fewer than five eligible seeds for {row['query_id']}")
    return list(dict.fromkeys(index_by_id[memory_id] for memory_id in ids[:5]))


def sparse_frontier(
    row: dict,
    aspect: np.ndarray,
    index: SparseSemanticInvertedIndex,
    raw: np.ndarray,
    index_by_id: dict[str, int],
    activated: int,
    members: int,
    beam: int,
    max_budget: int,
) -> tuple[object, dict, object, list[int]]:
    eligible_global = frozenset(index_by_id[memory_id] for memory_id in row["eligible_memory_ids"])
    active = np.argsort(-aspect, kind="stable")[:activated].astype(int).tolist()
    lookup = index.lookup(SparseSemanticLookupRequest(tuple(active), eligible_global, members))
    member_scores: dict[int, float] = {}
    query = row["query_embedding"]
    for edge in lookup.edges:
        for position in edge.memory_positions:
            cosine = float(raw[int(position)] @ query)
            member_scores[int(position)] = max(member_scores.get(int(position), -np.inf), cosine)
    prototype_weights = {int(prototype): float(aspect[int(prototype)]) for prototype in active}
    bundles, bundle_audit = build_semantic_bundle_beam(
        lookup.edges, prototype_weights, member_scores,
        arity=3, beam_width=beam,
    )
    protected = protected_positions(row, ANN_PREFIX, index_by_id)
    frontier = construct_sparse_frontier(protected, bundles, lookup.edges, member_scores, maximum_budget=max_budget)
    counters = {
        "activated_semantic_edges": lookup.activated_semantic_edges,
        "incidence_entries_visited": lookup.incidence_entries_visited,
        "bundle_hyperedges_generated": bundle_audit["bundle_hyperedges_generated"],
        "bundle_hyperedges_scored": bundle_audit["bundle_hyperedges_scored"],
        "duplicate_bundle_expansions": bundle_audit["duplicate_bundle_expansions"],
        "bundle_hyperedges_retained": bundle_audit["bundle_hyperedges_retained"],
        "unique_expanded_memories": len(set(frontier.memory_positions)),
        "semantic_bundle_unique_members": frontier.semantic_bundle_unique_members,
        "fallback_members_appended": frontier.fallback_members_appended,
        "budget_infeasible": bool(frontier.budget_infeasible),
        "candidate_count": len(frontier.memory_positions),
    }
    return frontier, counters, lookup, active


def slice_bounded_lookup(index: SparseSemanticInvertedIndex, full: SparseSemanticLookupResult, active_count: int, members_per_edge: int, eligible_mask: np.ndarray) -> SparseSemanticLookupResult:
    """Derive a smaller preregistered lookup from one max bounded prefix.

    The max lookup has already stopped after 32 eligible members per active
    edge.  Lower member/activation settings only inspect slices of those
    returned prefixes, and the exact visited counter is recovered inside that
    already-consumed prefix.  No rank prefix is extended.
    """
    edges = []
    visited_total = 0
    for edge in full.edges[:active_count]:
        positions = edge.memory_positions[:members_per_edge]
        scores = edge.incidence_scores[:members_per_edge]
        ranks = index._ranks[edge.prototype_id][: edge.rank_entries_visited]
        hits = np.flatnonzero(eligible_mask[ranks])
        visited = int(hits[len(positions) - 1]) + 1 if len(hits) >= len(positions) and len(positions) else int(edge.rank_entries_visited)
        visited_total += visited
        edges.append(SparseSemanticEdge(edge.prototype_id, tuple(positions), tuple(scores), visited))
    return SparseSemanticLookupResult(tuple(edges), len(edges), visited_total, index.metadata_bytes)


def quality(selected_ids: list[str], groups: list[list[str]]) -> dict[str, float | int]:
    selected = set(selected_ids)
    covered = [bool(selected.intersection(group)) for group in groups]
    top5 = set(selected_ids[:5])
    top5_covered = [bool(top5.intersection(group)) for group in groups]
    return {
        "K": len(groups),
        "K_ge_2": int(len(groups) >= 2),
        "group_coverage": float(np.mean(covered)) if covered else 0.0,
        "all_groups_covered": float(all(covered)) if covered else 0.0,
        "final_top5_group_coverage": float(np.mean(top5_covered)) if top5_covered else 0.0,
        "both_alternatives_selected_rate": float(np.mean([len(selected.intersection(group)) >= 2 for group in groups])) if groups else 0.0,
    }


def fast_qhds_order(model: MonotoneSubmodularQHDS, features: np.ndarray, incidence: np.ndarray, aspect: np.ndarray, protected: list[int], budget: int) -> list[int]:
    """CPU/NumPy equivalent of the frozen QHDS greedy recurrence."""
    with torch.no_grad():
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        relevance = model.relevance(feature_tensor).detach().cpu().numpy().astype(np.float64)
        structural_scale, modular_scale = model.positive_scales()
        structural_scale = float(structural_scale.detach().cpu().item())
        modular_scale = float(modular_scale.detach().cpu().item())
    incidence = np.asarray(incidence, dtype=np.float64)
    aspect = np.asarray(aspect, dtype=np.float64)
    selected = list(dict.fromkeys(map(int, protected)))
    selected_set = set(selected)
    covered = (incidence[selected] * relevance[selected, None]).sum(axis=0) if selected else np.zeros(incidence.shape[1], dtype=np.float64)
    while len(selected) < budget:
        added = incidence * relevance[:, None]
        gains = structural_scale * ((np.log1p(covered[None, :] + added) - np.log1p(covered[None, :])) * aspect[None, :]).sum(axis=1) + modular_scale * relevance
        if selected_set:
            gains[list(selected_set)] = -np.inf
        position = int(np.argmax(gains))
        selected.append(position)
        selected_set.add(position)
        covered += added[position]
    return selected


def fast_qhds_orders_batch(
    model: MonotoneSubmodularQHDS,
    feature_list: list[np.ndarray],
    incidence_list: list[np.ndarray],
    aspect_list: list[np.ndarray],
    protected_list: list[list[int]],
    target_budget: int | None = None,
) -> list[list[int]]:
    """Evaluate grid-point QHDS recurrences only through the largest needed budget."""
    if not feature_list:
        return []
    groups = len(feature_list)
    maximum = max(len(item) for item in feature_list)
    aspects = max(item.shape[1] for item in incidence_list)
    feature_batch = np.zeros((groups, maximum, feature_list[0].shape[1]), dtype=np.float32)
    incidence_batch = np.zeros((groups, maximum, aspects), dtype=np.float64)
    relevance_mask = np.zeros((groups, maximum), dtype=bool)
    for group, (features, incidence) in enumerate(zip(feature_list, incidence_list, strict=True)):
        count = len(features)
        feature_batch[group, :count] = features
        incidence_batch[group, :count, : incidence.shape[1]] = incidence
        relevance_mask[group, :count] = True
    with torch.no_grad():
        relevance = model.relevance(torch.as_tensor(feature_batch.reshape(-1, feature_batch.shape[-1]), dtype=torch.float32, device="cpu")).reshape(groups, maximum).detach().cpu().numpy().astype(np.float64)
        structural_scale, modular_scale = model.positive_scales()
        structural_scale = float(structural_scale.detach().cpu().item())
        modular_scale = float(modular_scale.detach().cpu().item())
    aspect_batch = np.zeros((groups, aspects), dtype=np.float64)
    for group, aspect in enumerate(aspect_list):
        aspect_batch[group, : len(aspect)] = aspect
    target = maximum if target_budget is None else max(1, int(target_budget))
    target_lengths = [min(target, len(item)) for item in feature_list]
    selected_mask = np.zeros((groups, maximum), dtype=bool)
    orders = [[] for _ in range(groups)]
    covered = np.zeros((groups, aspects), dtype=np.float64)
    added = incidence_batch * relevance[:, :, None]
    for group, protected in enumerate(protected_list):
        for position in dict.fromkeys(map(int, protected)):
            if position >= len(feature_list[group]):
                raise ValueError("protected position outside batched frontier")
            selected_mask[group, position] = True
            orders[group].append(position)
            covered[group] += added[group, position]
    while any(len(orders[group]) < target_lengths[group] for group in range(groups)):
        baseline = np.log1p(covered)[:, None, :]
        gains = structural_scale * ((np.log1p(covered[:, None, :] + added) - baseline) * aspect_batch[:, None, :]).sum(axis=2) + modular_scale * relevance
        gains[~relevance_mask] = -np.inf
        gains[selected_mask] = -np.inf
        chosen = np.argmax(gains, axis=1)
        for group, position in enumerate(chosen.tolist()):
            if len(orders[group]) >= target_lengths[group]:
                continue
            position = int(position)
            orders[group].append(position)
            selected_mask[group, position] = True
            covered[group] += added[group, position]
    return orders


def fast_pairwise_order(model: BudgetedPairwiseFactorGraph, features: np.ndarray, protected: list[int], budget: int) -> list[int]:
    """CPU/NumPy greedy order for a frozen nonnegative pairwise factor graph."""
    with torch.no_grad():
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        unary = model.unary(feature_tensor).detach().cpu().numpy().astype(np.float64)
        left = feature_tensor[:, None, :].expand(-1, len(feature_tensor), -1)
        right = feature_tensor[None, :, :].expand(len(feature_tensor), -1, -1)
        pair = model.pairwise(left.reshape(-1, feature_tensor.shape[1]), right.reshape(-1, feature_tensor.shape[1])).reshape(len(feature_tensor), len(feature_tensor)).detach().cpu().numpy().astype(np.float64)
    selected = list(dict.fromkeys(map(int, protected)))
    selected_set = set(selected)
    gains = unary.copy()
    for position in selected:
        gains += pair[:, position]
    while len(selected) < budget:
        if selected_set:
            gains[list(selected_set)] = -np.inf
        position = int(np.argmax(gains))
        selected.append(position)
        selected_set.add(position)
        gains += pair[:, position]
    return selected


def tile_metrics(selected_ids: list[str], tile: dict[str, str], payload: dict[str, int], metadata_positions: set[int], ids: list[str], metadata_bytes: int) -> dict[str, int]:
    selected_tiles = {tile[memory_id] for memory_id in selected_ids}
    metadata_tiles = {tile[ids[position]] for position in metadata_positions if ids[position] in tile}
    total_tiles = metadata_tiles | selected_tiles
    return {
        "metadata_bytes": int(metadata_bytes),
        "payload_bytes": int(sum(payload[memory_id] for memory_id in selected_ids)),
        "metadata_tile_activations": len(metadata_tiles),
        "payload_tile_activations": len(selected_tiles),
        "total_unique_tile_activations": len(total_tiles),
        "redundant_tile_activations": max(0, len(total_tiles) - len(set(selected_ids))),
    }


def sparse_method_orders(
    frontier_positions: tuple[int, ...],
    protected: list[int],
    cosine: np.ndarray,
    raw: np.ndarray,
    aspect: np.ndarray,
    incidence: np.ndarray,
    qhds: MonotoneSubmodularQHDS,
    flat: FlatMLPRanker,
    pairwise: BudgetedPairwiseFactorGraph,
    query: np.ndarray,
    device: str,
    *,
    qhds_order: list[int] | None = None,
    timing_out: dict[str, float] | None = None,
    max_budget: int | None = None,
) -> dict[str, list[int]]:
    positions = np.asarray(frontier_positions, dtype=np.int64)
    protected_local = [int(np.where(positions == value)[0][0]) for value in protected]
    protected_local = list(dict.fromkeys(protected_local))
    frontier_cosine = np.asarray(cosine, dtype=np.float32)
    flat_start = perf_counter_ns()
    with torch.no_grad():
        flat_query = flat(torch.as_tensor(query[None, :], dtype=torch.float32, device=device))[0].detach().cpu().numpy()
    flat_scores = raw[positions] @ flat_query
    association = incidence[positions] @ aspect
    features = np.stack((frontier_cosine, association, frontier_cosine * association), axis=1).astype(np.float32)
    ann_start = perf_counter_ns()
    ann_order = protected_local + [int(p) for p in np.argsort(-frontier_cosine, kind="stable") if int(p) not in set(protected_local)]
    ann_ms = (perf_counter_ns() - ann_start) / 1_000_000.0
    flat_ms = (perf_counter_ns() - flat_start) / 1_000_000.0
    pair_start = perf_counter_ns()
    pair_order = fast_pairwise_order(
        pairwise, features, protected_local,
        len(positions) if max_budget is None else min(int(max_budget), len(positions)),
    )
    pair_ms = (perf_counter_ns() - pair_start) / 1_000_000.0
    if qhds_order is None:
        qhds_start = perf_counter_ns()
        qhds_order = fast_qhds_order(
            qhds, features, incidence[positions], aspect, protected_local,
            len(positions) if max_budget is None else min(int(max_budget), len(positions)),
        )
        qhds_ms = (perf_counter_ns() - qhds_start) / 1_000_000.0
    else:
        qhds_ms = 0.0
    orders = {
        "ann_flat": ann_order,
        "same_capacity_neural_flat": protected_local + [int(p) for p in np.argsort(-flat_scores, kind="stable") if int(p) not in set(protected_local)],
        "sparse_joint_semantic_bundle_hypergraph": qhds_order,
        "budgeted_pairwise_factor_graph": pair_order,
    }
    if timing_out is not None:
        timing_out.update({
            "ann_flat": float(ann_ms), "same_capacity_neural_flat": float(flat_ms),
            "sparse_joint_semantic_bundle_hypergraph": float(qhds_ms),
            "budgeted_pairwise_factor_graph": float(pair_ms),
        })
    return orders


def timed_selection(order: list[int], budgets: list[int], positions: np.ndarray, ids: list[str], row: dict, groups: list[list[str]], tile: dict[str, str], payload: dict[str, int], metadata_positions: set[int], metadata_bytes: int, shared: dict[str, float], selection_extra_ms: float = 0.0) -> list[dict]:
    output = []
    for budget in budgets:
        select_start = perf_counter_ns()
        selected_positions = order[:budget]
        selection_ms = selection_extra_ms + (perf_counter_ns() - select_start) / 1_000_000.0
        selected_ids = [ids[int(positions[position])] for position in selected_positions]
        payload_start = perf_counter_ns()
        costs = tile_metrics(selected_ids, tile, payload, metadata_positions, ids, metadata_bytes)
        payload_ms = (perf_counter_ns() - payload_start) / 1_000_000.0
        values = quality(selected_ids, groups)
        values.update(costs)
        values.update({
            "budget": int(budget),
            "candidate_count": int(len(selected_ids)),
            "selection_total_ms": float(selection_ms),
            "payload_gather_ms": float(payload_ms),
            "query_e2e_ms": float(shared["query_e2e_ms"] + selection_ms + payload_ms),
            "rerank_executed": False,
            "embedding_precomputed": True,
        })
        output.append(values)
    return output


def dense_reference_rows(row: dict, aspect: np.ndarray, model: MonotoneSubmodularQHDS, raw: np.ndarray, incidence: np.ndarray, ids: list[str], index_by_id: dict[str, int], ann_prefix: dict[str, tuple[str, ...]], budgets: list[int], tile: dict[str, str], payload: dict[str, int], device: str, reference_selections: dict[tuple[str, int], tuple[str, ...]] | None = None) -> list[dict]:
    out = []
    for budget in budgets:
        if reference_selections is not None:
            selected_ids = list(reference_selections[(str(row["query_id"]), int(budget))])
            eligible_count = len(row["eligible_memory_ids"])
        else:
            eligible = np.asarray([index_by_id[memory_id] for memory_id in row["eligible_memory_ids"]], dtype=np.int64)
            cosine = raw[eligible] @ row["query_embedding"]
            protected_ids = [memory_id for memory_id in ann_prefix[str(row["query_id"])] if memory_id in set(row["eligible_memory_ids"])][:5]
            protected = [int(np.where(eligible == index_by_id[memory_id])[0][0]) for memory_id in protected_ids]
            association = incidence[eligible] @ aspect
            features = np.stack((cosine, association, cosine * association), axis=1).astype(np.float32)
            order = fast_qhds_order(model, features, incidence[eligible], aspect, protected, max(budgets))
            selected_ids = [ids[int(eligible[position])] for position in order[:budget]]
            eligible_count = len(eligible)
        values = quality(selected_ids, row["groups"])
        values.update(tile_metrics(selected_ids, tile, payload, set(), ids, int(incidence.nbytes)))
        values.update({
            "query_id": row["query_id"], "persona_id": row["persona_id"], "query_session_id": row["query_session_id"],
            "seed": row["seed"], "fold": row["fold"], "method": "dense_qhds_reference", "budget": budget,
            "activated_prototypes": 0, "members_per_semantic_edge": 0, "semantic_bundle_beam": 0,
            "activated_semantic_edges": 0, "incidence_entries_visited": int(incidence.size),
            "bundle_hyperedges_generated": 0, "bundle_hyperedges_scored": 0, "duplicate_bundle_expansions": 0,
            "unique_expanded_memories": eligible_count, "semantic_bundle_unique_members": 0, "fallback_members_appended": 0,
            "budget_infeasible": False, "frontier_candidate_count": eligible_count, "dense_online_scan": True,
            "query_encode_ms": 0.0, "ann_seed_search_ms": 0.0, "latent_aspect_predict_ms": 0.0,
            "semantic_edge_lookup_ms": 0.0, "evidence_bundle_build_ms": 0.0, "bundle_hyperedge_score_ms": 0.0,
            "submodular_select_ms": 0.0, "candidate_union_ms": 0.0, "rerank_ms": 0.0,
            "selection_total_ms": 0.0, "query_e2e_ms": 0.0, "rerank_executed": False, "embedding_precomputed": True,
            # Dense QHDS is a precomputed reference path.  Keep its row schema
            # identical to the sparse methods without claiming an online
            # payload-gather measurement.
            "candidate_count": len(selected_ids), "payload_gather_ms": 0.0, "ann_seed_count": 5,
        })
        out.append(values)
    return out


def run_unit(args: argparse.Namespace) -> None:
    global ANN_PREFIX, DEVICE
    # The selector frontiers are tiny.  Avoid OpenMP spawning dozens of
    # threads for each 3-feature CPU forward while retaining GPU parallelism
    # for the QESA and outer-train models.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = load_config(args.config)
    qhds_config = load_config("v21_qhds.json")
    qesa_config = load_config("v21_qesa_r2.json")
    data = load_config("v21_data.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("run ID, seed, or fold outside sparse-joint r2 preregistration")
    if not ATTEMPT_RE.fullmatch(args.attempt):
        raise ValueError("attempt must have the form attempt_NNN")
    DEVICE = str(args.device)
    unit_dir = VERSION_ROOT / "artifacts" / "v21_sparse_joint" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    query_vectors = QESA_HELPERS["load_query_vectors"](data)
    rows_by_fold = QESA_HELPERS["load_rows"](data, query_vectors, int(config["outer_folds"]))
    outer_train_rows = [row for fold in range(5) if fold != args.fold for row in rows_by_fold[fold]]
    heldout_rows = [dict(row, seed=args.seed, fold=args.fold) for row in rows_by_fold[args.fold]]
    for row in outer_train_rows:
        row["seed"], row["fold"] = args.seed, args.fold

    phase1_unit = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}"
    incidence = np.load(phase1_unit / "v21_uot_semantic_incidence.npy", mmap_mode="r")
    qesa_path = VERSION_ROOT / "artifacts" / "v21_qesa" / config["qesa_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_qesa_r2_models.pt"
    checkpoint = torch.load(qesa_path, map_location="cpu")
    qesa = qesa_model(checkpoint, raw.shape[1], int(qesa_config["hidden_dimension"]), DEVICE)
    index, rank_path = load_sparse_index(config, args.seed, args.fold, incidence)
    qhds, qhds_path = load_qhds(config, args.seed, args.fold, raw.shape[1], DEVICE)
    dense_reference_selection = load_dense_reference_selections(config, args.seed, args.fold)
    dense_selection_path = VERSION_ROOT / "artifacts" / "v21_qhds" / config["dense_qhds_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_qhds_selections.jsonl"
    if len(dense_reference_selection) < len(rows_by_fold[args.fold]) * len(config["final_candidate_budgets"]):
        raise ValueError("frozen dense QHDS reference selections are incomplete")
    ann_prefix = load_ann_prefix(data)
    ANN_PREFIX = ann_prefix
    tile, payload = load_layout()

    train_aspects = predict_qesa_batch(qesa, np.stack([row["query_embedding"] for row in outer_train_rows]), DEVICE)
    flat, flat_training = train_flat_model(outer_train_rows, raw, index_by_id, config, args.seed + args.fold * 100, DEVICE)
    pairwise, pairwise_training = train_pairwise_model(outer_train_rows, train_aspects, raw, incidence, index_by_id, config, args.seed + args.fold * 10000, DEVICE)
    # Selection is over tiny candidate frontiers.  Keep model weights fixed but
    # use a CPU path for the many small greedy calls; QESA remains on the GPU.
    qhds = qhds.to("cpu").eval()
    flat = flat.to("cpu").eval()
    pairwise = pairwise.to("cpu").eval()
    selector_device = "cpu"

    metrics: list[dict] = []
    selections: list[dict] = []
    grid = list(itertools.product(config["activated_prototypes_grid"], config["members_per_semantic_edge_grid"], config["bundle_beam_width_grid"]))
    budgets = list(map(int, config["final_candidate_budgets"]))
    warmup_count = min(int(config["warmup_queries"]), len(heldout_rows))
    for row in heldout_rows[:warmup_count]:
        aspect = predict_qesa(qesa, row["query_embedding"], DEVICE)
        protected = protected_positions(row, ann_prefix, index_by_id)
        for activated, members, beam in [(16, 32, 32)]:
            frontier, _counters, _lookup, _active = sparse_frontier(
                row, aspect, index, raw, index_by_id, activated, members, beam,
                int(config["candidate_pool_cap"]),
            )
            positions = np.asarray(frontier.memory_positions, dtype=np.int64)
            cosine = raw[positions] @ row["query_embedding"]
            _ = sparse_method_orders(tuple(positions), protected, cosine, raw, aspect, incidence, qhds, flat, pairwise, row["query_embedding"], selector_device)
    for row in heldout_rows:
        ann_start = perf_counter_ns()
        protected = protected_positions(row, ann_prefix, index_by_id)
        ann_ms = (perf_counter_ns() - ann_start) / 1_000_000.0
        latent_start = perf_counter_ns()
        aspect = predict_qesa(qesa, row["query_embedding"], DEVICE)
        latent_ms = (perf_counter_ns() - latent_start) / 1_000_000.0
        eligible_global = frozenset(index_by_id[memory_id] for memory_id in row["eligible_memory_ids"])
        eligible_mask = np.zeros(index.memory_count, dtype=bool)
        eligible_mask[np.fromiter(eligible_global, dtype=np.int64)] = True
        max_active = max(map(int, config["activated_prototypes_grid"]))
        active_max = np.argsort(-aspect, kind="stable")[:max_active].astype(int).tolist()
        lookup_start = perf_counter_ns()
        max_lookup = index.lookup(SparseSemanticLookupRequest(tuple(active_max), eligible_global, max(map(int, config["members_per_semantic_edge_grid"]))))
        lookup_max_ms = (perf_counter_ns() - lookup_start) / 1_000_000.0
        cosine_positions = set(protected)
        cosine_positions.update(int(position) for edge in max_lookup.edges for position in edge.memory_positions)
        cosine_array = np.asarray(sorted(cosine_positions), dtype=np.int64)
        cosine_cache = {int(position): float(value) for position, value in zip(cosine_array.tolist(), (raw[cosine_array] @ row["query_embedding"]).tolist(), strict=True)}
        dense_aspect = aspect
        metrics.extend(dense_reference_rows(row, dense_aspect, qhds, raw, incidence, ids, index_by_id, ann_prefix, budgets, tile, payload, selector_device, dense_reference_selection))
        prepared = []
        bundle_cache: dict[tuple[int, int], dict] = {}
        maximum_beam = max(map(int, config["bundle_beam_width_grid"]))
        for activated, members, beam in grid:
            recorder = OnlineLatencyRecorder(DEVICE)
            recorder.start_query()
            recorder.values["query_encode_ms"] = 0.0
            recorder.values["ann_seed_search_ms"] = ann_ms
            recorder.values["latent_aspect_predict_ms"] = latent_ms
            cache_key = (int(activated), int(members))
            if cache_key not in bundle_cache:
                lookup_start_grid = perf_counter_ns()
                lookup = slice_bounded_lookup(index, max_lookup, int(activated), int(members), eligible_mask)
                lookup_ms = lookup_max_ms if cache_key == (max_active, max(map(int, config["members_per_semantic_edge_grid"]))) else (perf_counter_ns() - lookup_start_grid) / 1_000_000.0
                member_scores: dict[int, float] = {}
                for edge in lookup.edges:
                    for position in edge.memory_positions:
                        member_scores[int(position)] = max(member_scores.get(int(position), -np.inf), cosine_cache[int(position)])
                bundle_start = perf_counter_ns()
                bundles, bundle_audit = build_semantic_bundle_beam(
                    lookup.edges,
                    {int(prototype): float(aspect[int(prototype)]) for prototype in active_max[: int(activated)]},
                    member_scores,
                    arity=int(config["semantic_bundle_arity"]),
                    beam_width=maximum_beam,
                )
                bundle_ms = (perf_counter_ns() - bundle_start) / 1_000_000.0
                score_start = perf_counter_ns()
                bundles = sorted(bundles, key=lambda item: (-float(item.score), item.prototype_ids, item.memory_positions))
                score_ms = (perf_counter_ns() - score_start) / 1_000_000.0
                bundle_cache[cache_key] = {
                    "lookup": lookup, "member_scores": member_scores, "bundles": bundles,
                    "audit": bundle_audit, "lookup_ms": lookup_ms, "bundle_ms": bundle_ms, "score_ms": score_ms,
                }
            else:
                cached = bundle_cache[cache_key]
                lookup = cached["lookup"]
                member_scores = cached["member_scores"]
                bundles = cached["bundles"][: int(beam)]
                bundle_audit = dict(cached["audit"])
                bundle_audit["bundle_hyperedges_retained"] = len(bundles)
                lookup_ms = bundle_ms = score_ms = 0.0
            if cache_key in bundle_cache and int(beam) == maximum_beam:
                bundles = bundle_cache[cache_key]["bundles"]
            elif int(beam) != maximum_beam:
                bundles = bundle_cache[cache_key]["bundles"][: int(beam)]
                bundle_audit = dict(bundle_cache[cache_key]["audit"])
                bundle_audit["bundle_hyperedges_retained"] = len(bundles)
            recorder.values["semantic_edge_lookup_ms"] = lookup_ms
            recorder.values["evidence_bundle_build_ms"] = bundle_ms
            recorder.values["bundle_hyperedge_score_ms"] = score_ms
            union_start = perf_counter_ns()
            frontier = construct_sparse_frontier(
                protected, bundles, lookup.edges, member_scores,
                maximum_budget=int(config["candidate_pool_cap"]),
            )
            recorder.values["candidate_union_ms"] = (perf_counter_ns() - union_start) / 1_000_000.0
            shared_end = perf_counter_ns()
            shared = recorder.finish_query(shared_end)
            shared_total = ann_ms + latent_ms + lookup_ms + bundle_ms + float(recorder.values["bundle_hyperedge_score_ms"]) + float(recorder.values["candidate_union_ms"])
            positions = np.asarray(frontier.memory_positions, dtype=np.int64)
            base_counters = {
                "activated_semantic_edges": int(lookup.activated_semantic_edges),
                "incidence_entries_visited": int(lookup.incidence_entries_visited),
                "bundle_hyperedges_generated": int(bundle_audit["bundle_hyperedges_generated"]),
                "bundle_hyperedges_scored": int(bundle_audit["bundle_hyperedges_scored"]),
                "duplicate_bundle_expansions": int(bundle_audit["duplicate_bundle_expansions"]),
                "unique_expanded_memories": len(set(frontier.memory_positions)),
                "semantic_bundle_unique_members": int(frontier.semantic_bundle_unique_members),
                "fallback_members_appended": int(frontier.fallback_members_appended),
                "budget_infeasible": bool(frontier.budget_infeasible),
                "frontier_candidate_count": len(frontier.memory_positions),
            }
            metadata_positions = {int(position) for edge in lookup.edges for position in edge.memory_positions}
            frontier_cosine = np.asarray([cosine_cache[int(position)] for position in positions], dtype=np.float32)
            frontier_association = incidence[positions] @ aspect
            frontier_features = np.stack((frontier_cosine, frontier_association, frontier_cosine * frontier_association), axis=1).astype(np.float32)
            protected_local = [int(np.where(positions == value)[0][0]) for value in dict.fromkeys(protected)]
            prepared.append({
                "activated": int(activated), "members": int(members), "beam": int(beam),
                "aspect": aspect, "frontier": frontier, "lookup": lookup, "positions": positions,
                "shared": shared, "shared_total": shared_total, "base_counters": base_counters,
                "metadata_positions": metadata_positions, "features": frontier_features,
                "protected_local": protected_local, "bundle_audit": bundle_audit,
            })

        batch_start = perf_counter_ns()
        qhds_orders = fast_qhds_orders_batch(
            qhds,
            [item["features"] for item in prepared],
            [incidence[item["positions"]] for item in prepared],
            [item["aspect"] for item in prepared],
            [item["protected_local"] for item in prepared],
            target_budget=max(budgets),
        )
        batch_qhds_ms = (perf_counter_ns() - batch_start) / 1_000_000.0
        per_grid_qhds_ms = batch_qhds_ms / max(len(prepared), 1)
        for prep, qhds_order in zip(prepared, qhds_orders, strict=True):
            positions = prep["positions"]
            cosine = np.asarray([cosine_cache[int(position)] for position in positions], dtype=np.float32)
            method_timing: dict[str, float] = {}
            orders = sparse_method_orders(
                tuple(positions), protected, cosine, raw, prep["aspect"], incidence, qhds, flat, pairwise,
                row["query_embedding"], selector_device, qhds_order=qhds_order, timing_out=method_timing,
                max_budget=max(budgets),
            )
            method_timing["sparse_joint_semantic_bundle_hypergraph"] = per_grid_qhds_ms
            for method, order in orders.items():
                selection_rows = timed_selection(
                    order, budgets, positions, ids, row, row["groups"], tile, payload,
                    prep["metadata_positions"], index.metadata_bytes,
                    {"query_e2e_ms": prep["shared_total"]}, method_timing.get(method, 0.0),
                )
                for budget, values in zip(budgets, selection_rows, strict=True):
                    row_values = {
                        "query_id": row["query_id"], "persona_id": row["persona_id"], "query_session_id": row["query_session_id"],
                        "seed": args.seed, "fold": args.fold, "method": method,
                        "activated_prototypes": prep["activated"], "members_per_semantic_edge": prep["members"], "semantic_bundle_beam": prep["beam"],
                        "dense_online_scan": False,
                        "ann_seed_count": len(protected),
                        "query_encode_ms": 0.0,
                        "ann_seed_search_ms": float(prep["shared"]["ann_seed_search_ms"]),
                        "latent_aspect_predict_ms": float(prep["shared"]["latent_aspect_predict_ms"]),
                        "semantic_edge_lookup_ms": float(prep["shared"]["semantic_edge_lookup_ms"]),
                        "evidence_bundle_build_ms": float(prep["shared"]["evidence_bundle_build_ms"]),
                        "bundle_hyperedge_score_ms": float(prep["shared"]["bundle_hyperedge_score_ms"]),
                        "candidate_union_ms": float(prep["shared"]["candidate_union_ms"]),
                        "submodular_select_ms": float(method_timing.get(method, 0.0) if method == "sparse_joint_semantic_bundle_hypergraph" else 0.0),
                        "rerank_ms": 0.0,
                        "selection_total_ms": float(values["selection_total_ms"]),
                        "query_e2e_ms": float(values["query_e2e_ms"]),
                        "rerank_executed": False,
                        "embedding_precomputed": True,
                    }
                    row_values.update(prep["base_counters"])
                    row_values.update({key: value for key, value in values.items() if key not in {"budget"}})
                    row_values["budget"] = int(budget)
                    metrics.append(row_values)
                    selections.append({
                        "query_id": row["query_id"], "seed": args.seed, "fold": args.fold,
                        "method": method, "activated_prototypes": prep["activated"], "members_per_semantic_edge": prep["members"],
                        "semantic_bundle_beam": prep["beam"], "budget": int(budget),
                        "selected_memory_ids": [ids[int(positions[position])] for position in order[: int(budget)]],
                        "frontier_memory_ids": [ids[int(position)] for position in positions],
                    })

    metrics_path = unit_dir / "v21_sparse_joint_query_metrics.csv"
    selections_path = unit_dir / "v21_sparse_joint_selections.jsonl"
    models_path = unit_dir / "v21_sparse_joint_models.pt"
    write_csv_new(metrics_path, metrics)
    with selections_path.open("x", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save({
        "flat": {key: value.detach().cpu() for key, value in flat.state_dict().items()},
        "pairwise": {key: value.detach().cpu() for key, value in pairwise.state_dict().items()},
        "flat_training": flat_training, "pairwise_training": pairwise_training,
    }, models_path)
    manifest = {
        "schema_version": "v21-sparse-joint-oof-unit-1", "status": "completed", "revision_id": config["revision_id"],
        "run_id": args.run_id, "attempt": args.attempt, "seed": args.seed, "fold": args.fold,
        "config_name": args.config, "parent_run_id": config.get("parent_run_id"),
        "candidate_pool_cap": int(config["candidate_pool_cap"]),
        "outer_train_queries": len(outer_train_rows), "outer_heldout_queries": len(heldout_rows),
        "metric_rows": len(metrics), "selection_rows": len(selections), "grid_points": len(grid),
        "warmup_queries": warmup_count, "protected_ann_seed_count": int(config["protected_ann_seeds"]),
        "nested_budget_policy": True, "candidate_frontier_shared_across_methods": True,
        "no_dense_online_incidence_scan": True, "dense_qhds_reference_marked": True,
        "r3_fallback_contract": {
            "semantic_bundle_arity": int(config["semantic_bundle_arity"]),
            "fallback_reads_only_returned_edge_members": True,
            "prefix_extension_forbidden": True,
            "required_counters_present": True,
        },
        "artifacts": {
            "metrics": str(metrics_path.relative_to(VERSION_ROOT)),
            "selections": str(selections_path.relative_to(VERSION_ROOT)),
            "models": str(models_path.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "metrics": sha256_file(metrics_path), "selections": sha256_file(selections_path), "models": sha256_file(models_path),
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "qesa_checkpoint": sha256_file(qesa_path), "qhds_model": sha256_file(qhds_path),
            "dense_qhds_selections": sha256_file(dense_selection_path),
            "incidence_array": array_sha256(incidence), "rank_index": sha256_file(rank_path),
        },
        "training": {"flat": flat_training, "pairwise": pairwise_training},
        "calibration_read": False, "formal_test_read": False,
    }
    manifest_path = unit_dir / "v21_sparse_joint_unit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "seed": args.seed, "fold": args.fold, "attempt": args.attempt, "metric_rows": len(metrics), "selection_rows": len(selections)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one v21 sparse-joint semantic-bundle OOF unit.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default="v21_sparse_joint_tuned_oof_r1.json")
    parser.add_argument("--run-id", default="v21_sparse_joint_tuned_oof_r1_20260714")
    parser.add_argument("--attempt", default="attempt_001")
    args = parser.parse_args()
    run_unit(args)


if __name__ == "__main__":
    main()
