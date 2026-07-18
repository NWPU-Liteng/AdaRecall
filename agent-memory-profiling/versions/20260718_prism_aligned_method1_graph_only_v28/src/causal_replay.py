from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import CausalGraphSnapshot, GraphSelectionResult
from .coretrieval_hypergraph import HistoryPropagation, propagate_coretrieval
from .dual_view_propagation import graph_node_scores
from .graph_only_selector import ObjectiveWeights, select_graph_only
from .query_edge_alignment import EdgeActivation, align_query_to_edges
from .semantic_propagation import semantic_distribution


@dataclass(frozen=True)
class QueryAudit:
    activation: EdgeActivation
    history: HistoryPropagation
    graph_scores: np.ndarray
    x0: np.ndarray
    x_sem: np.ndarray
    x_hist: np.ndarray
    agreement: np.ndarray
    recency: np.ndarray


def run_graph_only_query(
    snapshot: CausalGraphSnapshot,
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    *,
    config: Mapping[str, Any],
    history_events: Iterable[Mapping[str, Any]] = (),
    excluded_query_ids: set[str] | None = None,
    query_embeddings_by_id: Mapping[str, np.ndarray] | None = None,
) -> tuple[GraphSelectionResult, QueryAudit]:
    semantic_cfg = config["semantic"]
    history_cfg = config["history"]
    selector_cfg = config["selector"]
    activation = align_query_to_edges(
        snapshot,
        query_embedding,
        all_embeddings,
        centroid_weight=float(semantic_cfg["alignment_centroid_weight"]),
        medoid_weight=float(semantic_cfg["alignment_medoid_weight"]),
        temperature=float(semantic_cfg["alignment_temperature"]),
        active_high=int(semantic_cfg["active_high_confidence"]),
        active_low=int(semantic_cfg["active_low_confidence"]),
        confidence_entropy_max=float(semantic_cfg["confidence_entropy_max"]),
        confidence_margin_min=float(semantic_cfg["confidence_margin_min"]),
    )
    x0, x_sem, pool = semantic_distribution(
        snapshot,
        activation,
        restart=float(semantic_cfg["semantic_restart"]),
        steps=int(semantic_cfg["semantic_steps"]),
        pool_cap=int(config["budget"]["expanded_pool_cap"]),
    )
    history = propagate_coretrieval(
        snapshot,
        x_sem,
        query_embedding,
        history_events,
        excluded_query_ids=excluded_query_ids or set(),
        query_embeddings_by_id=query_embeddings_by_id,
        minimum_support=int(history_cfg["minimum_support"]),
        maximum_edge_size=int(history_cfg["maximum_edge_size"]),
        restart=float(history_cfg["history_restart"]),
    )
    graph_scores, agreement, recency = graph_node_scores(
        x_sem,
        history.scores,
        snapshot.arrival_orders,
        semantic_weight=float(history_cfg["semantic_weight"]),
        history_weight=float(history_cfg["history_weight"]),
        agreement_weight=float(history_cfg["agreement_weight"]),
        recency_weight=float(history_cfg["recency_weight"]),
    )
    result = select_graph_only(
        snapshot,
        pool,
        graph_scores,
        x0,
        x_sem,
        history.scores,
        agreement,
        recency,
        activation.active_columns,
        activation.weights,
        history_edge_ids_by_local=history.active_edge_ids_by_local,
        budget=int(config["budget"]["frontier_c"]),
        weights=ObjectiveWeights(
            relevance=float(selector_cfg["relevance_lambda"]),
            edge_coverage=float(selector_cfg["edge_coverage_lambda"]),
            facility=float(selector_cfg["facility_lambda"]),
        ),
    )
    result.audit.update(
        {
            "active_coarse_or_child_edges": len(activation.active_columns),
            "activation_entropy": activation.entropy,
            "activation_margin": activation.margin,
            "query_to_centroid_comparisons": activation.centroid_comparisons,
            "query_to_medoid_comparisons": activation.medoid_comparisons,
            "history_usable_events": history.usable_events,
            "fold_excluded_history_events": history.fold_excluded_events,
            "future_excluded_history_events": history.future_excluded_events,
        }
    )
    return result, QueryAudit(activation, history, graph_scores, x0, x_sem, history.scores, agreement, recency)
