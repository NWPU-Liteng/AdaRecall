from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT.parent
V28 = VERSIONS / "20260718_prism_aligned_method1_graph_only_v28"
V24 = VERSIONS / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V28))
sys.path.insert(0, str(V24 / "src"))

from src.causal_replay import run_graph_only_query  # noqa: E402
from src.semantic_propagation import semantic_distribution  # noqa: E402
from src.streaming_fuzzy_hierarchy import build_causal_snapshot  # noqa: E402
from v24.dataset import load_canonical  # noqa: E402

FRONTIER = V28 / "runs" / "v28_frontier_oof_20260719_attempt003_913cd3f" / "per_query_frontier.jsonl"
QWEN = V28 / "runs" / "v28_qwen_oof_20260719_fullranking" / "per_query_qwen.jsonl"
CONFIG = V28 / "configs" / "graph_only_main.yaml"
OUTPUT = ROOT / "artifacts" / "v28_mechanism_diagnosis.json"

EXPECTED = {
    "frontier": "834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98",
    "qwen": "439d32655eed0d6e136a181c9b66d373ad54e13937fbe16fe8131eea9e559696",
    "config": "d6252a2e669cb07226393d2bd6b90888f8973af8038dfaa4281b2b9174fb5e8b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numeric_summary(rows: Sequence[Mapping], fields: Sequence[str]) -> dict:
    output = {"count": len(rows)}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows if row.get(field) is not None], dtype=np.float64)
        output[field] = None if not len(values) else {
            "mean": float(values.mean()),
            "p25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "p75": float(np.quantile(values, 0.75)),
        }
    return output


def qwen_stage_summary(
    frontier: Mapping[tuple[str, str], dict],
    qwen: Mapping[tuple[str, str], dict],
    evaluations: Mapping[str, object],
    method: str,
) -> dict:
    dense_method = "dense_hnsw_32"
    counters = Counter()
    by_fold = defaultdict(Counter)
    by_type = defaultdict(Counter)
    by_namespace = defaultdict(Counter)
    for query_id in sorted(evaluations):
        graph_ids = set(frontier[(query_id, method)]["frontier_ids"])
        dense_ids = set(frontier[(query_id, dense_method)]["frontier_ids"])
        gold = set(evaluations[query_id].gold_memory_ids)
        graph_top5 = set(qwen[(query_id, method)]["final_top5"])
        dense_top5 = set(qwen[(query_id, dense_method)]["final_top5"])
        values = {
            "graph_frontier_gold": int(bool(graph_ids & gold)),
            "dense_frontier_gold": int(bool(dense_ids & gold)),
            "graph_novel_gold": int(bool((graph_ids - dense_ids) & gold)),
            "dense_only_gold": int(bool((dense_ids - graph_ids) & gold)),
            "graph_top5_gold": int(bool(graph_top5 & gold)),
            "dense_top5_gold": int(bool(dense_top5 & gold)),
            "graph_novel_gold_top5": int(bool((graph_ids - dense_ids) & graph_top5 & gold)),
        }
        counters.update(values)
        row = frontier[(query_id, method)]
        by_fold[str(row["fold"])].update(values)
        by_type[str(row["query_type"])].update(values)
        by_namespace[str(row["namespace_id"])].update(values)
    return {
        "overall_query_counts": dict(counters),
        "by_fold": {key: dict(value) for key, value in sorted(by_fold.items())},
        "by_query_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "by_namespace": {key: dict(value) for key, value in sorted(by_namespace.items(), key=lambda row: int(row[0]))},
    }


def candidate_features(
    provenance: Mapping,
    *,
    snapshot,
    memory_position: int,
    memory_content: str,
    qwen_rank: int,
    qwen_score: float,
    role: str,
) -> dict:
    edge_by_id = {edge.edge_id: (index, edge) for index, edge in enumerate(snapshot.edges)}
    edge_rows = [edge_by_id[edge_id] for edge_id in provenance["semantic_edge_ids"]]
    memberships = [float(snapshot.incidence[memory_position, index]) for index, _ in edge_rows]
    degrees = [edge.degree for _, edge in edge_rows]
    cohesions = [edge.cohesion for _, edge in edge_rows]
    return {
        "role": role,
        "selection_order": int(provenance["selection_order"]),
        "x_sem": float(provenance["x_sem"]),
        "agreement": float(provenance["agreement"]),
        "relevance_marginal": float(provenance["relevance_marginal"]),
        "edge_coverage_marginal": float(provenance["edge_coverage_marginal"]),
        "facility_marginal": float(provenance["facility_marginal"]),
        "total_marginal": float(
            provenance["relevance_marginal"]
            + provenance["edge_coverage_marginal"]
            + provenance["facility_marginal"]
        ),
        "active_membership_max": max(memberships),
        "active_membership_mean": float(np.mean(memberships)),
        "active_edge_degree_min": min(degrees),
        "active_edge_degree_mean": float(np.mean(degrees)),
        "active_edge_cohesion_max": max(cohesions),
        "active_edge_cohesion_mean": float(np.mean(cohesions)),
        "active_edge_count": len(edge_rows),
        "document_characters": len(memory_content),
        "document_whitespace_tokens": len(memory_content.split()),
        "qwen_rank": qwen_rank,
        "qwen_score": qwen_score,
        "coverage_and_facility_zero": int(
            float(provenance["edge_coverage_marginal"]) == 0.0
            and float(provenance["facility_marginal"]) == 0.0
        ),
    }


def main() -> None:
    observed = {"frontier": sha256(FRONTIER), "qwen": sha256(QWEN), "config": sha256(CONFIG)}
    if observed != EXPECTED:
        raise RuntimeError(f"immutable v28 lineage mismatch: {observed}")

    frontier_rows = read_jsonl(FRONTIER)
    qwen_rows = read_jsonl(QWEN)
    frontier = {(row["query_id"], row["method"]): row for row in frontier_rows}
    qwen = {(row["query_id"], row["method"]): row for row in qwen_rows}
    dataset = load_canonical(V24 / "inputs", split="train")
    evaluations = dataset.evaluations["train"]
    queries = dataset.queries["train"]
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    semantic_config = json.loads(json.dumps(config))
    semantic_config["history"]["history_weight"] = 0.0
    semantic_config["history"]["agreement_weight"] = 0.0

    embeddings = np.asarray(dataset.embeddings, dtype=np.float32).copy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    memory_ids = dataset.memory_ids
    namespaces = [memory.namespace_id for memory in dataset.memories]
    arrivals = np.asarray([memory.arrival_order for memory in dataset.memories], dtype=np.int64)
    contents = [memory.content for memory in dataset.memories]
    grouped = defaultdict(list)
    for query in queries:
        grouped[(query.namespace_id, int(query.cutoff_order))].append(query)

    stage_counts = Counter()
    selected_features = []
    pool_gold_features = []
    snapshot_diagnostics = []
    reproduction_mismatches = []
    for snapshot_number, ((namespace_id, cutoff), snapshot_queries) in enumerate(sorted(grouped.items()), start=1):
        semantic_cfg = config["semantic"]
        snapshot = build_causal_snapshot(
            embeddings,
            memory_ids,
            namespaces,
            arrivals,
            namespace_id=namespace_id,
            cutoff_order=cutoff,
            memberships_per_memory=int(semantic_cfg["memberships_per_memory"]),
            temperature=float(semantic_cfg["membership_temperature"]),
            medoids_per_edge=int(semantic_cfg["medoids_per_edge"]),
            max_degree_before_split=int(semantic_cfg["max_degree_before_split"]),
            split_children=int(semantic_cfg["split_children"]),
            merge_similarity=float(semantic_cfg["merge_similarity"]),
            minimum_merge_age=int(semantic_cfg["minimum_merge_age"]),
            seed=int(config["random_seed"]),
        )
        snapshot_diagnostics.append({
            "namespace_id": namespace_id,
            "cutoff_order": cutoff,
            "visible": len(snapshot.positions),
            "edge_count": len(snapshot.edges),
            "coarse_edge_count": snapshot.coarse_edge_count,
            "child_edge_count": len(snapshot.edges) - snapshot.coarse_edge_count,
            "degree_p95": snapshot.health["degree_p95"],
            "degree_max": snapshot.health["degree_max"],
            "cohesion_mean": snapshot.health["cohesion_mean"],
        })
        local_by_id = {memory_id: index for index, memory_id in enumerate(snapshot.memory_ids)}
        for query in snapshot_queries:
            query_vector = np.asarray(dataset.query_vector("train", query.query_id), dtype=np.float32).copy()
            result, audit = run_graph_only_query(snapshot, query_vector, embeddings, config=semantic_config)
            recorded = frontier[(query.query_id, "semantic_graph_only_32")]
            if result.memory_ids != recorded["frontier_ids"]:
                reproduction_mismatches.append(query.query_id)
                continue
            x0, x_sem, pool = semantic_distribution(
                snapshot,
                audit.activation,
                restart=float(semantic_cfg["semantic_restart"]),
                steps=int(semantic_cfg["semantic_steps"]),
                pool_cap=int(config["budget"]["expanded_pool_cap"]),
            )
            del x0, x_sem
            gold = set(evaluations[query.query_id].gold_memory_ids)
            pool_ids = {snapshot.memory_ids[int(position)] for position in pool}
            selected_ids = set(result.memory_ids)
            dense_ids = set(frontier[(query.query_id, "dense_hnsw_32")]["frontier_ids"])
            stage_counts["queries"] += 1
            stage_counts["active_child_queries"] += int(any(snapshot.edges[index].parent_id for index in audit.activation.active_columns))
            stage_counts["gold_in_active_pool_queries"] += int(bool(gold & pool_ids))
            stage_counts["gold_selected_queries"] += int(bool(gold & selected_ids))
            stage_counts["gold_pool_but_not_selected_queries"] += int(bool((gold & pool_ids) - selected_ids))
            stage_counts["graph_novel_gold_queries"] += int(bool(gold & (selected_ids - dense_ids)))
            qwen_row = qwen[(query.query_id, "semantic_graph_only_32")]
            rank_by_id = {memory_id: rank for rank, memory_id in enumerate(qwen_row["reranked_ids"], start=1)}
            score_by_id = dict(zip(qwen_row["reranked_ids"], qwen_row["reranker_scores"]))
            provenance_by_id = {row["memory_id"]: row for row in recorded["graph_provenance"]}
            for memory_id in result.memory_ids:
                if memory_id in gold and memory_id not in dense_ids:
                    role = "graph_novel_gold"
                elif memory_id in gold:
                    role = "shared_gold"
                else:
                    role = "selected_false_positive"
                local_position = local_by_id[memory_id]
                selected_features.append(candidate_features(
                    provenance_by_id[memory_id],
                    snapshot=snapshot,
                    memory_position=local_position,
                    memory_content=contents[int(snapshot.positions[local_position])],
                    qwen_rank=rank_by_id[memory_id],
                    qwen_score=float(score_by_id[memory_id]),
                    role=role,
                ))
            pool_order = {snapshot.memory_ids[int(position)]: rank for rank, position in enumerate(pool, start=1)}
            for memory_id in sorted((gold & pool_ids) - selected_ids):
                local_position = local_by_id[memory_id]
                columns = audit.activation.active_columns
                memberships = np.asarray(snapshot.incidence[local_position, list(columns)].todense()).reshape(-1)
                active = np.flatnonzero(memberships > 0)
                edge_rows = [snapshot.edges[columns[index]] for index in active]
                pool_gold_features.append({
                    "role": "pool_gold_not_selected",
                    "pool_rank_by_x_sem": pool_order[memory_id],
                    "x_sem": float(audit.x_sem[local_position]),
                    "graph_score": float(audit.graph_scores[local_position]),
                    "active_membership_max": float(memberships[active].max()),
                    "active_edge_degree_min": min(edge.degree for edge in edge_rows),
                    "active_edge_cohesion_max": max(edge.cohesion for edge in edge_rows),
                    "document_characters": len(contents[int(snapshot.positions[local_position])]),
                    "document_whitespace_tokens": len(contents[int(snapshot.positions[local_position])].split()),
                })
        print(
            f"diagnostic snapshot {snapshot_number}/{len(grouped)} ns={namespace_id} cutoff={cutoff}",
            flush=True,
        )

    if reproduction_mismatches:
        raise RuntimeError(f"v28 semantic reproduction mismatch for {len(reproduction_mismatches)} queries")
    feature_fields = (
        "selection_order",
        "x_sem",
        "relevance_marginal",
        "edge_coverage_marginal",
        "facility_marginal",
        "total_marginal",
        "active_membership_max",
        "active_edge_degree_min",
        "active_edge_cohesion_max",
        "active_edge_count",
        "document_characters",
        "document_whitespace_tokens",
        "qwen_rank",
        "qwen_score",
        "coverage_and_facility_zero",
    )
    by_role = {
        role: numeric_summary([row for row in selected_features if row["role"] == role], feature_fields)
        for role in ("graph_novel_gold", "shared_gold", "selected_false_positive")
    }
    output = {
        "schema_version": "method1-v30-v28-mechanism-diagnosis-1",
        "status": "POSTHOC_DIAGNOSIS_COMPLETE_NO_PROMOTION",
        "scientific_scope": "gold-aware post-hoc analysis of consumed train OOF; not promotion evidence",
        "immutable_inputs": observed,
        "query_count": len(queries),
        "snapshot_count": len(grouped),
        "semantic_reproduction_mismatches": 0,
        "stage_counts": dict(stage_counts),
        "snapshot_structure": {
            "snapshots_with_children": sum(row["child_edge_count"] > 0 for row in snapshot_diagnostics),
            "total_child_edges": sum(row["child_edge_count"] for row in snapshot_diagnostics),
            "mean_coarse_degree_p95": float(np.mean([row["degree_p95"] for row in snapshot_diagnostics])),
            "maximum_coarse_degree": max(row["degree_max"] for row in snapshot_diagnostics),
            "mean_coarse_cohesion": float(np.mean([row["cohesion_mean"] for row in snapshot_diagnostics])),
        },
        "selected_candidate_features_by_role": by_role,
        "pool_gold_not_selected": numeric_summary(
            pool_gold_features,
            (
                "pool_rank_by_x_sem",
                "x_sem",
                "graph_score",
                "active_membership_max",
                "active_edge_degree_min",
                "active_edge_cohesion_max",
                "document_characters",
                "document_whitespace_tokens",
            ),
        ),
        "qwen_stage_semantic": qwen_stage_summary(frontier, qwen, evaluations, "semantic_graph_only_32"),
        "qwen_stage_dual": qwen_stage_summary(frontier, qwen, evaluations, "dual_graph_only_32"),
        "interpretation_guard": {
            "gold_may_enter_new_algorithm": False,
            "diagnosis_may_promote_configuration": False,
            "fresh_authorized": False,
        },
    }
    write_json(OUTPUT, output)
    print(json.dumps({"status": output["status"], "output": str(OUTPUT), "stage_counts": output["stage_counts"]}, indent=2))


if __name__ == "__main__":
    main()
