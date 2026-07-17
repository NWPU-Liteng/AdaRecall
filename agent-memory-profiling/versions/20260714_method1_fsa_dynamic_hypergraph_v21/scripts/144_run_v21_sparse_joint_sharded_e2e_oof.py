#!/usr/bin/env python3
"""Run one tuned-r2 unit with exact persona-sharded lookup and reused models."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json  # noqa: E402
from v21.persona_sharded_index import PersonaDispatchSparseSemanticIndex, PersonaShardedSparseSemanticIndex  # noqa: E402


WORKER = runpy.run_path(str(VERSION_ROOT / "scripts/128_train_v21_sparse_joint_tuned_oof_r2.py"), run_name="v21_sharded_e2e_worker")
ORIGINAL_LOAD_INDEX = WORKER["load_sparse_index"]
CURRENT_ARGS: argparse.Namespace | None = None


def source_model_path(config: dict) -> Path:
    if CURRENT_ARGS is None:
        raise RuntimeError("worker arguments are not initialized")
    return (
        VERSION_ROOT / "artifacts/v21_sparse_joint" / config["source_model_run_id"]
        / f"seed_{CURRENT_ARGS.seed}" / f"fold_{CURRENT_ARGS.fold}" / "attempts" / config["source_model_attempt"]
        / "v21_sparse_joint_models.pt"
    )


def load_sharded_index(config: dict, seed: int, fold: int, incidence):
    _global, rank_path = ORIGINAL_LOAD_INDEX(config, seed, fold, incidence)
    data = load_config("v21_data.json")
    ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    memory = {str(row["memory_id"]): str(row["persona_id"]) for row in iter_jsonl(data["memory"]["items"])}
    personas = tuple(memory[memory_id] for memory_id in ids)
    sharded = PersonaShardedSparseSemanticIndex(incidence, personas)
    return PersonaDispatchSparseSemanticIndex(sharded, personas), rank_path


def reuse_flat(rows, raw, index_by_id, config: dict, seed: int, device: str):
    checkpoint = torch.load(source_model_path(config), map_location="cpu")
    model = WORKER["FlatMLPRanker"](raw.shape[1], int(config["flat_hidden_dimension"])).to(device)
    model.load_state_dict(checkpoint["flat"])
    return model.eval(), {**checkpoint["flat_training"], "reused_from_parent": True}


def reuse_pairwise(rows, predictions, raw, incidence, index_by_id, config: dict, seed: int, device: str):
    checkpoint = torch.load(source_model_path(config), map_location="cpu")
    model = WORKER["BudgetedPairwiseFactorGraph"](3, int(config["pairwise_hidden_dimension"])).to(device)
    model.load_state_dict(checkpoint["pairwise"])
    return model.eval(), {**checkpoint["pairwise_training"], "reused_from_parent": True}


def main() -> None:
    global CURRENT_ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_sparse_joint_sharded_e2e_oof_r1.json")
    parser.add_argument("--run-id", default="v21_sparse_joint_sharded_e2e_oof_r1_20260715")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--device", default="cuda:0")
    CURRENT_ARGS = parser.parse_args()
    config = load_config(CURRENT_ARGS.config)
    if config.get("minimum_effect_threshold") is not None or config.get("index_policy") not in {"persona_by_prototype_rank_shards_exact_equivalent", "global_prototype_rank_control"}:
        raise ValueError("invalid sharded e2e config")
    if config["index_policy"] == "persona_by_prototype_rank_shards_exact_equivalent":
        WORKER["load_sparse_index"] = load_sharded_index
    WORKER["train_flat_model"] = reuse_flat
    WORKER["train_pairwise_model"] = reuse_pairwise
    WORKER["run_unit"](CURRENT_ARGS)


if __name__ == "__main__":
    main()
