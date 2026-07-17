#!/usr/bin/env python3
"""Run one active-prototype expansion unit while reusing cap64 outer models."""
from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v21.io import load_config  # noqa: E402

WORKER = runpy.run_path(str(ROOT / "scripts/128_train_v21_sparse_joint_tuned_oof_r2.py"), run_name="v21_active24_worker")
CURRENT = None

def source_path(config: dict) -> Path:
    return ROOT / "artifacts/v21_sparse_joint" / config["source_model_run_id"] / f"seed_{CURRENT.seed}" / f"fold_{CURRENT.fold}" / "attempts" / config["source_model_attempt"] / "v21_sparse_joint_models.pt"

def reuse_flat(rows, raw, index_by_id, config, seed, device):
    checkpoint = torch.load(source_path(config), map_location="cpu")
    model = WORKER["FlatMLPRanker"](raw.shape[1], int(config["flat_hidden_dimension"])).to(device)
    model.load_state_dict(checkpoint["flat"])
    return model.eval(), {**checkpoint["flat_training"], "reused_from": config["source_model_run_id"]}

def reuse_pairwise(rows, predictions, raw, incidence, index_by_id, config, seed, device):
    checkpoint = torch.load(source_path(config), map_location="cpu")
    model = WORKER["BudgetedPairwiseFactorGraph"](3, int(config["pairwise_hidden_dimension"])).to(device)
    model.load_state_dict(checkpoint["pairwise"])
    return model.eval(), {**checkpoint["pairwise_training"], "reused_from": config["source_model_run_id"]}

def main():
    global CURRENT
    p=argparse.ArgumentParser(); p.add_argument('--config',default='v21_sparse_joint_cap64_active24_oof_r1.json'); p.add_argument('--run-id',required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--fold',type=int,required=True); p.add_argument('--attempt',default='attempt_001'); p.add_argument('--device',default='cuda:0'); CURRENT=p.parse_args()
    config=load_config(CURRENT.config)
    if config['run_id'] != CURRENT.run_id or config['index_run_id'] != 'v21_sparse_semantic_index_r1_20260714': raise ValueError('invalid active24 config')
    WORKER['train_flat_model']=reuse_flat; WORKER['train_pairwise_model']=reuse_pairwise
    WORKER['run_unit'](CURRENT)
if __name__=='__main__': main()
