#!/usr/bin/env python3
"""Produce a reproducible functional/cost probe for the agent-memory PIM path."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend import (BankPimConfig, BankPimModel, plan_attention_route,
                         validate_attention_functional)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite probe output')
    rng = np.random.default_rng(args.seed)
    query = rng.normal(size=(8, 32)).astype(np.float32)
    keys = rng.normal(size=(257, 2, 32)).astype(np.float32)
    values = rng.normal(size=(257, 2, 32)).astype(np.float32)
    functional = validate_attention_functional(query, keys, values, banks=16)
    profile_path = ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json'
    profile = json.loads(profile_path.read_text())
    backend = BankPimModel(BankPimConfig(**profile['config']))
    attention = [plan_attention_route(backend, context_tokens=context, query_heads=32,
                                      kv_heads=8, head_dim=128,
                                      resident_bytes=2 * context * 8 * 128 * 2)
                 for context in (256, 1024, 4096, 16384)]
    cold_attention = [plan_attention_route(backend, context_tokens=context, query_heads=32,
                                           kv_heads=8, head_dim=128,
                                           resident_bytes=2 * context * 8 * 128 * 2,
                                           cold_bytes=2 * context * 8 * 128 * 2)
                      for context in (256, 1024, 4096, 16384)]
    overflow = plan_attention_route(backend, 1024, query_heads=32, kv_heads=8, head_dim=128,
                                    resident_bytes=backend.config.pim_capacity_bytes + 1)
    payload = {'schema_version': 'agent-memory-pim-probe-v2', 'seed': args.seed,
               'profile': 'pim_backend/mobile_lpddr5_bank_profile.json',
               'profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
               'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), ROOT / 'pim_backend/agent_memory_pim.py',
                                              ROOT / 'pim_backend/lpddr_bank_pim.py']},
               'functional_attention': functional, 'attention_routes': attention,
               'cold_attention_routes': cold_attention, 'overflow_capacity_route': overflow,
               'boundary': 'functional NumPy equivalence plus uncalibrated LPDDR phase model; not device E2E'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({'output': str(args.output), 'functional_pass': functional['exact_within_tolerance']}))


if __name__ == '__main__':
    main()
