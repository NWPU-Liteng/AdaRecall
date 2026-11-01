#!/usr/bin/env python3
"""Audit 8/10/12-bit row-packed decoder residency against one 8 GiB pool."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.hyperpim_sc import packed_bit_sliced_decoder_plan


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def build_audit(r0_projection: dict, r2_projection: dict, *, capacity_bytes=8 * 1024**3):
    matrices = r0_projection['decode_token_accounting']['decoder_bank_residency']['matrices']
    if len(matrices) != 249:
        raise ValueError('frozen 249-matrix decoder inventory required')
    pool = r0_projection['decode_token_accounting']['pool_residency']
    token_table = int(pool['token_table_bytes'])
    grammar = int(pool['grammar_state_bytes'])
    scratch = int(pool['scratch_bytes'])
    r2_pool = r2_projection['decode_token_accounting']['pool_residency']
    if r2_pool.get('feasible') is not True:
        raise ValueError('current W8A16+r2 pool must be the feasible deployment reference')

    variants = {}
    for bits in (8, 10, 12):
        plan = packed_bit_sliced_decoder_plan(
            matrices, weight_bits=bits, row_bytes=2048,
            capacity_bytes=capacity_bytes)
        total_alias = plan['total_bytes'] + token_table + grammar + scratch
        total_folded = plan['total_bytes'] + grammar + scratch
        variants[f'w{bits}a16'] = {
            'weight_bits': bits,
            'decoder_plan': plan,
            'pool_with_bf16_embedding_alias_bytes': total_alias,
            'headroom_with_bf16_embedding_alias_bytes': capacity_bytes - total_alias,
            'feasible_with_bf16_embedding_alias': total_alias <= capacity_bytes,
            'pool_with_bank_resident_embedding_row_fetch_bytes': total_folded,
            'headroom_with_bank_resident_embedding_row_fetch_bytes': (
                capacity_bytes - total_folded),
            'feasible_with_bank_resident_embedding_row_fetch': (
                total_folded <= capacity_bytes),
        }

    return {
        'schema_version': 'w12-bank-capacity-audit-v1',
        'complete': True,
        'scope': 'capacity accounting only; no quality or latency qualification',
        'inputs_sha256': {},
        'model_inventory': {
            'decoder_matrices': len(matrices),
            'decoder_weight_elements': sum(
                int(row['out_features']) * int(row['in_features']) for row in matrices),
            'token_table_bytes': token_table,
            'grammar_state_bytes': grammar,
            'scratch_bytes': scratch,
            'capacity_bytes': capacity_bytes,
        },
        'current_w8a16_r2_reference': {
            'pool_total_bytes': r2_pool['total_bytes'],
            'headroom_bytes': r2_pool['headroom_bytes'],
            'feasible': r2_pool['feasible'],
        },
        'variants': variants,
        'mechanism_finding': (
            'For the 2560-wide decoder rows, a 12-bit logical row is 3840 B and '
            'therefore still occupies exactly two 2048 B physical rows. The '
            'uniform 12-bit plane removes the residual payload and residual scale '
            'plane while retaining a single group-scale plane.'),
        'recommended_deployment_form': (
            'uniform 12-bit bit-sliced decoder plane with row-packed storage; '
            'BF16 embedding alias retained during functional quality replay; '
            'bank-resident embedding row fetch is a separate capacity option'),
        'capacity_boundary': (
            'the row-packed model is a target layout, not a physical controller or '
            'repair-map measurement; it does not include refresh metadata, repair, '
            'runtime migration, or the initial checkpoint conversion buffer; the '
            'embedding-fold path is an unexecuted architectural option'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--r0-projection', type=Path, required=True)
    parser.add_argument('--r2-projection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capacity-gib', type=float, default=8.0)
    args = parser.parse_args()
    capacity_bytes = int(args.capacity_gib * 1024**3)
    if capacity_bytes <= 0:
        raise ValueError('positive capacity required')
    r0_path, r2_path = args.r0_projection, args.r2_projection
    result = build_audit(json.loads(r0_path.read_text()),
                         json.loads(r2_path.read_text()),
                         capacity_bytes=capacity_bytes)
    result['inputs_sha256'] = {str(r0_path): digest(r0_path),
                               str(r2_path): digest(r2_path)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({
        'current': result['current_w8a16_r2_reference'],
        'variants': {
            name: {
                'pool_with_alias': value['pool_with_bf16_embedding_alias_bytes'],
                'free_with_alias': value['headroom_with_bf16_embedding_alias_bytes'],
                'feasible_with_alias': value['feasible_with_bf16_embedding_alias'],
                'pool_folded': value['pool_with_bank_resident_embedding_row_fetch_bytes'],
                'free_folded': value['headroom_with_bank_resident_embedding_row_fetch_bytes'],
                'feasible_folded': value['feasible_with_bank_resident_embedding_row_fetch'],
            }
            for name, value in result['variants'].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
