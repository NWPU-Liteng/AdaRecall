"""Replay main-run vector-scan shapes through the shared LPDDR timing model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.lpddr_pim import SharedLpddrTimingModel


def profile(run_root: Path, index_root: Path, output: Path, max_rows: int = 0) -> dict:
    model = SharedLpddrTimingModel()
    records, per_user = [], {}
    for run_path in sorted(run_root.glob('user_*_shard_*.json')):
        payload = json.loads(run_path.read_text())
        if not payload.get('complete'):
            continue
        user_id = str(payload['person_id'])
        index_path = index_root / f'proof_index_user_{user_id}' / 'index.json'
        if not index_path.is_file():
            raise ValueError(f'missing index metadata for user {user_id}: {index_path}')
        index_meta = json.loads(index_path.read_text())
        vector_count = int(index_meta['records'])
        selected = payload.get('rows', [])
        if max_rows:
            selected = selected[:max_rows]
        user_rows = []
        for row in selected:
            result = model.simulate_scan(vector_count, 1024, bytes_per_element=2,
                                         route='pim', top_indices=tuple())
            host = model.simulate_scan(vector_count, 1024, bytes_per_element=2,
                                       route='cpu', top_indices=tuple())
            item = {'question_id': row['question_id'], 'user_id': user_id,
                    'vector_count': vector_count, 'dimension': 1024,
                    'bytes_per_element': 2, 'selected_object_count': len(row.get('materialized_object_ids', [])),
                    'pim_cycles': result.cycles, 'host_cycles': host.cycles,
                    'pim_trace_events': len(result.trace),
                    'bank_conflicts': result.bank_conflicts,
                    'channel_conflicts': result.channel_conflicts,
                    'refresh_stalls': result.refresh_stalls,
                    'topk_overlap': result.topk_overlap}
            user_rows.append(item)
            records.append(item)
        per_user[user_id] = {'questions': len(user_rows),
                             'pim_cycles': sum(item['pim_cycles'] for item in user_rows),
                             'host_cycles': sum(item['host_cycles'] for item in user_rows)}
    payload = {
        'schema': 'mobilemem-omni-lpddr-pim-profile-v1',
        'rows': len(records),
        'per_user': per_user,
        'totals': {'pim_cycles': sum(item['pim_cycles'] for item in records),
                   'host_cycles': sum(item['host_cycles'] for item in records)},
        'rows_detail': records,
        'configuration': model.config.__dict__,
        'claim_boundary': 'shared LPDDR timing-model estimate; no PIM MAC silicon, calibration, full Reader residency, or mobile E2E claim',
        'route': 'vector-scan cost only; Reader prefill/decode remains outside this estimate',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--index-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=0)
    args = parser.parse_args()
    payload = profile(args.run_root, args.index_root, args.output, args.max_rows)
    print(json.dumps({key: payload[key] for key in ('rows', 'totals', 'claim_boundary')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
