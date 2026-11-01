"""Project executed batch-1 decode Linear trace onto the fixed LPDDR-PIM model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.decode_gemv import project_trace
from pim_backend.lpddr_bank_pim import BankPimConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh projection artifact required')
    smoke = json.loads(args.smoke.read_text())
    if not smoke.get('complete'):
        raise ValueError('incomplete raw-image smoke')
    profile = json.loads(args.profile.read_text())
    config = BankPimConfig(**profile['config'])
    result = project_trace(smoke['operator_trace']['operators'], config)
    result.update({
        'schema_version': 'executed-decode-linear-pim-projection-v1',
        'smoke_sha256': hashlib.sha256(args.smoke.read_bytes()).hexdigest(),
        'profile_sha256': hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        'reader_config_sha256': smoke['reader_config_sha256'],
        'reader_weight_bytes': smoke['reader_weight_bytes'],
        'questions': len(smoke['rows']),
        'decode_tokens_observed': sum(len(row['answer_token_ids']) for row in smoke['rows']),
        'profile_name': profile['name'],
        'profile_calibration_status': profile['calibration_status'],
        'full_model_residency_validated': False,
        'capacity_scope': 'PIM pool only; full model plus runtime state DRAM capacity is not qualified',
        'source_files_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ('scripts/project_decode_pim.py',
                                            'pim_backend/decode_gemv.py',
                                            'pim_backend/lpddr_bank_pim.py')},
        'operator_trace_coverage': smoke['operator_trace']['coverage'],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'executed_decode_linear_calls', 'candidate_matrices', 'resident_matrices',
        'host_linear_cycles', 'warm_hybrid_linear_cycles', 'cold_hybrid_linear_cycles',
        'warm_linear_speedup_scenario', 'cold_linear_speedup_scenario',
        'end_to_end_speedup', 'boundary')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
