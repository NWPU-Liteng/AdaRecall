"""Register development promotion gates without changing failed denominators."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from system_audit import SCHEMA


def promotion_gates(audit):
    accuracy = (audit.get('local_judge') or {}).get('local_semantic_answer_accuracy')
    selection = audit.get('execution_selection', {})
    return {'engineering_pass': audit['engineering_pass'],
            'development32_temporal_single_stream': audit.get('questions') == 32
                and selection.get('limit') == 768 and selection.get('sample_count') == 32
                and selection.get('shard_count') == 1 and selection.get('shard_index') == 0,
            'system_v2_audit': audit.get('schema') == SCHEMA and audit.get('system_v2_pass') is True,
            'audit_protocol': audit.get('protocol') == 'audit',
            'source_event_hit_recall_ge_0_90': audit['source_event_hit_recall'] >= 0.90,
            'local_development_semantic_accuracy_ge_0_75': accuracy is not None and accuracy >= 0.75,
            'host_service_p95_le_20_seconds': audit['service_including_validation_seconds']['p95'] <= 20,
            'generation_limits_not_reached': not any(audit['generation_limit_counts'].values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    output = args.result.parent / 'promotion.json'
    if output.exists():
        raise ValueError('promotion record is immutable')
    audit = json.loads(args.audit.read_text())
    result_sha256 = hashlib.sha256(args.result.read_bytes()).hexdigest()
    if audit.get('provenance', {}).get('result_sha256') != result_sha256:
        raise ValueError('audit must bind this exact result file')
    gates = promotion_gates(audit)
    record = {'gates': gates, 'engineering_candidate_for_development128': all(gates.values()),
              'ready_for_development128': False,
              'pending_contract_evidence': ['independent calibrated quality/witness audit',
                                            'bound PIM mechanism and calibration artifacts',
                                            'frozen continuous-stream execution contract'],
              'sealed_test_authorized': False, 'publication_ready': False,
              'reference_judge_accuracy': None,
              'source_questions': audit.get('questions'),
              'source_execution_selection': audit.get('execution_selection'),
              'gate_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'result_sha256': result_sha256,
              'audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(),
              'boundary': 'predefined development milestones, not a publication-quality certificate; no test-set tuning'}
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
