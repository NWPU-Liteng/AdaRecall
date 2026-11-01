"""Dependency-aware backend planning for evidence state and Reader execution."""
from __future__ import annotations

from pim_backend.evidence_state import compile_evidence_state, plan_evidence_route


def choose_backend(*, reader_dtype, object_frontier_bytes, input_tokens,
                   lifecycle=None, question_id=None, evidence_profile=None,
                   evidence_stage_seconds=None, total_service_seconds=None,
                   quality_contract=False, grounding_route=None):
    if reader_dtype != 'bf16':
        raise ValueError('proof candidate currently freezes BF16 Reader')
    evidence_state = None
    if lifecycle is None:
        evidence_route = {
            'schema': 'evidence-backend-route-v1',
            'chosen_backend': 'host',
            'candidate_backend': 'pim',
            'route_scope': 'evidence_state_only',
            'physical_execution': False,
            'quality_contract': False,
            'payload_bytes': 0,
            'metadata_bytes': 0,
            'packed_bytes': 0,
            'host_stage_seconds': None,
            'total_service_seconds': None,
            'pim_stage_seconds': None,
            'net_stage_savings_seconds': None,
            'end_to_end_speedup': None,
            'capacity_admitted': None,
            'profile_qualified': False,
            'reason_codes': ['evidence_lifecycle_missing', 'host_fallback_required'],
        }
    else:
        evidence_state = compile_evidence_state(
            lifecycle, question_id=question_id, reader_input_tokens=input_tokens)
        evidence_route = plan_evidence_route(
            evidence_state, profile=evidence_profile,
            host_stage_seconds=evidence_stage_seconds,
            total_service_seconds=total_service_seconds,
            quality_contract=quality_contract)
    if grounding_route is None:
        grounding_route = {
            'schema': 'evidence-backend-route-v1',
            'chosen_backend': 'host',
            'candidate_backend': 'pim',
            'route_scope': 'evidence_state_only',
            'physical_execution': False,
            'quality_contract': False,
            'payload_bytes': 0,
            'metadata_bytes': 0,
            'packed_bytes': 0,
            'host_stage_seconds': None,
            'total_service_seconds': None,
            'pim_stage_seconds': None,
            'net_stage_savings_seconds': None,
            'end_to_end_speedup': None,
            'capacity_admitted': None,
            'profile_qualified': False,
            'reason_codes': ['grounding_frontier_missing', 'host_fallback_required'],
        }
    return {'schema': 'proof-backend-plan-v2', 'chosen_backend': 'host',
            'phase_routes': [{'phase': 'grounding_frontier',
                              'backend': grounding_route['chosen_backend']},
                             {'phase': 'evidence_materialize',
                              'backend': evidence_route['chosen_backend']},
                             {'phase': 'prefill', 'backend': 'host'},
                             {'phase': 'decode', 'backend': 'host'}],
            'evidence_state': evidence_state,
            'grounding_route': grounding_route,
            'evidence_route': evidence_route,
            'precompiled_templates': ['host_bf16', 'lpddr_pim_bf16_projection_candidate'],
            'candidate_route': {'backend': 'pim', 'enabled': False,
                'reason': 'LPDDR calibration, full Reader capacity and shared-bus makespan are unknown',
                'input_frontier_bytes': object_frontier_bytes, 'input_tokens': input_tokens},
            'physical_execution': False,
            'unknowns': ['device LPDDR bandwidth/profile calibration', 'full Reader weight/state capacity',
                         'Host/PIM shared-channel contention', 'validated BF16 numerical contract'],
            'boundary': 'host serving trace is measured server software; PIM route is a frozen interface and not a speedup claim'}
