"""Compile reader evidence into a routable, resource-aware state frontier."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math


CONSUMER_BITS = {
    'reader_prefill': 1,
    'reader_decode': 2,
}
KIND_CODES = {'text': 1, 'image': 2}
STATE_SCHEMA = 'evidence-state-frontier-v1'
CANDIDATE_FRONTIER_SCHEMA = 'evidence-candidate-frontier-v1'


@dataclass(frozen=True)
class EvidenceRouteProfile:
    """A complete cost contract for a candidate evidence backend."""

    name: str
    host_bytes_per_second: float
    pim_bytes_per_second: float
    pim_capacity_bytes: int
    pim_setup_seconds: float
    pim_transfer_seconds_per_byte: float
    pim_merge_seconds: float
    qualified: bool = False
    enabled: bool = False
    executable: bool = False

    def validate(self) -> None:
        if not self.name:
            raise ValueError('route profile name required')
        positive_values = {
            'host_bytes_per_second': self.host_bytes_per_second,
            'pim_bytes_per_second': self.pim_bytes_per_second,
            'pim_capacity_bytes': self.pim_capacity_bytes,
        }
        for field, value in positive_values.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value <= 0:
                raise ValueError(f'{field} must be finite and positive')
        nonnegative_values = {
            'pim_setup_seconds': self.pim_setup_seconds,
            'pim_transfer_seconds_per_byte': self.pim_transfer_seconds_per_byte,
            'pim_merge_seconds': self.pim_merge_seconds,
        }
        for field, value in nonnegative_values.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(f'{field} must be finite and nonnegative')


def _finite_nonnegative(value, field):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{field} must be finite and nonnegative')
    return value


def compile_evidence_state(lifecycle, *, question_id=None, reader_input_tokens=None):
    """Build a fixed-width logical state table from the selected evidence.

    The output is a planning representation, not a device layout or a latency
    claim.  It makes payload, metadata, consumers and source identities explicit
    so a backend decision cannot be made from object count alone.
    """
    if lifecycle.get('schema') != 'proof-lifecycle-v1':
        raise ValueError('proof lifecycle schema required')
    states = lifecycle.get('states')
    if not isinstance(states, list):
        raise ValueError('lifecycle states must be a list')
    if reader_input_tokens is None:
        reader_input_tokens = lifecycle.get('reader_input_tokens')
    if reader_input_tokens is None or int(reader_input_tokens) < 0:
        raise ValueError('nonnegative reader input token count required')

    records = []
    seen_ids = set()
    source_ids = set()
    for state in states:
        state_id = state.get('state_id')
        kind = state.get('kind')
        resident_bytes = state.get('resident_bytes_estimate')
        consumers = state.get('consumers')
        source_identity = state.get('producer') or state_id
        if not isinstance(state_id, str) or not state_id:
            raise ValueError('state identity required')
        if state_id in seen_ids:
            raise ValueError('duplicate evidence state identity')
        if kind not in KIND_CODES:
            raise ValueError('text or image evidence state required')
        if not isinstance(consumers, list) or not consumers:
            raise ValueError('each evidence state needs a consumer')
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError('source identity required')
        resident_bytes = int(resident_bytes)
        if resident_bytes <= 0:
            raise ValueError('evidence state bytes must be positive')
        consumer_mask = 0
        for consumer in consumers:
            if consumer not in CONSUMER_BITS:
                raise ValueError('unknown evidence consumer')
            consumer_mask |= CONSUMER_BITS[consumer]
        source_ids.add(source_identity)
        seen_ids.add(state_id)
        records.append({
            'state_id': state_id,
            'state_id_sha256': hashlib.sha256(state_id.encode()).hexdigest(),
            'kind': kind,
            'kind_code': KIND_CODES[kind],
            'source_identity': source_identity,
            'resident_bytes': resident_bytes,
            'consumer_mask': consumer_mask,
            'consumer_count': len(consumers),
            'release_after': state.get('release_after'),
            'recompute': state.get('recompute'),
            'identity_bound': state.get('identity_bound') is True,
        })
    payload_bytes = sum(record['resident_bytes'] for record in records)
    metadata_bytes = len(records) * 32
    return {
        'schema': STATE_SCHEMA,
        'question_id': question_id,
        'reader_input_tokens': int(reader_input_tokens),
        'records': records,
        'state_count': len(records),
        'unique_source_count': len(source_ids),
        'payload_bytes': payload_bytes,
        'metadata_bytes': metadata_bytes,
        'packed_bytes': payload_bytes + metadata_bytes,
        'consumer_contract': {'reader_prefill': 1, 'reader_decode': 2},
        'record_format': 'fixed-width metadata v1 plus source payload bytes',
        'boundary': 'logical evidence-state planning bytes; not device residency or process RSS',
    }


def plan_evidence_route(state, *, profile=None, host_stage_seconds=None,
                        total_service_seconds=None, quality_contract=False):
    """Select Host or a qualified PIM candidate using whole-stage accounting."""
    if state.get('schema') not in {STATE_SCHEMA, CANDIDATE_FRONTIER_SCHEMA}:
        raise ValueError('evidence state frontier schema required')
    host_stage_seconds = _finite_nonnegative(host_stage_seconds, 'host_stage_seconds')
    total_service_seconds = _finite_nonnegative(total_service_seconds, 'total_service_seconds')
    if (total_service_seconds is not None and host_stage_seconds is not None
            and host_stage_seconds > total_service_seconds):
        raise ValueError('host evidence stage cannot exceed total service')

    base = {
        'schema': 'evidence-backend-route-v1',
        'chosen_backend': 'host',
        'candidate_backend': 'pim',
        'route_scope': 'evidence_state_only',
        'physical_execution': False,
        'quality_contract': bool(quality_contract),
        'payload_bytes': state['payload_bytes'],
        'metadata_bytes': state['metadata_bytes'],
        'packed_bytes': state['packed_bytes'],
        'host_stage_seconds': host_stage_seconds,
        'total_service_seconds': total_service_seconds,
        'pim_stage_seconds': None,
        'net_stage_savings_seconds': None,
        'end_to_end_speedup': None,
        'capacity_admitted': None,
        'profile_qualified': False,
        'reason_codes': [],
    }
    if profile is None:
        base['reason_codes'].append('missing_backend_profile')
        return base
    if isinstance(profile, dict):
        profile = EvidenceRouteProfile(**profile)
    if not isinstance(profile, EvidenceRouteProfile):
        raise TypeError('profile must be EvidenceRouteProfile or dict')
    profile.validate()
    base['profile'] = profile.name
    base['profile_qualified'] = profile.qualified
    base['capacity_admitted'] = state['packed_bytes'] <= profile.pim_capacity_bytes
    if not base['capacity_admitted']:
        base['reason_codes'].append('capacity_rejected')
    if not profile.qualified:
        base['reason_codes'].append('profile_unqualified')
    if not profile.enabled:
        base['reason_codes'].append('profile_disabled')
    if not profile.executable:
        base['reason_codes'].append('backend_not_implemented')
    if host_stage_seconds is None:
        base['reason_codes'].append('host_stage_unmeasured')
    pim_stage_seconds = (profile.pim_setup_seconds
                         + state['packed_bytes'] / profile.pim_bytes_per_second
                         + state['packed_bytes'] * profile.pim_transfer_seconds_per_byte
                         + profile.pim_merge_seconds)
    base['pim_stage_seconds'] = pim_stage_seconds
    if host_stage_seconds is not None:
        base['net_stage_savings_seconds'] = host_stage_seconds - pim_stage_seconds
        if base['net_stage_savings_seconds'] <= 0:
            base['reason_codes'].append('not_net_positive')
    if total_service_seconds is not None and host_stage_seconds is not None:
        candidate_service_seconds = total_service_seconds - host_stage_seconds + pim_stage_seconds
        if candidate_service_seconds <= 0:
            raise ValueError('candidate service time must be positive')
        base['route_scope'] = 'end_to_end_bound'
        base['end_to_end_speedup'] = total_service_seconds / candidate_service_seconds
        if base['end_to_end_speedup'] <= 1:
            base['reason_codes'].append('not_end_to_end_positive')

    gates = (
        base['capacity_admitted'],
        base['profile_qualified'],
        profile.enabled,
        profile.executable,
        host_stage_seconds is not None,
        base['net_stage_savings_seconds'] is not None and base['net_stage_savings_seconds'] > 0,
        bool(quality_contract),
    )
    if all(gates):
        base['chosen_backend'] = 'pim'
        base['physical_execution'] = True
        base['reason_codes'] = ['all_route_gates_passed']
    else:
        base['reason_codes'].append('host_fallback_required')
    return base
