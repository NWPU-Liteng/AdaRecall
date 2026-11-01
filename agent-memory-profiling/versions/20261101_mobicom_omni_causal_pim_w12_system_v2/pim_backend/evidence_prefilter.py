"""Recall-preserving evidence-state prefilter planning.

The prefilter is intentionally a qualification primitive, not a semantic
retriever.  It may remove a candidate only when the query contains an exact
structural relation and the candidate table carries the corresponding
provenance metadata.  Missing or incomplete metadata always fails open to the
Host grounding path.
"""
from __future__ import annotations

from pim_backend.proof_query import (compile_candidate_table, execute_proof_query,
                                     compile_proof_query)


SCHEMA = 'evidence-state-prefilter-v1'


def _record_id(record):
    identity = record.get('base_record_id') or record.get('record_id')
    if not isinstance(identity, str) or not identity:
        raise ValueError('candidate base_record_id is required')
    return identity


def audit_evidence_prefilter(prefilter_result, baseline_candidates, host_witnesses):
    baseline_keys = sorted({(record.get('demand_id'), _record_id(record))
                            for record in baseline_candidates})
    baseline_ids = sorted({_record_id(record) for record in baseline_candidates})
    audit = {
        'baseline_frontier_candidate_count': len(baseline_keys),
        'filtered_baseline_frontier_candidate_count': None,
        'baseline_frontier_reduction_fraction': None,
        'host_witness_count': 0,
        'host_witness_missing_ids': [],
        'host_witness_coverage': None,
        'safe_to_replace_host_grounding': False,
    }
    if not prefilter_result.get('applied'):
        return audit

    query_result = prefilter_result.get('result') or {}
    eligible_ids = set(prefilter_result.get('package_base_record_ids')
                       or query_result.get('eligible_base_record_ids', []))
    eligible_keys = {(item.get('demand_id'), item.get('base_record_id'))
                     for item in prefilter_result.get('package_candidate_keys', [])}
    retained_baseline_keys = ({(item.get('demand_id'), item.get('base_record_id'))
                               for item in prefilter_result.get('package_candidate_keys', [])}
                              if prefilter_result.get('package_candidate_keys') else
                              {(record.get('demand_id'), _record_id(record))
                               for record in baseline_candidates
                               if _record_id(record) in eligible_ids})
    retained_baseline_keys &= set(baseline_keys)
    audit['filtered_baseline_frontier_candidate_count'] = len(retained_baseline_keys)
    audit['baseline_frontier_reduction_fraction'] = (
        1.0 - len(retained_baseline_keys) / len(baseline_keys) if baseline_keys else 0.0)
    host_keys = {(witness.get('demand_id'), identity)
                 for witness in host_witnesses
                 for identity in (witness.get('retrieval_record_ids')
                                  or witness.get('record_ids', []))}
    missing_host_keys = sorted(
        key for key in host_keys
        if ((key not in eligible_keys) if key[0] and eligible_keys
            else key[1] not in eligible_ids))
    host_ids = sorted({identity for _, identity in host_keys})
    missing_host_ids = sorted({identity for _, identity in missing_host_keys})
    audit.update({
        'host_witness_count': len(host_ids),
        'host_witness_missing_ids': missing_host_ids,
        'host_witness_coverage': not missing_host_ids if host_ids else None,
    })
    return audit


def _has_exact_relation_metadata(plan, table):
    relations = plan.get('relations', [])
    scoped_demands = [demand for demand in plan.get('demands', [])
                      if demand.get('evidence_scope', {}).get('kind')]
    if not relations and not scoped_demands:
        return False, 'no_exact_evidence_predicate'
    for demand in scoped_demands:
        required_kind = demand['evidence_scope']['kind']
        rows = [row for row in table['rows'] if row['demand_id'] == demand['demand_id']]
        if not rows or any(row.get('evidence_metadata', {}).get('kind') is None
                           for row in rows):
            return False, 'evidence_scope_metadata_incomplete'
    for relation in relations:
        if relation['kind'] in {'same_binding', 'different_binding'}:
            variable = relation['variable']
            demands = {relation['left'], relation['right']}
            rows = [row for row in table['rows'] if row['demand_id'] in demands]
            if not rows or any(variable not in row['bindings'] for row in rows):
                return False, 'binding_metadata_incomplete'
        elif relation['kind'] == 'source_relation':
            rows = [row for row in table['rows']
                    if row['demand_id'] in {relation['left'], relation['right']}]
            if not rows or any(relation['relation_id'] not in row['source_relations']
                               for row in rows):
                return False, 'source_relation_metadata_incomplete'
    return True, None


def _exact_lineage(record):
    metadata = record.get('evidence_metadata')
    if not isinstance(metadata, dict):
        metadata = record
    session_id = metadata.get('session_id')
    container_id = metadata.get('container_id')
    object_ids = metadata.get('object_ids', record.get('object_ids'))
    source_atom_ids = metadata.get('source_atom_ids', record.get('source_atom_ids'))
    primary_object_id = metadata.get('primary_object_id')
    if (not isinstance(session_id, str) or not session_id
            or not isinstance(container_id, str) or not container_id
            or not isinstance(object_ids, list)
            or any(not isinstance(value, str) or not value for value in object_ids)
            or not isinstance(source_atom_ids, list)
            or any(not isinstance(value, str) or not value for value in source_atom_ids)
            or not isinstance(primary_object_id, str) or not primary_object_id):
        return None
    return {
        'session_id': session_id,
        'container_id': container_id,
        'object_ids': set(object_ids),
        'source_atom_ids': set(source_atom_ids),
        'primary_object_id': primary_object_id,
    }


def _same_demand_evidence_package(seed, candidate, seed_lineage, candidate_lineage):
    if seed['demand_id'] != candidate['demand_id']:
        return False
    if (seed_lineage['session_id'] == candidate_lineage['session_id']
            or seed_lineage['container_id'] == candidate_lineage['container_id']):
        return True
    if seed_lineage['source_atom_ids'] & candidate_lineage['source_atom_ids']:
        return True
    if seed_lineage['object_ids'] & candidate_lineage['object_ids']:
        return True
    return (seed_lineage['primary_object_id'] in candidate_lineage['object_ids']
            or candidate_lineage['primary_object_id'] in seed_lineage['object_ids'])


def _expand_evidence_packages(candidates, seed_ids):
    lineages = [_exact_lineage(record) for record in candidates]
    if any(lineage is None for lineage in lineages):
        return None
    selected = {index for index, record in enumerate(candidates)
                if _record_id(record) in seed_ids}
    changed = True
    while changed:
        changed = False
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in selected:
                continue
            if any(_same_demand_evidence_package(
                    candidates[seed_index], candidate,
                    lineages[seed_index], lineages[candidate_index])
                    for seed_index in selected):
                selected.add(candidate_index)
                changed = True
    return [record for index, record in enumerate(candidates) if index in selected]


def plan_evidence_state_prefilter(*, demands, relations, operations, candidates,
                                  answer_mode='structured', max_results=None):
    """Build a fail-open prefilter plan over a grounding candidate pool.

    The returned ``candidates`` list is suitable for a shadow comparison.  A
    caller must not replace the Host grounding input unless it has separately
    completed the baseline witness-coverage gate.
    """
    original = list(candidates)
    query_plan = compile_proof_query(
        demands=demands, relations=relations, operations=operations,
        answer_mode=answer_mode)
    table = compile_candidate_table(original)
    exact_metadata, fallback_reason = _has_exact_relation_metadata(query_plan, table)
    base = {
        'schema': SCHEMA,
        'execution_backend': 'host_reference',
        'physical_pim_execution': False,
        'candidate_table': {
            'row_count': len(table['rows']),
            'row_width_bytes': table['row_width_bytes'],
            'descriptor_bytes': table['descriptor_bytes'],
            'sidecar_identity_count': table['sidecar_identity_count'],
        },
        'input_candidate_count': len(original),
        'filtered_candidate_count': len(original),
        'seed_candidate_count': len(original),
        'package_expansion_count': 0,
            'package_base_record_ids': sorted({_record_id(record) for record in original}),
        'package_candidate_keys': [
            {'demand_id': record['demand_id'], 'base_record_id': _record_id(record)}
            for record in original],
        'package_proposed': False,
        'coverage_gate_passed': None,
        'effective_candidate_count': len(original),
        'candidate_reduction_fraction': 0.0,
        'exact_metadata': exact_metadata,
        'applied': False,
        'fallback_reason': fallback_reason,
        'result': None,
    }
    if not exact_metadata:
        return {**base, 'candidates': original}

    if max_results is None:
        max_results = max(1, min(65536, len(table['rows']) * 16))
    if type(max_results) is not int or max_results < 1:
        raise ValueError('max_results must be a positive integer')
    result = execute_proof_query(query_plan, table, max_results=max_results)
    eligible = set(result.get('eligible_base_record_ids', []))
    seed_candidates = [record for record in original if _record_id(record) in eligible]
    filtered = _expand_evidence_packages(original, eligible)
    if filtered is None:
        return {**base, 'fallback_reason': 'evidence_lineage_metadata_incomplete',
                'result': result, 'candidates': original}
    if (not result.get('structural_contract_complete') or
            result.get('frontier_truncated') or not filtered):
        reason = ('structural_result_truncated' if result.get('frontier_truncated')
                  else result.get('fallback_reason', 'structural_contract_incomplete'))
        return {**base, 'fallback_reason': reason, 'result': result, 'candidates': original}

    reduction = 1.0 - len(filtered) / len(original) if original else 0.0
    return {**base, 'seed_candidate_count': len(seed_candidates),
            'filtered_candidate_count': len(filtered),
            'candidate_reduction_fraction': reduction,
            'package_expansion_count': len(filtered) - len(seed_candidates),
            'package_base_record_ids': sorted({_record_id(record) for record in filtered}),
            'package_candidate_keys': [
                {'demand_id': record['demand_id'], 'base_record_id': _record_id(record)}
                for record in filtered],
            'package_proposed': len(filtered) < len(original),
            'applied': len(filtered) < len(original),
            'fallback_reason': None if len(filtered) < len(original) else 'no_candidate_reduction',
            'result': result, 'candidates': filtered}


def enforce_host_witness_coverage(prefilter_result, audit, baseline_candidates):
    """Fail open when the evidence package drops any observed Host witness."""
    result = dict(prefilter_result)
    result['frontier_audit'] = audit
    result['coverage_gate_passed'] = (
        audit.get('host_witness_coverage') is True if result.get('package_proposed') else None)
    result['fallback_to_baseline'] = False
    result['effective_candidate_count'] = result.get('filtered_candidate_count')
    baseline_frontier_count = audit.get('baseline_frontier_candidate_count')
    if baseline_frontier_count is None:
        baseline_frontier_count = len({_record_id(record) for record in baseline_candidates})
    effective_frontier_count = baseline_frontier_count
    if result.get('package_proposed') and result['coverage_gate_passed'] is True:
        effective_frontier_count = audit['filtered_baseline_frontier_candidate_count']
    result['effective_frontier_candidate_count'] = effective_frontier_count
    result['effective_frontier_reduction_fraction'] = (
        1.0 - effective_frontier_count / baseline_frontier_count
        if baseline_frontier_count else 0.0)
    if result.get('package_proposed') and audit.get('host_witness_coverage') is not True:
        baseline_count = len(baseline_candidates)
        result['proposed_filtered_candidate_count'] = result['filtered_candidate_count']
        result['proposed_candidate_reduction_fraction'] = result['candidate_reduction_fraction']
        result.update({
            'applied': False,
            'fallback_reason': ('host_witness_coverage_failed'
                                if audit.get('host_witness_count', 0)
                                else 'no_host_witnesses_to_validate'),
            'filtered_candidate_count': baseline_count,
            'candidate_reduction_fraction': 0.0,
            'effective_candidate_count': baseline_count,
            'effective_frontier_candidate_count': baseline_frontier_count,
            'effective_frontier_reduction_fraction': 0.0,
            'fallback_to_baseline': True,
        })
    return result
