"""Conservative proof-query compilation and reference execution.

The reference path operates on source identities and candidate metadata only.
It never turns a lexical or structural match into semantic evidence.  A future
PIM implementation may replace the execution primitive, but must preserve the
same result contract and fall back when the contract is not satisfied.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Iterable


SCHEMA = 'proof-query-plan-v1'
CANDIDATE_SCHEMA = 'proof-query-candidate-table-v1'


def _require_string(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{field} must be a non-empty string')
    return value


def _normalise_ids(values, field):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f'{field} must be a list')
    result = []
    seen = set()
    for value in values:
        value = _require_string(value, field)
        if value in seen:
            raise ValueError(f'{field} contains duplicate identity: {value}')
        seen.add(value)
        result.append(value)
    return result


def compile_proof_query(*, demands, relations=(), operations=(), answer_mode='structured'):
    """Compile the parser output into a deterministic, hardware-neutral IR.

    The IR describes only source observations and constraints.  It does not
    infer claims from query text and does not contain answer values.
    """
    if answer_mode not in {'structured', 'contextual'}:
        raise ValueError('answer_mode must be structured or contextual')
    if not isinstance(demands, list) or not demands:
        raise ValueError('at least one source demand is required')
    demand_ids = []
    demand_records = []
    for demand in demands:
        if not isinstance(demand, dict):
            raise ValueError('each demand must be an object')
        demand_id = _require_string(demand.get('demand_id'), 'demand_id')
        if demand_id in demand_ids:
            raise ValueError(f'duplicate demand_id: {demand_id}')
        text = _require_string(demand.get('text'), f'demand {demand_id} text')
        variables = demand.get('variables', [])
        variables = _normalise_ids(variables, f'demand {demand_id} variables')
        evidence_scope = demand.get('evidence_scope', {})
        if not isinstance(evidence_scope, dict) or set(evidence_scope) - {'kind'}:
            raise ValueError(f'demand {demand_id} evidence_scope must contain only kind')
        if evidence_scope.get('kind') not in {None, 'source_text', 'image_locator'}:
            raise ValueError(f'demand {demand_id} evidence_scope kind is unsupported')
        demand_ids.append(demand_id)
        demand_record = {
            'demand_id': demand_id,
            'text': text,
            'variables': variables,
            'required': demand.get('required', True) is True,
            'query_quote': demand.get('query_quote'),
        }
        if evidence_scope:
            demand_record['evidence_scope'] = {'kind': evidence_scope['kind']}
        demand_records.append(demand_record)

    relation_records = []
    relation_ids = set()
    valid_kinds = {'same_binding', 'different_binding', 'source_relation'}
    for index, relation in enumerate(relations or []):
        if not isinstance(relation, dict):
            raise ValueError('each relation must be an object')
        kind = _require_string(relation.get('kind'), f'relation {index} kind')
        if kind not in valid_kinds:
            raise ValueError(f'unsupported relation kind: {kind}')
        relation_id = relation.get('relation_id', f'rel{index}')
        relation_id = _require_string(relation_id, f'relation {index} relation_id')
        if relation_id in relation_ids:
            raise ValueError(f'duplicate relation_id: {relation_id}')
        left = _require_string(relation.get('left'), f'relation {relation_id} left')
        right = _require_string(relation.get('right'), f'relation {relation_id} right')
        if left not in demand_ids or right not in demand_ids or left == right:
            raise ValueError(f'relation {relation_id} references invalid demands')
        if kind in {'same_binding', 'different_binding'}:
            variable = _require_string(relation.get('variable'), f'relation {relation_id} variable')
        else:
            variable = None
        relation_ids.add(relation_id)
        relation_records.append({
            'relation_id': relation_id,
            'kind': kind,
            'left': left,
            'right': right,
            'variable': variable,
            'query_quote': relation.get('query_quote'),
        })

    operation_records = []
    operation_ids = set()
    operation_kinds = {'difference', 'comparison', 'selection', 'aggregation'}
    available = set(demand_ids)
    for index, operation in enumerate(operations or []):
        if not isinstance(operation, dict):
            raise ValueError('each operation must be an object')
        operation_id = _require_string(operation.get('operation_id', f'op{index}'),
                                       f'operation {index} operation_id')
        if operation_id in operation_ids or operation_id in available:
            raise ValueError(f'duplicate operation identity: {operation_id}')
        kind = _require_string(operation.get('kind'), f'operation {operation_id} kind')
        if kind not in operation_kinds:
            raise ValueError(f'unsupported operation kind: {kind}')
        inputs = _normalise_ids(operation.get('inputs', []), f'operation {operation_id} inputs')
        if not inputs or not set(inputs) <= available:
            raise ValueError(f'operation {operation_id} has a forward or missing input')
        operation_ids.add(operation_id)
        available.add(operation_id)
        operation_records.append({
            'operation_id': operation_id,
            'kind': kind,
            'inputs': inputs,
            'query_quote': operation.get('query_quote'),
        })

    payload = {
        'schema': SCHEMA,
        'answer_mode': answer_mode,
        'demands': demand_records,
        'relations': relation_records,
        'operations': operation_records,
        'semantic_claims': False,
        'execution_contract': 'source_identity_and_constraint_filter_only',
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    payload['plan_sha256'] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


def _candidate_identity(candidate, index):
    identity = candidate.get('witness_id') or candidate.get('candidate_id')
    return _require_string(identity or f'candidate:{index}', 'candidate identity')


def compile_candidate_table(candidates: Iterable[dict]):
    """Pack candidate metadata into a fixed-width logical table.

    ``descriptor_bytes`` is a storage accounting value, not a latency model.
    Variable-length source/object identities remain in the Host-side sidecar.
    """
    rows = []
    seen = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError('candidate must be an object')
        identity = _candidate_identity(candidate, index)
        if identity in seen:
            raise ValueError(f'duplicate candidate identity: {identity}')
        demand_id = _require_string(candidate.get('demand_id'), 'candidate demand_id')
        object_ids = _normalise_ids(candidate.get('object_ids', []), f'candidate {identity} object_ids')
        bindings = candidate.get('bindings', {})
        if not isinstance(bindings, dict):
            raise ValueError(f'candidate {identity} bindings must be an object')
        source_relations = _normalise_ids(candidate.get('source_relations', []),
                                          f'candidate {identity} source_relations')
        source_atom_ids = _normalise_ids(candidate.get('source_atom_ids', []),
                                         f'candidate {identity} source_atom_ids')
        score = float(candidate.get('score', 0.0))
        if not math.isfinite(score):
            raise ValueError(f'candidate {identity} score must be finite')
        row = {
            'candidate_id': identity,
            'demand_id': demand_id,
            'base_record_id': candidate.get('base_record_id', identity),
            'object_ids': object_ids,
            'bindings': {str(key): str(value) for key, value in sorted(bindings.items())},
            'source_relations': source_relations,
            'source_atom_ids': source_atom_ids,
            'score': score,
            'source_identity': candidate.get('source_identity'),
        }
        exact_metadata = candidate.get('evidence_metadata')
        if exact_metadata is None:
            exact_metadata = {
                field: candidate.get(field)
                for field in ('kind', 'session_id', 'container_id', 'primary_object_id',
                              'source_span')
                if candidate.get(field) is not None
            }
        if not isinstance(exact_metadata, dict):
            raise ValueError(f'candidate {identity} evidence_metadata must be an object')
        row['evidence_metadata'] = exact_metadata
        rows.append(row)
        seen.add(identity)
    descriptor_width = 64
    return {
        'schema': CANDIDATE_SCHEMA,
        'rows': rows,
        'row_width_bytes': descriptor_width,
        'descriptor_bytes': len(rows) * descriptor_width,
        'sidecar_identity_count': sum(len(row['object_ids']) + len(row['source_atom_ids']) for row in rows),
        'semantic_claims': False,
    }


def _compatible(candidate, chosen, relations):
    for previous in chosen.values():
        common = set(candidate['bindings']) & set(previous['bindings'])
        for variable in common:
            if candidate['bindings'][variable] == previous['bindings'][variable]:
                continue
            if any(relation['kind'] == 'different_binding'
                   and relation['variable'] == variable
                   and {relation['left'], relation['right']} ==
                   {candidate['demand_id'], previous['demand_id']}
                   for relation in relations):
                continue
            return False
    joined = {**chosen, candidate['demand_id']: candidate}
    for relation in relations:
        if relation['left'] not in joined or relation['right'] not in joined:
            continue
        left = joined[relation['left']]
        right = joined[relation['right']]
        if relation['kind'] == 'source_relation':
            available = set(left['source_relations']) | set(right['source_relations'])
            if relation['relation_id'] not in available:
                return False
            continue
        variable = relation['variable']
        if variable not in left['bindings'] or variable not in right['bindings']:
            return False
        equal = left['bindings'][variable] == right['bindings'][variable]
        if equal != (relation['kind'] == 'same_binding'):
            return False
    return True


def execute_proof_query(plan, table, *, max_results=256):
    """Execute only structural constraints and preserve a safe fallback.

    The output is a candidate frontier, not a semantic answer.  A caller must
    verify that the Host grounding result is covered before replacing it.
    """
    if plan.get('schema') != SCHEMA or table.get('schema') != CANDIDATE_SCHEMA:
        raise ValueError('proof query and candidate table schemas must match the registered versions')
    if type(max_results) is not int or max_results < 1:
        raise ValueError('max_results must be a positive integer')
    demands = [item['demand_id'] for item in plan['demands']]
    relations = plan['relations']
    domains = defaultdict(list)
    rejected = defaultdict(int)
    demand_scopes = {item['demand_id']: item.get('evidence_scope', {})
                     for item in plan['demands']}
    for row in table['rows']:
        if row['demand_id'] not in demands:
            rejected['unknown_demand'] += 1
            continue
        required_kind = demand_scopes[row['demand_id']].get('kind')
        if (required_kind is not None
                and row.get('evidence_metadata', {}).get('kind') != required_kind):
            rejected['evidence_scope_kind_mismatch'] += 1
            continue
        domains[row['demand_id']].append(row)
    missing = [demand_id for demand_id in demands if not domains[demand_id]]
    selected = []
    frontier_truncated = False
    if not missing:
        order = sorted(demands, key=lambda demand_id: (len(domains[demand_id]), demand_id))
        frontier = [({}, [])]
        for demand_id in order:
            next_frontier = []
            for chosen, rows in frontier:
                for row in sorted(domains[demand_id], key=lambda item: (-item['score'], item['candidate_id'])):
                    if _compatible(row, chosen, relations):
                        next_frontier.append(({**chosen, demand_id: row}, rows + [row]))
            frontier_truncated = frontier_truncated or len(next_frontier) > max_results
            frontier = next_frontier[:max_results]
            if not frontier:
                break
        for _, rows in frontier:
            selected.append(rows)
    unsatisfied_constraints = []
    if not selected and not missing:
        missing = list(demands)
        unsatisfied_constraints = [relation['relation_id'] for relation in relations]
    selected_rows = [row for rows in selected for row in rows]
    candidate_ids = sorted({row['candidate_id'] for row in selected_rows})
    base_record_ids = sorted({row['base_record_id'] for row in selected_rows})
    object_ids = sorted({object_id for row in selected_rows for object_id in row['object_ids']})
    return {
        'schema': 'proof-query-result-v1',
        'plan_sha256': plan['plan_sha256'],
        'input_candidate_count': len(table['rows']),
        'eligible_candidate_count': len(candidate_ids),
        'eligible_candidate_ids': candidate_ids,
        'eligible_base_record_ids': base_record_ids,
        'proof_tuple_count': len(selected),
        'frontier_truncated': frontier_truncated,
        'object_ids': object_ids,
        'missing_demands': missing,
        'unsatisfied_constraints': unsatisfied_constraints,
        'rejection_counts': dict(sorted(rejected.items())),
        'structural_contract_complete': not missing and bool(selected),
        'semantic_claims': False,
        'safe_to_replace_host_grounding': False,
        'fallback_reason': 'host_semantic_grounding_required',
    }


def audit_host_grounding_coverage(result, grounded_witnesses):
    """Check whether a Host witness result is covered by the query frontier."""
    host_ids = sorted({record_id
                       for witness in grounded_witnesses
                       for record_id in (witness.get('retrieval_record_ids')
                                         or witness.get('record_ids', []))})
    eligible = set(result.get('eligible_base_record_ids', []))
    missing = [identity for identity in host_ids if identity not in eligible]
    covered = not missing
    return {
        'host_witness_count': len(host_ids),
        'host_witness_ids': host_ids,
        'missing_host_witness_ids': missing,
        'coverage': covered,
        'safe_to_replace_host_grounding': False,
        'reason': ('host_witness_inside_structural_frontier_but_semantics_unproven'
                   if covered else 'host_witness_outside_frontier'),
    }
