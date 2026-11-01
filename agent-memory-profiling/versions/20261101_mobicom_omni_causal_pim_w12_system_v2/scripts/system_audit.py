"""Fail-closed structural checks for the integrated memory execution path."""
from __future__ import annotations


SCHEMA = 'observable-materialization-v1'


def reference_execution_valid(row):
    validation = row.get('prefix_validation', {})
    reuse = row.get('prefix_state_reuse', {})
    stage = row.get('execution_stages', {}).get('first', {})
    identity = row.get('submitted_output_token_ids_sha256')
    return (validation.get('status') == 'reference_execution'
            and validation.get('reference_path_executed') is True
            and validation.get('candidate_reuse_executed') is False
            and reuse.get('execution_contract') == 'unsplit_cold_reference_v1'
            and reuse.get('prefill_execution') == 'unsplit_reference'
            and reuse.get('status') in {'full_recompute', 'checkpoint_created', 'checkpoint_not_admitted', 'capacity_recompute'}
            and reuse.get('reused_tokens') == 0
            and stage.get('causal_kv_reused_tokens') == 0
            and stage.get('prompt_tokens', 0) > 0
            and stage.get('residual_prefill_tokens') == stage.get('prompt_tokens')
            and row.get('numeric_audit_scheduled') is False
            and isinstance(identity, str) and len(identity) == 64
            and validation.get('output_token_ids_sha256') == identity)


def materialization_valid(row, budget):
    try:
        plan = row['evidence_plan']
        atoms = set(plan['materialized'])
        closures = plan['unique_closures']
        accepted = plan['accepted_closure_indices']
        union = set().union(*(set(closures[index]) for index in accepted))
        costs = plan['atom_costs']
        references = {str(index): sum(index in closures[closure] for closure in accepted) for index in union}
        identities = plan['atom_identities']
        chosen = row['selected_atoms']
        utility_valid = plan['absolute_gap'] == plan['upper_bound'] - plan['objective'] >= 0
        if 'objective_units' in plan:
            denominator = plan['utility_denominator']
            units = plan['closure_utility_units']
            scores = plan['input_evidence_scores']
            mapping = plan['input_to_unique_closure']
            if denominator <= 0 or len(scores) != len(mapping) or len(units) != len(closures):
                return False
            reconstructed = [0] * len(closures)
            for index, closure in enumerate(mapping):
                if not 0 <= scores[index] <= 1:
                    return False
                numerator, divisor = float(scores[index]).as_integer_ratio()
                if denominator % divisor:
                    return False
                reconstructed[closure] = max(reconstructed[closure], numerator * (denominator // divisor))
            utility_valid = (denominator > 0 and units == reconstructed
                             and plan['objective_units'] == sum(units[index] for index in accepted)
                             and plan['objective'] == plan['objective_units'] / denominator
                             and plan['upper_bound'] == plan['upper_bound_units'] / denominator
                             and plan['absolute_gap_units'] == plan['upper_bound_units'] - plan['objective_units'] >= 0
                             and plan['absolute_gap'] == plan['absolute_gap_units'] / denominator)
            if plan['evidence_score_source'].startswith('frozen reranker'):
                reranker_scores = row['retrieval_audit']['reranker_scores']
                primary = sorted(range(len(reranker_scores)), key=lambda index: (-reranker_scores[index], index))[:len(scores)]
                utility_valid = utility_valid and scores == [reranker_scores[index] for index in primary]
        return (atoms == union and len(chosen) == len(union)
                and {atom['identity'] for atom in chosen} == {identities[index] for index in union}
                and sum(costs[index] for index in union) == plan['used_tokens'] <= budget
                and references == plan['shared_references']
                and utility_valid
                and (not plan['optimality_certified'] or plan['search_exhausted'] and plan['absolute_gap'] == 0)
                and all(atom['timestamp'] <= row['effective_timestamp'] for atom in chosen))
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def shared_primary_valid(rows, configuration):
    if configuration.get('int8_store_contract') != 'immutable_host_int8_primary_v1' or not rows:
        return False
    identities, addresses = {}, {}
    last_cutoff = None
    representation = None
    for row in rows:
        trace = row.get('int8_primary', {})
        cutoff = row.get('effective_timestamp', '')
        dimension = trace.get('dimension', 0)
        if (row.get('status') != 'ok' or not cutoff or (last_cutoff is not None and cutoff < last_cutoff)
                or trace.get('contract') != configuration['int8_store_contract']
                or trace.get('placement') != 'host_cpu_primary_not_pim_resident'
                or trace.get('primary_values_reference_exact') is not True
                or trace.get('physical_bank_int32_exact') is not True
                or not dimension or dimension % 32):
            return False
        current_representation = trace.get('representation_sha256', '')
        if (len(current_representation) != 64
                or (representation is not None and current_representation != representation)):
            return False
        representation = current_representation
        produced = trace.get('producer_entries', [])
        for entry in produced:
            identity, ordinal = entry.get('identity'), entry.get('ordinal', -1)
            if (not identity or identity in identities or ordinal in addresses or ordinal < 0
                    or entry.get('address') != ordinal * dimension or entry.get('bytes') != dimension
                    or entry.get('timestamp', '9999') > cutoff or len(entry.get('sha256', '')) != 64):
                return False
            identities[identity], addresses[ordinal] = entry, entry
        ordinals = trace.get('scan_ordinals', [])
        if (len(ordinals) != row.get('visible_atoms') or not ordinals or len(set(ordinals)) != len(ordinals)
                or any(ordinal not in addresses or addresses[ordinal]['timestamp'] > cutoff for ordinal in ordinals)
                or trace.get('producer_vectors') != len(produced)
                or trace.get('producer_write_bytes') != len(produced) * dimension
                or trace.get('reused_vectors') != len(ordinals) - len(produced)
                or trace.get('scan_vector_read_bytes') != len(ordinals) * dimension
                or trace.get('committed_bytes') != len(identities) * dimension
                or not trace['committed_bytes'] <= trace.get('allocated_bytes', -1) <= trace.get('capacity_bytes', -1)
                or trace.get('capacity_bytes') != configuration.get('int8_capacity_mib', 0) * 1024**2
                or trace.get('query_write_address') != 2 * 1024**3
                or trace.get('query_write_bytes') != dimension or trace.get('query_read_bytes') != dimension
                or trace.get('result_address') != 2 * 1024**3 + dimension
                or trace.get('result_payload_bytes') != min(128, len(ordinals)) * 8
                or trace.get('result_write_bytes') != ((min(128, len(ordinals)) * 8 + 31) // 32) * 32
                or trace.get('result_read_bytes') != 0
                or sum(trace.get('bank_read_transactions', {}).values()) * 32 != len(ordinals) * dimension):
            return False
        contiguous = ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))
        if trace.get('gather_copy_bytes') != (0 if contiguous else len(ordinals) * dimension):
            return False
        selected = trace.get('selected_entries', [])
        if ([entry.get('identity') for entry in selected] != [atom['identity'] for atom in row.get('selected_atoms', [])]
                or any(identities.get(entry.get('identity')) != entry or entry.get('ordinal') not in ordinals for entry in selected)):
            return False
        last_cutoff = cutoff
    return True


def system_checks(rows, configuration):
    valid = [row for row in rows if row.get('status') == 'ok']
    checks = {'single_pass_protocol': configuration.get('reader_protocol') == 'single',
              'nonempty_execution': bool(rows) and len(valid) == len(rows)}
    if 'int8_store_contract' in configuration:
        checks['immutable_host_primary_trace'] = shared_primary_valid(rows, configuration)
    checks['typed_closure_trace'] = bool(valid) and all(
        row.get('trajectory_hypergraph', {}).get('contract') ==
        'observable-closure-materialization-v1'
        and bool(row['trajectory_hypergraph'].get('bundles')) for row in valid)
    checks['search_work_bounded'] = bool(valid) and all(
        row.get('evidence_plan', {}).get('expanded_nodes', float('inf')) <=
        configuration.get('max_search_nodes', 0) for row in valid)
    checks['materialization_budget_and_identity'] = bool(valid) and all(
        materialization_valid(row, configuration.get('budget_tokens', -1)) for row in valid)
    checks['dense_only_candidates'] = bool(valid) and all(
        row.get('retrieval_audit', {}).get('candidate_source') == 'frozen dense event index'
        and row.get('retrieval_audit', {}).get('candidate_granularity') ==
        'event with all visible atom windows' for row in valid)
    checks['dense_event_budget_bound'] = bool(valid) and all(
        len(row.get('retrieval_audit', {}).get('dense_top_events', [])) <=
        configuration.get('dense_event_count', 0) for row in valid)
    checks['pim_no_future_reuse_forecast'] = bool(valid) and all(
        row.get('pim_retrieval', {}).get('policy') == 'current_known_scan_cost_only_v1'
        and row['pim_retrieval'].get('known_dag_references') == 1
        and row['pim_retrieval'].get('resident_bytes', float('inf')) <=
        row['pim_retrieval'].get('capacity_bytes', -1) for row in valid)
    checks['reranker_numeric_contract'] = configuration.get('reranker_state') == 'full' or (bool(valid) and all(
        row.get('reranker_state_reuse', {}).get('numeric_validation', {}).get('status') == 'pass'
        or (row.get('reranker_state_reuse', {}).get('numeric_validation', {}).get('fallback_to_full_scores') is True
            and row['reranker_state_reuse'].get('submitted_reused_prefix_tokens') == 0) for row in valid))
    checks['submitted_single_pass_answer'] = bool(valid) and all(
        row.get('submitted_answer_source') == 'single_pass'
        and row.get('execution_stages', {}).get('proof', {}).get('stop_reason') == 'not_executed'
        for row in valid)
    scheduled = [row for row in valid if row.get('numeric_audit_scheduled')]
    checks['scheduled_numeric_audit_complete'] = not configuration.get('cross_query_prefix') or (bool(scheduled) and all(
        row.get('prefix_validation', {}).get('status') in {'pass', 'fail'}
        and row['prefix_validation'].get('full_greedy_token_ids_equal') is not None
        and row['prefix_validation'].get('stop_reason_equal') is not None for row in scheduled))
    checks['numeric_fallback_safe'] = all(
        row.get('prefix_validation', {}).get('status') != 'fail'
        or (row.get('prefix_state_reuse', {}).get('status') == 'numeric_contract_full_recompute'
            and row.get('prefix_state_reuse', {}).get('reused_tokens', 0) == 0)
        for row in valid)
    checks['reference_execution_not_misreported_as_equivalence'] = all(
        row.get('prefix_validation', {}).get('status') != 'reference_execution'
        or reference_execution_valid(row) for row in valid)
    checks['all_actual_reuse_audited'] = bool(valid) and all(
        row.get('prefix_state_reuse', {}).get('execution_contract') != 'unsplit_cold_reference_v1'
        or row['prefix_state_reuse'].get('prefill_execution') == 'unsplit_reference'
        or (row.get('numeric_audit_scheduled') is True
            and row.get('prefix_validation', {}).get('status') in {'pass', 'fail'}) for row in valid)
    checks['state_store_bounded'] = bool(valid) and all(
        not configuration.get('cross_query_prefix') or (
            row.get('prefix_state_reuse', {}).get('store') is not None
            and row['prefix_state_reuse']['store'].get('tensor_bytes', float('inf')) <=
            row['prefix_state_reuse']['store'].get('capacity_bytes', -1)) for row in valid)
    checks['actual_reader_operator_trace'] = bool(valid) and all(
        bool(row.get('operator_trace', {}).get('operators')) for row in valid)
    checks['residual_pim_model_only'] = bool(valid) and all(
        row.get('pim_reader_residual', {}).get('hardware_output_substitution') is False
        and 'layer_types' in row['pim_reader_residual']
        and 'pool_capacity_bytes' in row['pim_reader_residual'] for row in valid)
    return checks
