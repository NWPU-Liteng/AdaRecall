"""Whole-cohort integrity and mechanism screening, never post-hoc question selection."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_omni_smoke_closure import digest
from omni_prefix_opportunities import PrefixOpportunityIndex, multimodal_prefix_symbols


def analyze_graph(graph):
    events, values, allocations = graph['events'], graph['values'], graph['allocations']
    if not graph.get('context_identity'):
        raise ValueError('immutable execution context required')
    for name, records in (('event_id', events), ('value_id', values), ('allocation_id', allocations)):
        if [record[name] for record in records] != list(range(len(records))):
            raise ValueError('contiguous observed identities required')
    for value in values:
        if not 0 <= value['allocation_id'] < len(allocations):
            raise ValueError('unknown value storage identity')
        producer = value['producer_event']
        if producer is not None and not 0 <= producer < len(events):
            raise ValueError('unknown value producer')
    writers, readers, groups, linears, mutations, parameters_read = {}, {}, {}, [], set(), set()
    opaque_regions = graph.get('opaque_regions', [])
    for region in opaque_regions:
        if (not region.get('name') or not 0 <= region['start_event'] <= region['end_event'] < len(events)
                or region.get('materialized_event') is not True
                or region.get('completion_event') != region['end_event']
                or region.get('child_event_ids') != list(range(region['start_event'], region['end_event']))):
            raise ValueError('invalid opaque region boundary')
        if any(not 0 <= value < len(values)
               for value in region.get('input_values', []) + region.get('output_values', []) + region.get('mutated_values', [])):
            raise ValueError('opaque region references an unknown value')
        completion = events[region['completion_event']]
        if (completion['operator'] != 'opaque::' + region['name']
                or not completion.get('opaque_region_completion')
                or any(completion[field] != region[field] for field in ('input_values', 'output_values', 'mutated_values'))
                or not set(region['child_event_ids']).issubset(completion['dependencies'])):
            raise ValueError('opaque outputs and all child dependencies must bind completion')
    for event in events:
        event_id = event['event_id']
        if any(not 0 <= dependency < event_id for dependency in event['dependencies']):
            raise ValueError('topological dependencies required')
        for value_id in event['input_values']:
            if not 0 <= value_id < len(values):
                raise ValueError('unknown observed input')
            producer = values[value_id]['producer_event']
            if producer is not None and producer not in event['dependencies']:
                raise ValueError('actual input producer dependency missing')
            allocation = values[value_id]['allocation_id']
            if allocation in writers and writers[allocation] not in event['dependencies']:
                raise ValueError('allocation write dependency missing')
            if any(label.startswith('parameter:') for label in allocations[allocation]['external_labels']):
                parameters_read.add(allocation)
        for value_id in event['output_values'] + event['mutated_values']:
            if not 0 <= value_id < len(values) or values[value_id]['producer_event'] != event_id:
                raise ValueError('output or mutation producer differs')
        scope = event['linear_scope']
        if scope is not None and event['operator'] in {
                'aten.linear.default', 'aten.mm.default', 'aten.addmm.default', 'aten.bmm.default'}:
            activation, weight = values[scope['input_value']], values[scope['weight_value']]
            if (scope['rows'] != 1 or weight['shape'] != [scope['out_features'], scope['in_features']]
                    or weight['logical_bytes'] != scope['out_features'] * scope['in_features'] * scope['weight_element_bytes']):
                raise ValueError('actual decode linear shape or weight identity differs')
            allocation_id = weight['allocation_id']
            labels = allocations[allocation_id]['external_labels']
            if not any(label.startswith('parameter:') for label in labels):
                raise ValueError('linear weight is not a registered model parameter')
            row = {'event_id': event_id, 'name': scope['name'], 'weight_bytes': weight['logical_bytes'],
                   'input_logical_bytes': activation['logical_bytes'],
                   'parameter_aliases': labels,
                   'in_features': scope['in_features'], 'out_features': scope['out_features'],
                   'weight_element_bytes': scope['weight_element_bytes']}
            linears.append(row)
            epoch = writers.get(activation['allocation_id'])
            groups.setdefault((scope['input_value'], epoch), []).append(row)
        read_allocations = {values[value]['allocation_id'] for value in event['input_values']}
        for allocation in event['write_allocations']:
            if not 0 <= allocation < len(allocations):
                raise ValueError('unknown written allocation')
            if not readers.get(allocation, set()).issubset(set(event['dependencies'])):
                raise ValueError('write must wait for all preceding alias readers')
            writers[allocation] = event_id
            readers[allocation] = set()
            if any(label.startswith('cache.') for label in allocations[allocation]['external_labels']):
                mutations.add(allocation)
        for allocation in read_allocations - set(event['write_allocations']):
            readers.setdefault(allocation, set()).add(event_id)
    cache_allocations = [allocation for allocation in allocations
                         if any(label.startswith('cache.') for label in allocation['external_labels'])]
    qualified_groups = [group for group in groups.values() if len(group) > 1]
    return {
        'events': len(events), 'dependency_edges': sum(len(event['dependencies']) for event in events),
        **({'opaque_regions': len(opaque_regions),
            'materialized_opaque_regions': sum(region['materialized_event'] for region in opaque_regions)}
           if 'opaque_regions' in graph else {}),
        'operator_counts': dict(Counter(event['operator'] for event in events)),
        'observed_linear_calls': len(linears), 'linear_weight_logical_read_bytes_per_step': sum(
            row['weight_bytes'] for row in linears),
        'registered_cache_storage_bytes': sum(row['storage_bytes'] for row in cache_allocations),
        'visible_mutated_cache_allocations': len(mutations),
        'visible_mutated_cache_storage_bytes': sum(allocations[index]['storage_bytes'] for index in mutations),
        'parameter_storage_bytes_touched': sum(allocations[index]['storage_bytes'] for index in parameters_read),
        'shared_linear_input_groups_with_unchanged_write_epoch': qualified_groups,
        'linear_events': linears,
        'seeded_nondeterministic_ops': sum(event['nondeterministic_seeded'] for event in events),
        'boundary': 'tensor/storage footprint and ATen-visible hazards only; not measured DRAM transactions, bank locality, kernel timing or fused-kernel qualification; custom kernels may be opaque',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.run_root / 'execution_plan.json'
    plan = json.loads(plan_path.read_text())
    status_path = args.run_root / 'execution_status.json'
    status = json.loads(status_path.read_text())
    if not status.get('all_workers_succeeded') or status['execution_plan_sha256'] != digest(plan_path):
        raise ValueError('whole fixed cohort must finish before audit')
    for name, expected in plan['source_sha256'].items():
        if digest(args.run_root / 'sources' / name) != expected:
            raise ValueError('registered source snapshot differs')
    config = json.loads((Path(plan['reader']) / 'config.json').read_text())
    if digest(Path(plan['reader']) / 'config.json') != plan['checkpoint_sha256']['config.json']:
        raise ValueError('model input identity differs')
    cohort_path = Path(plan['cohort_plan'])
    if digest(cohort_path) != plan['cohort_plan_sha256']:
        raise ValueError('original cohort changed')
    cohort = json.loads(cohort_path.read_text())
    if [(job['person_id'], job['question_ids']) for job in plan['jobs']] != [
            (job['person_id'], job['question_ids']) for job in cohort['jobs']]:
        raise ValueError('execution must include every registered request without reordering')
    users, result_hashes, graph_counts = [], {}, Counter()
    for job in plan['jobs']:
        path = args.run_root / ('user_' + job['person_id']) / 'result.json'
        result = json.loads(path.read_text())
        smoke_path = Path(job['smoke'])
        if digest(smoke_path) != job['smoke_sha256']:
            raise ValueError('original serving outputs changed')
        smoke = json.loads(smoke_path.read_text())
        if (not result.get('complete') or result.get('evaluation_label_files_read') != []
                or result['execution_plan_sha256'] != digest(plan_path)
                or result['producer_smoke_sha256'] != job['smoke_sha256']
                or [row['question_id'] for row in result['rows']] != job['question_ids']):
            raise ValueError('complete per-user natural order required')
        result_hashes[str(path)] = digest(path)
        index = PrefixOpportunityIndex()
        rows = []
        for row, original in zip(result['rows'], smoke['rows']):
            input_path, graph_path = Path(row['input']), Path(row['graph'])
            if digest(input_path) != row['input_sha256'] or digest(graph_path) != row['graph_sha256']:
                raise ValueError('input/graph producer hashes differ')
            inputs, graph = json.loads(input_path.read_text()), json.loads(graph_path.read_text())
            if (inputs['question_id'] != original['question_id']
                    or len(inputs['input_token_ids']) != original['input_tokens']
                    or inputs['input_identity'] != graph['context_identity']
                    or inputs['accepted_atom_ids'] != original['hypergraph_evidence_plan']['accepted_atom_ids']
                    or inputs['selected_hyperedges'] != original['hypergraph_evidence_plan']['selected_hyperedges']):
                raise ValueError('actual graph must bind the frozen evidence and input')
            if 'scripts/omni_execution_input_contract.py' in plan['source_sha256']:
                expected_contract = {'processor_inputs_sha256': original['processor_inputs_sha256'],
                    'images': original['images'], 'embedding_producer_outputs': [
                        {name: record[name] for name in ('producer_sha256', 'output_sha256')}
                        for record in original['embedding_producers']],
                    'independent_image_materialization_executed': False,
                    'reader_state_cross_request_reuse_executed': False}
                if inputs.get('frozen_serving_input_contract') != expected_contract:
                    raise ValueError('full serving processor/evidence producer binding differs')
            merge = config['vision_config']['spatial_merge_size']
            symbols = multimodal_prefix_symbols(inputs['input_token_ids'], config['image_token_id'],
                [grid[0] * grid[1] * grid[2] // merge**2 for grid in inputs['image_grid_thw']],
                inputs['ordered_batch_identity'])
            prefix = index.observe(row['question_id'], symbols)
            if prefix != row['past_only_prefix_opportunity'] or prefix != inputs['past_only_prefix_opportunity']:
                raise ValueError('prefix opportunities require past-only whole-stream recomputation')
            checks = {'token_ids_equal': row['answer_token_ids'] == original['answer_token_ids'],
                      'stop_reason_equal': row['stop_reason'] == original['stop_reason']}
            analyzed = analyze_graph(graph)
            graph_counts.update(analyzed['operator_counts'])
            rows.append({'question_id': row['question_id'], **checks,
                'prefix_opportunity': prefix, 'input_tokens': len(symbols),
                'graph_analysis': analyzed, 'graph_sha256': row['graph_sha256'],
                'input_sha256': row['input_sha256']})
        users.append({'person_id': job['person_id'], 'requests': len(rows), 'rows': rows})
    all_rows = [row for user in users for row in user['rows']]
    result = {'schema_version': 'omni-whole-execution-cohort-audit-v1', 'complete': True,
        'execution_plan_sha256': digest(plan_path), 'execution_status_sha256': digest(status_path),
        'user_result_sha256': result_hashes, 'requests': len(all_rows), 'users': users,
        'complete_greedy_token_matches': sum(row['token_ids_equal'] for row in all_rows),
        'stop_reason_matches': sum(row['stop_reason_equal'] for row in all_rows),
        'traced_requests': sum(row['graph_analysis']['events'] > 0 for row in all_rows),
        'operator_counts_across_first_decode_steps': dict(graph_counts),
        'past_only_matched_prefix_tokens': sum(row['prefix_opportunity']['matched_prefix_tokens'] for row in all_rows),
        'total_input_tokens': sum(row['input_tokens'] for row in all_rows),
        'evaluation_label_files_read': [], 'new_quality_evaluated': False,
        'actual_pim_routing_executed': False, 'cross_request_reader_state_restored': False,
        'end_to_end_speedup': None, 'publication_ready': False,
        'script_sha256': digest(Path(__file__)),
        'boundary': 'all fixed requests retained; actual original-image greedy replay, first-decode-step ATen-visible DAG, past-only input overlap and mechanism screening; no fused PIM, general KV equivalence, full-system scheduling or end-to-end acceleration claim'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('requests', 'complete_greedy_token_matches',
        'stop_reason_matches', 'traced_requests', 'past_only_matched_prefix_tokens', 'total_input_tokens')}, indent=2))


if __name__ == '__main__':
    main()
