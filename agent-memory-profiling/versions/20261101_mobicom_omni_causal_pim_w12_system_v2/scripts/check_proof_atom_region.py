"""Label-free structural audit of the staged atom/region candidate."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from omni_proof_grounding import expand_compact_sources, validate_witnesses
from omni_proof_join import _compatible
from omni_proof_planner import parse_query_graph
from omni_proof_query_contract import parse_query_contract, render_operation_plan, retrieval_views
from omni_proof_runtime import file_digest, write_json
from omni_proof_sources import materialize


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_run(run, expected, records, objects, queries, protocol):
    require(run.get('complete') and run['question_ids'] == expected, 'incomplete or changed request list')
    require([row['question_id'] for row in run['rows']] == expected, 'changed row denominator/order')
    require(not run['evaluation_label_files_read'] and not run['validation_accessed']
            and not run['holdout_accessed'], 'serving evaluation isolation failed')
    by_record = {record['record_id']: record for record in records}
    summaries = []
    for row in run['rows']:
        plan, question_id = row['proof_plan'], row['question_id']
        require(not row['evaluation_label_files_read'], 'answer labels entered serving')
        contract = None
        if 'query_contract' in plan:
            contract = parse_query_contract(plan['query_parse']['parsed'], queries[question_id])
            demands, relations, status = contract['source_facts'], contract['relations'], contract['status']
            require(contract == plan['query_contract'], 'query contract differs from parsed output')
            require(plan['query_operations'] == contract['operations']
                    and plan['reader_operation_plan'] == render_operation_plan(contract)
                    and not plan['operations_require_source_witnesses'], 'derivation/source boundary mismatch')
            require(plan['retrieval']['views'] == retrieval_views(queries[question_id], contract)
                    and plan['retrieval']['original_question_preserved'], 'original retrieval context lost')
            require(not ({item['operation_id'] for item in contract['operations']}
                         & set(plan['required_demand_ids'])), 'derived result became a source demand')
            if contract['answer_mode'] == 'contextual':
                require(plan['selection_mode'] == 'nonproof_retrieval_package', 'contextual query claimed proof')
        else:
            demands, relations, status = parse_query_graph(plan['query_parse']['parsed'], queries[question_id])
        require(status == plan['query_parse']['status'] and demands == plan['demands']
                and relations == plan['relations'], 'query graph differs from validated model output')
        aliases = {alias: {**by_record[identity], 'record_id': alias}
                   for alias, identity in plan['grounding']['alias_to_record_id'].items()}
        grounded, rejected = validate_witnesses(expand_compact_sources(
            plan['grounding']['parsed'], aliases, objects), demands, relations, aliases, objects)
        require(rejected == plan['grounding']['rejections'], 'source rejection audit differs')
        by_witness = {witness['witness_id']: witness for witness in grounded}
        require(set(by_witness) == {witness['witness_id'] for witness
                in plan['proof_hypergraph']['witnesses']}, 'grounded witness identity mismatch')
        if plan['selection_mode'] == 'source_located_proof_join':
            require(status == 'ok' and not plan.get('strict_proof_failure'), 'invalid graph accepted as proof')
            selected = plan['selected_witnesses']
            require({witness['demand_id'] for witness in selected} == {demand['demand_id'] for demand in demands},
                    'required conjunct was dropped')
            chosen = {}
            for witness in selected:
                require(witness['witness_id'] in by_witness and
                        all(witness[field] == by_witness[witness['witness_id']][field]
                            for field in ('object_ids', 'anchors', 'bindings', 'source_relations', 'source_hints')),
                        'selected witness bypasses source validator')
                require(_compatible(witness, chosen, relations), 'selected witnesses violate a source binding/relation')
                chosen[witness['demand_id']] = witness
            identities = {identity for witness in selected for identity in witness['object_ids']}
        else:
            require(plan['selection_mode'] == 'nonproof_retrieval_package'
                    and plan.get('strict_proof_failure') and not plan['selected_witnesses'],
                    'retrieval fallback masquerades as a witness join')
            identities = set()
            for record in plan['selected_records']:
                original = by_record[record['record_id']]
                require(set(record['object_ids']) == set(original['object_ids']
                        + original.get('retrieval_context_object_ids', [])), 'nonproof sources changed')
                identities.update(record['object_ids'])
        require(plan['semantic_proof'] is False, 'source location overstated as semantic proof')
        require(sorted(identities) == plan['accepted_object_ids'] == row['materialized_object_ids'],
                'unique physical object accounting failed')
        rendered = materialize(objects, identities)
        require(rendered['text'] == plan['text'], 'Reader text differs from original source objects')
        require([image['image_sha256'] for image in rendered['images']] == plan['final_render']['image_sha256s'],
                'Reader image objects differ from selected original images')
        require(len(set(plan['final_render']['image_sha256s'])) == len(row['images']),
                'image charged or materialized more than once')
        require(row['input_tokens'] == plan['final_render']['input_tokens']
                == plan['assessment']['exact_input_tokens'] <= protocol['input_tokens'], 'processor budget mismatch')
        require(plan['final_render']['text_tokens'] <= protocol['text_tokens']
                and len(row['images']) <= protocol['max_images'], 'text or image budget mismatch')
        require(not row['backend_plan']['physical_execution'], 'unqualified PIM entered execution')
        summaries.append({'question_id': question_id, 'parse_status': status,
            'answer_mode': contract['answer_mode'] if contract else None,
            'operations': len(contract['operations']) if contract else 0,
            'operation_kinds': [item['kind'] for item in contract['operations']] if contract else [],
            'required_demands': sum(demand['required'] for demand in demands), 'demands': len(demands),
            'relations': len(relations), 'source_located_witnesses': len(grounded),
            'selection_mode': plan['selection_mode'], 'unique_objects': len(identities),
            'selected_source_records': len({record_id for witness in plan['selected_witnesses']
                                           for record_id in witness['retrieval_record_ids']}),
            'retrieval_package_records': len(plan['selected_records']) if plan.get('strict_proof_failure') else 0,
            'input_tokens': row['input_tokens'], 'images': len(row['images']),
            'parse_seconds': plan['query_parse']['seconds'], 'grounding_seconds': plan['grounding']['seconds']})
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--stage', choices=['smoke', 'full'], required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root / 'protocol.json').read_text())
    for relative, expected in protocol['source_files_sha256'].items():
        require(file_digest(args.root / 'runtime_mainline' / relative) == expected, 'frozen source changed: ' + relative)
    cohort = json.loads((args.root / 'cohort.json').read_text())
    manifests = [json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    queries = {row['question_id']: row['question'] for row in manifests}
    rows, run_hashes = [], {}
    for user, identities in cohort['question_ids_by_user'].items():
        path = args.root / (f'smoke_user_{user}.json' if args.stage == 'smoke' else f'proof_user_{user}.json')
        expected = identities[:2] if args.stage == 'smoke' else identities
        index = args.root / f'proof_index_user_{user}'
        records = json.loads((index / 'records.json').read_text())
        objects = json.loads((index / 'source.json').read_text())['objects']
        rows.extend(check_run(json.loads(path.read_text()), expected, records, objects, queries, protocol))
        run_hashes[path.name] = file_digest(path)
    summary = {'complete': True, 'structural_checks_passed': True, 'stage': args.stage,
        'requests': len(rows), 'rows': rows, 'run_sha256': run_hashes,
        'required_demand_count_histogram': dict(Counter(row['required_demands'] for row in rows)),
        'selection_mode_histogram': dict(Counter(row['selection_mode'] for row in rows)),
        'source_located_multisource_joins': sum(row['selected_source_records'] > 1 for row in rows),
        'query_relations': sum(row['relations'] for row in rows),
        'quality_measured': False, 'evaluation_label_files_read': [],
        'main_experiment_closed': False,
        'boundary': 'source identity, logical structure and exact budget audit only; not QA sufficiency, method gain or semantic entailment'}
    write_json(args.root / f'{args.stage}_structural_audit.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
