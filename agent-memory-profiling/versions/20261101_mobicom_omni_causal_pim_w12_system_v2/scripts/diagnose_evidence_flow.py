"""Offline label-side attribution of evidence loss across the serving pipeline."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recall(stage: set[str], gold: set[str]) -> float | None:
    return len(stage & gold) / len(gold) if gold else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh diagnostic output required')
    labels = {row['question_id']: row for row in read_jsonl(args.prepared / 'labels.jsonl')}
    manifests = {row['question_id']: row for row in read_jsonl(args.prepared / 'manifests.jsonl')}
    graphs = json.loads((args.prepared / 'mobilemem_graph.json').read_text())
    event_times = {node['event_id']: node['timestamp'] for graph in graphs for node in graph['nodes']}
    result = json.loads(args.result.read_text())
    if not result.get('complete'):
        raise ValueError('cannot diagnose incomplete run')
    expected_ids = result['selected_question_ids']
    if (len(set(expected_ids)) != len(expected_ids)
            or [row['question_id'] for row in result['rows']] != expected_ids):
        raise ValueError('missing, duplicate or reordered results')
    rows, losses, closure_trace_complete = [], Counter(), True
    for row in result['rows']:
        question_id = row['question_id']
        label, manifest = labels[question_id], manifests[question_id]
        if manifest['split'] != 'development':
            raise ValueError('diagnostic is development-only')
        gold = set(label.get('source_evidence_ids', []))
        retrieval = row.get('retrieval_audit', {})
        candidate = set(retrieval.get('candidate_events', []))
        primary = set(retrieval.get('primary_events', []))
        bundles = row.get('trajectory_hypergraph', {}).get('bundles', [])
        if bundles and all('bundle_event_ids' in bundle for bundle in bundles):
            closure = set().union(*(set(bundle['bundle_event_ids']) for bundle in bundles))
        else:
            closure_trace_complete = False
            closure = set()
        selected = {atom['event_id'] for atom in row.get('selected_atoms', [])}
        visible_gold = {event_id for event_id in gold
                        if event_times.get(event_id, '9999') <= manifest['effective_timestamp']}
        negative_or_absence = (label.get('question_type') == 'adversarial'
                               or any(term in ' '.join(label.get('golden_answers', [])).lower()
                                      for term in ('无法', '未提及', 'cannot', 'unknown', 'not available')))
        stages = {'temporally_visible': visible_gold, 'retrieval_candidates': candidate & gold,
                  'rerank_primary': primary & gold}
        closure_available = bool(bundles and all('bundle_event_ids' in bundle for bundle in bundles))
        if closure_available:
            stages['hypergraph_bundles'] = closure & gold
        stages['union_materialized'] = selected & gold
        stage_recall = {name: recall(events, gold) for name, events in stages.items()}
        transitions = []
        for previous, current in zip(stages, list(stages)[1:]):
            dropped = stages[previous] - stages[current]
            recovered = stages[current] - stages[previous]
            if dropped:
                losses[f'{previous}->{current}'] += len(dropped)
            transitions.append({'from': previous, 'to': current, 'dropped': sorted(dropped),
                                'recovered': sorted(recovered),
                                'intermediate_stages_unobserved':
                                previous == 'rerank_primary' and current == 'union_materialized'})
        rows.append({'question_id': question_id, 'question': manifest['question'],
                     'question_type': label.get('question_type'),
                     'evidence_semantics': 'absence_or_adversarial; do not interpret positive-source recall as answerability'
                     if negative_or_absence else 'positive_source_witnesses',
                     'gold_source_events': sorted(gold),
                     'stages': {name: sorted(events) for name, events in stages.items()},
                     'stage_recall': stage_recall,
                     'transitions': transitions,
                     'final_missing_sources': sorted(gold - selected),
                     'first_incomplete_source_stage': next((name for name, events in stages.items()
                                                           if gold - events), None),
                     'closure_trace_available': closure_available,
                     'answer_support_assessed': False})
    report = {'rows': rows, 'stage_loss_counts': dict(losses),
              'question_count': len(rows),
              'status_counts': dict(Counter(row.get('status') for row in result['rows'])),
              'provenance': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (args.result, args.prepared / 'manifests.jsonl',
                                          args.prepared / 'labels.jsonl',
                                          args.prepared / 'mobilemem_graph.json', Path(__file__))},
              'closure_trace_complete_for_all_rows': closure_trace_complete,
              'rerank_boundary': 'reranker scores all retrieved candidates; only primary cutoff removes candidates',
              'stage_boundary': 'query semantic parsing, span-level entailment, and final answer support remain unassessed',
              'answer_support_assessed': False,
              'boundary': 'offline label-side diagnostic only; labels never enter serving; source-event presence is not logical support, and absence questions require separate global-coverage/abstention review'}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'rows': len(rows), 'stage_loss_counts': dict(losses),
                      'closure_trace_complete_for_all_rows': closure_trace_complete,
                      'output': str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
