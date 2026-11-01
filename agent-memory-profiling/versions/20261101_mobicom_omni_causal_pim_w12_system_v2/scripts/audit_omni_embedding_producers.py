"""Replay label-free hyperedge producer provenance and bounded-store lifecycle."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

from audit_omni_smoke_closure import verify_source_snapshot
from omni_embedding_producer import producer_identity, stable_producer_batches
from omni_evidence_planner import build_local_hyperedges
from run_omni_registered_cohort import digest


def audit_producers(run, graph, vector_bytes):
    if (not run['complete'] or run['evaluation_label_files_read'] != [] or vector_bytes < 1
            or graph['person_id'] != run['rows'][0]['person_id']):
        raise ValueError('complete label-free stream and original user graph required')
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    sessions = {}
    for atom in graph['atoms']:
        sessions.setdefault(atom['session_id'], []).append(atom)
    capacity = run['embedding_state_reuse']['capacity_bytes']
    enabled = run['embedding_state_reuse_enabled']
    entries, output_hashes = OrderedDict(), {}
    resident = peak = hits = misses = evictions = rejected = 0
    reused_vectors = computed_vectors = reused_producers = producers = 0
    for row in run['rows']:
        edges = build_local_hyperedges(row['retrieved_session_ids'], sessions, radius=2)
        rejected_ids = row['hypergraph_evidence_plan']['embedding_over_budget_edge_ids']
        if not set(rejected_ids).issubset({edge['edge_id'] for edge in edges}):
            raise ValueError('unknown rejected embedding operand')
        edges = [edge for edge in edges if edge['edge_id'] not in rejected_ids]
        batches = stable_producer_batches(edges, atoms, 32)
        records = row['embedding_producers']
        if (records != row['hypergraph_evidence_plan']['embedding_producers']
                or len(batches) != len(records)):
            raise ValueError('complete original producer batches required')
        for batch, record in zip(batches, records):
            identity = producer_identity(batch, run['embedding_execution_identity'])
            if (record['producer_sha256'] != identity
                    or record['edge_ids'] != [edge['edge_id'] for edge in batch]
                    or len(record['output_sha256']) != 64):
                raise ValueError('producer identity must bind exact original ordered operands')
            output_hash = record['output_sha256']
            if identity in output_hashes and output_hashes[identity] != output_hash:
                raise ValueError('same immutable producer changed its output bytes')
            keys = [(edge['edge_id'], identity) for edge in batch]
            present = []
            for key in keys:
                found = enabled and key in entries
                present.append(found)
                if enabled:
                    hits += found
                    misses += not found
                    if found:
                        entries.move_to_end(key)
            reused = all(present)
            if (record['reused'] != reused or record['reused_vectors'] != (len(batch) if reused else 0)
                    or record['computed_vectors'] != (0 if reused else len(batch))
                    or record['partial_hits_recomputed'] != (0 if reused else sum(present))):
                raise ValueError('hit must consume an earlier resident full producer; partial hits recompute')
            if reused and identity not in output_hashes:
                raise ValueError('reuse cannot precede producer execution')
            if not reused and enabled:
                for key in keys:
                    if vector_bytes > capacity:
                        rejected += 1
                        continue
                    if key in entries:
                        entries.pop(key)
                        resident -= vector_bytes
                    while entries and resident + vector_bytes > capacity:
                        entries.popitem(last=False)
                        resident -= vector_bytes
                        evictions += 1
                    entries[key] = True
                    resident += vector_bytes
                    peak = max(peak, resident)
            output_hashes[identity] = output_hash
            reused_vectors += record['reused_vectors']
            computed_vectors += record['computed_vectors']
            reused_producers += reused
            producers += 1
    expected = {'capacity_bytes': capacity, 'resident_bytes': resident, 'peak_resident_bytes': peak,
        'hits': hits, 'misses': misses, 'evictions': evictions, 'rejected': rejected}
    if run['embedding_state_reuse'] != expected:
        raise ValueError('bounded-store counters, occupancy or lifecycle differ from replay')
    return {'requests': len(run['rows']), 'producer_invocations': producers,
        'reused_producers': reused_producers, 'reused_vectors': reused_vectors,
        'computed_vectors': computed_vectors, 'store_lifecycle': expected,
        'producer_provenance_integrity': 'pass'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh producer audit required')
    plan_path = args.run_root / 'cohort_plan.json'
    plan = json.loads(plan_path.read_text())
    graph_path = args.prepared / 'omni_graph.json'
    audit_path = args.prepared / 'audit.json'
    audit = json.loads(audit_path.read_text())
    if (digest(audit_path) != plan['prepared_audit_sha256']
            or digest(graph_path) != audit['artifacts_sha256']['omni_graph.json']):
        raise ValueError('original serving graph identity changed')
    graphs = {graph['person_id']: graph for graph in json.loads(graph_path.read_text())}
    provenance = {str(path): digest(path) for path in (plan_path, graph_path, audit_path)}
    results = []
    for job in plan['jobs']:
        path = Path(job['output'])
        run = json.loads(path.read_text())
        registration_path = path.with_suffix('.registration.json')
        registration = json.loads(registration_path.read_text())
        if (not verify_source_snapshot(run, registration)
                or [row['question_id'] for row in run['rows']] != job['question_ids']):
            raise ValueError('frozen complete producer source/order required')
        config_path = Path(run['embedder']) / 'config.json'
        if digest(config_path) != run['embedder_config_sha256']:
            raise ValueError('embedding output dimension source changed')
        dimension = json.loads(config_path.read_text())['hidden_size']
        results.append({'person_id': job['person_id'],
            **audit_producers(run, graphs[job['person_id']], dimension * 4)})
        for source in (path, registration_path, config_path):
            provenance[str(source)] = digest(source)
    result = {'schema_version': 'omni-embedding-producer-audit-v1', 'complete': True,
        'producer_provenance_integrity': 'pass', 'per_user': results,
        **{field: sum(user[field] for user in results) for field in (
            'requests', 'producer_invocations', 'reused_producers', 'reused_vectors', 'computed_vectors')},
        'inputs_sha256': provenance, 'evaluation_label_files_read': [],
        'source_files_sha256': {str(path): digest(path) for path in (
            Path(__file__), Path(__file__).with_name('omni_embedding_producer.py'))},
        'boundary': 'source-bound graph operands, output digests and bounded-store execution replay; no saved tensor payload numerical reexecution or performance/novelty qualification'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('requests', 'producer_invocations',
        'reused_producers', 'reused_vectors', 'computed_vectors', 'producer_provenance_integrity')}))


if __name__ == '__main__':
    main()
