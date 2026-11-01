"""Content-addressed, session-local embedding producers independent of cache misses."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import time


def stable_producer_batches(edges, atoms_by_id, batch_size=32):
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('positive producer batch size required')
    if len({edge['edge_id'] for edge in edges}) != len(edges):
        raise ValueError('unique edge identities required')
    sessions = defaultdict(list)
    for edge in edges:
        members = [atoms_by_id[identity] for identity in edge['atom_ids']]
        owners = {atom['session_id'] for atom in members}
        if len(owners) != 1 or not edge['embedding_text']:
            raise ValueError('nonempty session-local observable producer required')
        sessions[next(iter(owners))].append(edge)
    batches = []
    for session_id in sorted(sessions):
        ordered = sorted(sessions[session_id], key=lambda edge: (
            tuple(atoms_by_id[identity]['ordinal'] for identity in edge['atom_ids']), edge['edge_id']))
        batches.extend(ordered[begin:begin + batch_size] for begin in range(0, len(ordered), batch_size))
    return batches


def producer_identity(batch, execution_identity):
    if not execution_identity or not batch:
        raise ValueError('nonempty immutable execution identity and producer batch required')
    payload = {'execution_identity': execution_identity,
        'ordered_operands': [(edge['edge_id'], edge['embedding_text']) for edge in batch],
        'contract': 'session-local-canonical-batch-v1'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_producers(edges, atoms_by_id, store, encode, execution_identity, batch_size=32,
                      output_digest=None, request_index=None):
    vectors, records = {}, []
    for batch in stable_producer_batches(edges, atoms_by_id, batch_size):
        identity = producer_identity(batch, execution_identity)
        operands = [{**edge, 'edge_id': edge['edge_id'] + ':producer:' + identity} for edge in batch]
        lookup_started = time.perf_counter()
        cached = [store.get(operand) if store is not None else None for operand in operands]
        lookup_seconds = time.perf_counter() - lookup_started
        reused = all(value is not None for value in cached)
        encode_seconds = 0.0
        write_seconds = 0.0
        if reused:
            values = cached
        else:
            encode_started = time.perf_counter()
            values = list(encode([edge['embedding_text'] for edge in batch]))
            encode_seconds = time.perf_counter() - encode_started
            if len(values) != len(batch):
                raise ValueError('producer output cardinality differs from exact ordered operands')
            if store is not None:
                write_started = time.perf_counter()
                for operand, value in zip(operands, values):
                    store.put(operand, value)
                write_seconds = time.perf_counter() - write_started
        for edge, value in zip(batch, values):
            vectors[edge['edge_id']] = value
        records.append({'producer_sha256': identity, 'edge_ids': [edge['edge_id'] for edge in batch],
                        'request_index': request_index,
                        'reuse_scope': 'prior_request' if reused else 'miss',
                        'reused': reused, 'computed_vectors': 0 if reused else len(batch),
                        'reused_vectors': len(batch) if reused else 0,
                        'partial_hits_recomputed': 0 if reused else sum(value is not None for value in cached),
                        'lookup_seconds': lookup_seconds, 'encode_seconds': encode_seconds,
                        'write_seconds': write_seconds})
        if output_digest is not None:
            records[-1]['output_sha256'] = output_digest(values)
    return vectors, records
