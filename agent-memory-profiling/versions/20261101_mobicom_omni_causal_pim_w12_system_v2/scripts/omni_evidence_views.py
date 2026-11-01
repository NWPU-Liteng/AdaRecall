"""Independent observable evidence views with unchanged exchange delivery closure."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from omni_evidence_planner import images_for_atoms


def build_evidence_views(edges, atoms_by_id):
    views = []
    for edge in edges:
        atoms = [atoms_by_id[identity] for identity in edge['atom_ids']]
        if edge['embedding_text'].strip():
            views.append({'view_id': 'text:' + edge['edge_id'], 'edge_id': edge['edge_id'],
                          'view_type': 'text', 'atom_ids': edge['atom_ids']})
        for ref in images_for_atoms(atoms):
            identity = ref.get('resolved', ref['normalized'])
            checksum = hashlib.sha256(identity.encode()).hexdigest()
            views.append({'view_id': 'image:' + edge['edge_id'] + ':' + checksum,
                          'edge_id': edge['edge_id'], 'view_type': 'image',
                          'atom_ids': edge['atom_ids'], 'image_identity': identity,
                          'source_atom_id': ref['source_atom_id'], 'image_ref': ref})
    if len({view['view_id'] for view in views}) != len(views):
        raise ValueError('unique evidence-view identities required')
    return views


def produce_evidence_views(edges, atoms_by_id, exchange_vectors, encode, load_image, *,
                           batch_size=8, progress=None):
    if batch_size < 1 or len(exchange_vectors) != len(edges):
        raise ValueError('positive batch size and aligned exchange producers required')
    started = time.perf_counter()
    views = build_evidence_views(edges, atoms_by_id)
    edge_by_id = {edge['edge_id']: edge for edge in edges}
    reusable = {edge['edge_id']: vector for edge, vector in zip(edges, exchange_vectors)
                if not images_for_atoms([atoms_by_id[identity] for identity in edge['atom_ids']])}
    produced, jobs = {}, {}
    for view in views:
        key = ('image', view['image_identity']) if view['view_type'] == 'image' else ('text', view['edge_id'])
        if view['view_type'] == 'text' and view['edge_id'] in reusable:
            produced[key] = (reusable[view['edge_id']], {'origin': 'inherited_identical_text_exchange'})
        else:
            jobs.setdefault(key, view)
    pending = list(jobs.items())
    for begin in range(0, len(pending), batch_size):
        batch = pending[begin:begin + batch_size]
        documents, source_records = [], []
        for _, view in batch:
            if view['view_type'] == 'image':
                image, checksum = load_image(view['image_ref'])
                documents.append({'text': '', 'images': [image]})
                source_records.append({'origin': 'new_single_image', 'image_sha256': checksum})
            else:
                documents.append({'text': edge_by_id[view['edge_id']]['embedding_text'], 'images': []})
                source_records.append({'origin': 'new_text_only_exchange'})
        encoded, records = encode(documents)
        if hasattr(encoded, 'detach'):
            encoded = encoded.detach().cpu().numpy()
        encoded = np.asarray(encoded)
        if len(encoded) != len(batch) or len(records) != len(batch):
            raise ValueError('missing evidence-view producer')
        for (key, _), vector, source, record in zip(batch, encoded, source_records, records):
            produced[key] = (vector, {**source, **record})
        if progress is not None:
            progress({'evidence_views_encoded': begin + len(batch), 'evidence_view_jobs': len(pending),
                      'seconds': time.perf_counter() - started})
    vectors, records = [], []
    for view in views:
        key = ('image', view['image_identity']) if view['view_type'] == 'image' else ('text', view['edge_id'])
        vector, producer = produced[key]
        vectors.append(vector)
        records.append({**view, 'producer': producer})
    payload = np.asarray(vectors, dtype=np.float32)
    if (payload.ndim != 2 or payload.shape[1] != exchange_vectors.shape[1]
            or not np.isfinite(payload).all()
            or not np.allclose(np.linalg.norm(payload, axis=1), 1., atol=1e-4)):
        raise ValueError('normalized same-encoder evidence-view vectors required')
    return records, payload, {'encoded_producers': len(pending),
                              'construction_seconds': time.perf_counter() - started}
