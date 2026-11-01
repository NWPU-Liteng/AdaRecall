"""Evidence-consumer lifetimes and physical memory frontier for one request."""
from __future__ import annotations


def compile_lifecycle(objects, object_ids, *, reader_input_tokens, image_bytes):
    selected = [objects[identity] for identity in object_ids]
    states = []
    for item in selected:
        if item['kind'] == 'text':
            resident = len(item['text'].encode('utf-8'))
        else:
            resident = int(image_bytes[item['source_sha256']])
        states.append({'state_id': item['object_id'], 'kind': item['kind'],
                       'resident_bytes_estimate': resident, 'producer': item.get('source_sha256'),
                       'consumers': ['reader_prefill'], 'last_consumer': 'reader_prefill',
                       'release_after': 'reader_prefill', 'recompute': 'not_used',
                       'identity_bound': True})
    frontier = sum(item['resident_bytes_estimate'] for item in states)
    events = [{'phase': 'materialize', 'consumer': 'reader_prefill',
               'state_ids': [item['state_id'] for item in states], 'resident_bytes': frontier,
               'reader_input_tokens': reader_input_tokens,
               'accounting': 'UTF-8 source text and decoded RGB raster logical bytes'},
              {'phase': 'release', 'consumer': 'none', 'state_ids': [item['state_id'] for item in states],
               'resident_bytes': 0}]
    return {'schema': 'proof-lifecycle-v1', 'states': states,
            'reader_input_tokens': int(reader_input_tokens), 'events': events,
            'frontier_bytes': frontier, 'peak_frontier_bytes': frontier,
            'consumers': {'reader_prefill': [item['state_id'] for item in states]},
            'future_consumer_oracle': False,
            'boundary': 'logical source materialization bytes, not total process RSS or GPU working set; no Reader KV or cross-request state reuse'}
