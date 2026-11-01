"""Bounded, run-local reuse of immutable observable-edge embedding results."""
from collections import OrderedDict
import hashlib


class EdgeEmbeddingStore:
    def __init__(self, capacity_bytes):
        if capacity_bytes <= 0:
            raise ValueError('positive embedding-state capacity required')
        self.capacity_bytes = capacity_bytes
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected = 0
        self.entries = OrderedDict()

    def key(self, edge):
        return (edge['edge_id'], hashlib.sha256(edge['embedding_text'].encode()).hexdigest())

    def get(self, edge):
        key = self.key(edge)
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, edge, vector):
        key = self.key(edge)
        size = vector.numel() * vector.element_size()
        if size > self.capacity_bytes:
            self.rejected += 1
            return False
        if key in self.entries:
            old = self.entries.pop(key)
            self.resident_bytes -= old.numel() * old.element_size()
        while self.resident_bytes + size > self.capacity_bytes:
            _, old = self.entries.popitem(last=False)
            self.resident_bytes -= old.numel() * old.element_size()
            self.evictions += 1
        self.entries[key] = vector.detach().clone()
        self.resident_bytes += size
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        return True

    def summary(self):
        return {name: getattr(self, name) for name in (
            'capacity_bytes', 'resident_bytes', 'peak_resident_bytes',
            'hits', 'misses', 'evictions', 'rejected')}
