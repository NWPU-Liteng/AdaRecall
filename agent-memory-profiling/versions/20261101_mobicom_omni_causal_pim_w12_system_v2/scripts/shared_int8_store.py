"""Immutable host INT8 primary and address-derived functional scan, not PIM timing."""
from __future__ import annotations

import hashlib
import json

import numpy as np


CONTRACT = 'immutable_host_int8_primary_v1'
POOL_BYTES = 2 * 1024**3
TRANSFER_BYTES = 32


def quantize(values):
    return np.clip(np.rint(values * 127), -127, 127).astype(np.int8)


class SharedInt8Store:
    def __init__(self, atoms, contract, capacity_bytes):
        if not atoms or not contract or not 0 < capacity_bytes <= POOL_BYTES:
            raise ValueError('nonempty registry, representation contract and bounded capacity required')
        self.registry = {atom['identity']: (atom['ordinal'], atom['timestamp']) for atom in atoms}
        if (len(self.registry) != len(atoms)
                or sorted(ordinal for ordinal, _ in self.registry.values()) != list(range(len(atoms)))):
            raise ValueError('unique identities and contiguous canonical ordinals required')
        self.contract = contract
        self.capacity_bytes = capacity_bytes
        self.storage = None
        self.entries = {}
        self.last_cutoff = None

    def materialize(self, atoms, vectors, cutoff, contract):
        if contract != self.contract or (self.last_cutoff is not None and cutoff < self.last_cutoff):
            raise ValueError('representation mismatch or nonchronological request')
        if (vectors.ndim != 2 or vectors.dtype != np.float32 or len(atoms) != len(vectors)
                or not len(atoms) or not vectors.shape[1] or vectors.shape[1] % TRANSFER_BYTES
                or not np.isfinite(vectors).all()):
            raise ValueError('finite FP32 aligned visible vectors required')
        dimension = vectors.shape[1]
        ordinals = [atom['ordinal'] for atom in atoms]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError('duplicate request slots')
        for atom in atoms:
            if (self.registry.get(atom['identity']) != (atom['ordinal'], atom['timestamp'])
                    or atom['timestamp'] > cutoff):
                raise ValueError('identity, address or visibility mismatch')
        if self.storage is None:
            allocated = len(self.registry) * dimension
            if allocated > self.capacity_bytes:
                raise ValueError('primary allocation exceeds declared capacity; no silent eviction')
            self.storage = np.empty((len(self.registry), dimension), dtype=np.int8)
        if self.storage.shape[1] != dimension:
            raise ValueError('representation dimension changed')
        produced = []
        for position, atom in enumerate(atoms):
            if atom['identity'] in self.entries:
                continue
            values = quantize(vectors[position])
            self.storage[atom['ordinal']] = values
            entry = {'identity': atom['identity'], 'ordinal': atom['ordinal'],
                     'timestamp': atom['timestamp'], 'address': atom['ordinal'] * dimension,
                     'bytes': dimension, 'sha256': hashlib.sha256(values.tobytes()).hexdigest()}
            self.entries[atom['identity']] = entry
            produced.append(entry)
        contiguous = ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))
        values = (self.storage[ordinals[0]:ordinals[-1] + 1].view() if contiguous
                  else self.storage[ordinals])
        values.setflags(write=False)
        self.last_cutoff = cutoff
        return values, {'contract': CONTRACT, 'representation_sha256': hashlib.sha256(contract.encode()).hexdigest(),
                        'placement': 'host_cpu_primary_not_pim_resident',
                        'dimension': dimension, 'capacity_bytes': self.capacity_bytes,
                        'allocated_bytes': self.storage.nbytes, 'committed_bytes': len(self.entries) * dimension,
                        'producer_vectors': len(produced), 'producer_write_bytes': len(produced) * dimension,
                        'reused_vectors': len(atoms) - len(produced), 'producer_entries': produced,
                        'scan_ordinals': ordinals, 'scan_vector_read_bytes': values.nbytes,
                        'gather_copy_bytes': 0 if contiguous else values.nbytes,
                        'query_write_address': POOL_BYTES, 'query_write_bytes': dimension,
                        'result_address': POOL_BYTES + dimension,
                        'boundary': 'declared logical shared-address traffic; not captured DRAM or PIM commands'}

    def inventory(self):
        return sorted(self.entries.values(), key=lambda entry: entry['ordinal'])

    def save(self, path):
        with path.open('xb') as handle:
            for entry in self.inventory():
                handle.seek(entry['address'])
                handle.write(self.storage[entry['ordinal']].tobytes())
        return {'inventory': self.inventory(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'representation_sha256': hashlib.sha256(self.contract.encode()).hexdigest(),
                'committed_bytes': sum(entry['bytes'] for entry in self.entries.values()),
                'file_bytes': path.stat().st_size,
                'inventory_sha256': hashlib.sha256(json.dumps(self.inventory(), sort_keys=True).encode()).hexdigest()}


def physical_bank_topk(values, query, ordinals, top_k):
    if (values.dtype != np.int8 or query.dtype != np.int8 or values.ndim != 2 or query.ndim != 1
            or values.shape != (len(ordinals), len(query)) or not len(ordinals) or not len(query)
            or len(query) % TRANSFER_BYTES or len(query) * 128**2 > np.iinfo(np.int32).max
            or len(set(ordinals)) != len(ordinals) or not 0 < top_k <= len(values)
            or any(ordinal < 0 or (ordinal + 1) * len(query) > POOL_BYTES for ordinal in ordinals)):
        raise ValueError('aligned in-capacity INT8 operands and valid top-k required')
    offsets = np.arange(len(query) // TRANSFER_BYTES) * TRANSFER_BYTES
    addresses = np.asarray(ordinals, dtype=np.int64)[:, None] * len(query) + offsets[None, :]
    transactions = addresses // TRANSFER_BYTES
    banks = (transactions & 1) * 16 + ((transactions >> 7) & 3) * 4 + ((transactions >> 9) & 3)
    chunks = values.reshape(len(values), -1, TRANSFER_BYTES)
    query_chunks = query.reshape(-1, TRANSFER_BYTES)
    scores = np.zeros(len(values), dtype=np.int32)
    bank_reads = {}
    for bank in range(32):
        positions, chunk_ids = np.where(banks == bank)
        if not len(positions):
            continue
        partial = np.sum(chunks[positions, chunk_ids].astype(np.int32)
                         * query_chunks[chunk_ids].astype(np.int32), axis=1, dtype=np.int32)
        np.add.at(scores, positions, partial)
        bank_reads[f'{bank // 16}/{(bank % 16) // 4}/{bank % 4}'] = len(positions)
    selected = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
    return selected, scores[selected], bank_reads
