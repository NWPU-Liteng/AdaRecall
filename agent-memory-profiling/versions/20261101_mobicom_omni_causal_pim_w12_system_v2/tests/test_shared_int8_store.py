from pathlib import Path
from copy import deepcopy
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from shared_int8_store import SharedInt8Store, physical_bank_topk, quantize
from system_audit import shared_primary_valid


class SharedInt8StoreTests(unittest.TestCase):
    def setUp(self):
        self.atoms = [{'identity': f'atom-{index}', 'ordinal': index, 'timestamp': str(index)}
                      for index in range(4)]
        self.vectors = np.random.default_rng(19).uniform(-1, 1, (4, 1024)).astype(np.float32)
        self.store = SharedInt8Store(self.atoms, 'weights-pooling-quantization', 4096)

    def materialize(self, indices, cutoff='3', contract='weights-pooling-quantization'):
        return self.store.materialize([self.atoms[index] for index in indices], self.vectors[indices], cutoff, contract)

    def test_producer_once_and_readonly_shared_view(self):
        first, trace = self.materialize([0, 1], cutoff='1')
        self.assertEqual(trace['producer_write_bytes'], 2048)
        second, trace = self.materialize([0, 1, 2])
        self.assertEqual(trace['producer_vectors'], 1)
        self.assertEqual(trace['reused_vectors'], 2)
        self.assertEqual(trace['gather_copy_bytes'], 0)
        self.assertTrue(np.shares_memory(first, second))
        self.assertTrue(np.array_equal(second, quantize(self.vectors[:3])))
        self.assertFalse(second.flags.writeable)
        _, trace = self.materialize([0, 2])
        self.assertEqual(trace['producer_vectors'], 0)
        self.assertEqual(trace['gather_copy_bytes'], 2048)
        self.assertEqual(len(self.store.inventory()), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'primary.bin'
            snapshot = self.store.save(path)
            self.assertEqual(snapshot['file_bytes'], 3072)
            self.assertEqual(path.read_bytes(), quantize(self.vectors[:3]).tobytes())

    def test_contract_visibility_identity_capacity_and_chronology_fail_closed(self):
        for indices, cutoff, contract in [([0, 1], '0', 'weights-pooling-quantization'),
                                          ([0], '0', 'different'), ([0, 0], '0', 'weights-pooling-quantization')]:
            with self.assertRaises(ValueError):
                self.materialize(indices, cutoff, contract)
        self.assertIsNone(self.store.storage)
        self.materialize([0, 1], '1')
        with self.assertRaises(ValueError):
            self.materialize([0], '0')
        corrupted = {**self.atoms[0], 'identity': 'other'}
        with self.assertRaises(ValueError):
            self.store.materialize([corrupted], self.vectors[:1], '3', self.store.contract)
        tiny = SharedInt8Store(self.atoms, 'contract', 1024)
        with self.assertRaises(ValueError):
            tiny.materialize(self.atoms[:1], self.vectors[:1], '0', 'contract')
        self.assertIsNone(tiny.storage)

    def test_physical_scan_exact_including_ties_and_nonprefix_addresses(self):
        for dimension in (32, 1024, 2048):
            rng = np.random.default_rng(dimension)
            values = rng.integers(-128, 128, (5, dimension), dtype=np.int8)
            values[1] = values[0]
            query = rng.integers(-128, 128, dimension, dtype=np.int8)
            selected, scores, bank_reads = physical_bank_topk(values, query, [0, 1, 17, 69, 1184], 5)
            reference = values.astype(np.int32) @ query.astype(np.int32)
            order = np.lexsort((np.arange(5), -reference))
            self.assertEqual(selected.tolist(), order.tolist())
            self.assertEqual(scores.tolist(), reference[order].tolist())
            self.assertEqual(sum(bank_reads.values()) * 32, values.nbytes)
        for ordinals in ([0, 0], [-1, 1], [0, 2**21]):
            with self.assertRaises(ValueError):
                physical_bank_topk(values[:2], query, ordinals, 2)

    def test_online_audit_rejects_missing_and_mutated_producers(self):
        store = SharedInt8Store(self.atoms, 'contract', 1024**2)
        rows = []
        for count in (2, 3):
            atoms = self.atoms[:count]
            values, trace = store.materialize(atoms, self.vectors[:count], str(count), 'contract')
            _, _, reads = physical_bank_topk(values, np.ones(1024, dtype=np.int8), trace['scan_ordinals'], count)
            trace.update(primary_values_reference_exact=True, physical_bank_int32_exact=True,
                         bank_read_transactions=reads, query_read_bytes=1024,
                         result_payload_bytes=count * 8, result_write_bytes=32, result_read_bytes=0,
                         selected_entries=[store.entries[atoms[0]['identity']]])
            rows.append({'status': 'ok', 'effective_timestamp': str(count), 'int8_primary': trace,
                         'visible_atoms': count, 'selected_atoms': [atoms[0]]})
        config = {'int8_store_contract': 'immutable_host_int8_primary_v1', 'int8_capacity_mib': 1}
        self.assertTrue(shared_primary_valid(rows, config))
        for field, value in [('producer_entries', []), ('producer_write_bytes', 0),
                             ('placement', 'pim'), ('committed_bytes', 0), ('query_read_bytes', 0)]:
            changed = deepcopy(rows)
            changed[0]['int8_primary'][field] = value
            self.assertFalse(shared_primary_valid(changed, config))
        changed = deepcopy(rows)
        changed[1]['int8_primary']['producer_entries'] = rows[0]['int8_primary']['producer_entries']
        self.assertFalse(shared_primary_valid(changed, config))
        self.assertFalse(shared_primary_valid(rows, {**config, 'int8_store_contract': 'unrecognized'}))


if __name__ == '__main__':
    unittest.main()
