import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from qualify_lpddr_materialization import emit_shared_traffic, qualify, rd16_coordinates


class LPDDRMaterializationTraceTests(unittest.TestCase):
    def sample(self):
        atoms = [{'identity': 'one', 'timestamp': '2026-01-01', 'event_id': 'first'},
                 {'identity': 'two', 'timestamp': '2026-01-02', 'event_id': 'second'}]
        rows = [{'question_id': 'q1', 'effective_timestamp': '2026-01-01', 'visible_atoms': 1,
                 'status': 'ok', 'split': 'development', 'feature_cache': {'evictions': 0, 'misses': 1}},
                {'question_id': 'q2', 'effective_timestamp': '2026-01-02', 'visible_atoms': 2,
                 'status': 'ok', 'split': 'development', 'feature_cache': {'evictions': 0, 'misses': 2}}]
        return atoms, rows

    def emit(self, atoms, rows):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'trace'
            report = emit_shared_traffic(atoms, rows, 32, output)
            return report, output.read_text().splitlines()

    def test_rd16_mapping_covers_channels_columns_banks_and_capacity(self):
        self.assertEqual(rd16_coordinates(0), [0, 0, 0, 0, 0, 0])
        self.assertEqual(rd16_coordinates(32), [1, 0, 0, 0, 0, 0])
        self.assertEqual(rd16_coordinates(64), [0, 0, 0, 0, 0, 16])
        self.assertEqual(rd16_coordinates(4096), [0, 0, 1, 0, 0, 0])
        self.assertEqual(rd16_coordinates(16384), [0, 0, 0, 1, 0, 0])
        self.assertEqual(rd16_coordinates(65536), [0, 0, 0, 0, 1, 0])
        self.assertEqual(rd16_coordinates(2**32 - 32), [1, 0, 3, 3, 65535, 1008])
        for address in (-32, 16, 2**32):
            with self.assertRaises(ValueError):
                rd16_coordinates(address)

    def test_each_immutable_vector_is_written_once_before_any_read(self):
        atoms, rows = self.sample()
        report, lines = self.emit(atoms, rows)
        self.assertEqual(report['producer_vector_bytes'], 64)
        self.assertEqual(report['producer_query_bytes'], 64)
        self.assertEqual(report['read_requests'], 5)
        self.assertEqual(report['write_requests'], 6)
        self.assertEqual(len(report['phases']), 6)
        self.assertEqual(lines.count('ST 0'), 1)
        self.assertEqual(lines.count('ST 32'), 1)
        self.assertEqual(lines.count('LD 0'), 2)
        committed, pending = set(), set()
        for line in lines:
            operation, operand = line.split()
            if operation == 'ST':
                pending.add(int(operand))
            elif operation == 'LD':
                self.assertIn(int(operand), committed)
            else:
                committed.update(pending)
                pending.clear()

    def test_invalid_temporal_capacity_and_identity_inputs_fail_closed(self):
        atoms, rows = self.sample()
        changes = [lambda values: values.reverse(),
                   lambda values: values[0].update(split='sealed_test'),
                   lambda values: values[0].update(visible_atoms=2),
                   lambda values: values[1]['feature_cache'].update(evictions=1),
                   lambda values: values[1]['feature_cache'].update(misses=1)]
        for change in changes:
            changed = copy.deepcopy(rows)
            change(changed)
            with self.assertRaises(ValueError):
                self.emit(atoms, changed)
        atoms[1]['identity'] = atoms[0]['identity']
        with self.assertRaises(ValueError):
            self.emit(atoms, rows)

    def test_qualification_checks_commands_callbacks_tail_and_mapping(self):
        atoms, rows = self.sample()
        traffic, _ = self.emit(atoms, rows)
        fences, reads, writes = [], 0, 0
        for index, phase in enumerate(traffic['phases']):
            reads += phase['read_requests']
            writes += phase['write_requests']
            fences.append({'phase': index, 'cycle': (index + 1) * 224,
                           'accepted_reads': reads, 'accepted_writes': writes,
                           'completed_reads': reads, 'pending': False, 'quiet_tail_cycles': 224})
        driver = {'complete': True, 'cycles': 1344, 'tck_ns': 1.25,
                  'read_requests': reads, 'write_requests': writes, 'completed_reads': reads,
                  'pending_at_finish': False, 'write_refresh_tail_cycles': 224,
                  'bank_requests': traffic['bank_requests'], 'fences': fences,
                  'mapping_samples': [{'address': 32, 'coordinates': rd16_coordinates(32)}]}
        channels = [{'channel': channel, 'commands': {
            'RD16': traffic['channel_requests'][str(channel)].get('LD', 0),
            'WR16': traffic['channel_requests'][str(channel)].get('ST', 0)}} for channel in (0, 1)]
        report = qualify(traffic, driver, channels)
        self.assertTrue(report['host_trace_qualified'])
        self.assertFalse(report['device_calibrated'])
        self.assertFalse(report['pim_backend_qualified'])
        self.assertFalse(report['allows_development128'])
        for field, value in (('completed_reads', 0), ('pending_at_finish', True),
                             ('tck_ns', 0.5), ('bank_requests', {})):
            changed = copy.deepcopy(driver)
            changed[field] = value
            self.assertFalse(qualify(traffic, changed, channels)['host_trace_qualified'])
        changed = copy.deepcopy(driver)
        changed['fences'][-1]['quiet_tail_cycles'] = 1
        self.assertFalse(qualify(traffic, changed, channels)['host_trace_qualified'])
        changed = copy.deepcopy(channels)
        changed[0]['commands']['WR16'] -= 1
        self.assertFalse(qualify(traffic, driver, changed)['host_trace_qualified'])


if __name__ == '__main__':
    unittest.main()
