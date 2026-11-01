import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_visual_reuse import audit_visual_records


class VisualReuseAuditTests(unittest.TestCase):
    def fixture(self):
        model_identity = hashlib.sha256(json.dumps({'weight': 'hash'}, sort_keys=True).encode()).hexdigest()
        snapshot = dict(capacity_bytes=100, resident_bytes=0, peak_resident_bytes=0,
            model_identity=model_identity, processor_identity='processor', policy='ordered-whole-batch-exact-v1',
            hits=0, misses=0, evictions=0, rejected=0, bypasses=0, integrity_failures=0,
            image_hits=0, hit_transfer_seconds=0., miss_compute_seconds=0., resolve_seconds=0.)
        first = dict(snapshot, resident_bytes=4, peak_resident_bytes=4, misses=1,
                     miss_compute_seconds=1., resolve_seconds=1.1)
        second = dict(first, hits=1, image_hits=1, hit_transfer_seconds=.1, resolve_seconds=1.3)
        rows = []
        for index, before, after in ((0, snapshot, first), (1, first, second)):
            rows.append(dict(question_id=f'q{index}', generation_equivalent=True,
                visual_features_bitwise_equivalent=True, visual_cache_before=copy.deepcopy(before),
                visual_cache_after=copy.deepcopy(after), images=[{'sha256': 'a'}],
                reference_vision_seconds=[1.], service_seconds=3., prefill_seconds=2.))
        smoke = dict(complete=True, verify_generation=True, evaluation_label_files_read=[],
            rows=rows, visual_state_reuse=second, visual_state_identity_files_sha256={'weight': 'hash'},
            source_files_sha256={'file': 'hash'})
        registration = dict(question_ids=['q0', 'q1'], arguments={'visual_cache_bytes': 100},
                            source_files_sha256={'file': 'hash'})
        joint = dict(known_reservations={'declared_visual_state_budget_bytes': 100},
            known_joint_pool_fits=True, end_to_end_speedup=None, observed_visual_state_peak_bytes=4)
        return smoke, registration, joint

    def test_natural_hit_and_exact_equivalence(self):
        result = audit_visual_records(*self.fixture())
        self.assertEqual(result['natural_batch_hits'], 1)
        self.assertEqual(result['natural_batch_hit_rate'], .5)
        self.assertFalse(result['main_experiment_closed'])
        self.assertIsNone(result['end_to_end_speedup'])

    def test_mutated_equivalence_capacity_reference_or_identities_fail(self):
        for kind in ('feature', 'tokens', 'capacity', 'reference', 'natural', 'identity', 'sources'):
            smoke, registration, joint = self.fixture()
            if kind == 'feature':
                smoke['rows'][1]['visual_features_bitwise_equivalent'] = False
            elif kind == 'tokens':
                smoke['rows'][1]['generation_equivalent'] = False
            elif kind == 'capacity':
                joint['known_reservations']['declared_visual_state_budget_bytes'] = 0
            elif kind == 'reference':
                smoke['rows'][1]['visual_cache_after']['hits'] += 1
            elif kind == 'natural':
                smoke['rows'][1]['images'][0]['sha256'] = 'unseen'
            elif kind == 'identity':
                smoke['rows'][1]['visual_cache_before']['model_identity'] = 'changed'
            else:
                registration['source_files_sha256'] = {}
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_visual_records(smoke, registration, joint)


if __name__ == '__main__':
    unittest.main()
