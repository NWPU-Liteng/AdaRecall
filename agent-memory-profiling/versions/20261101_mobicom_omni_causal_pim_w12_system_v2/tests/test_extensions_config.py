import hashlib
import json
import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from extensions_config import Extensions, load_extensions


DEFAULTS = {
    'as_of_rule': 'session_end_before_question',
    'calibration': {
        'alpha': 0.1,
        'calibrator_path': None,
        'calibrator_sha256': None,
        'candidate_policy': 'fixed48',
        'min_per_demand': 4,
    },
    'candidate_dump': None,
    'join_algorithm': 'dfs_maxmin',
    'noisy_or': {
        'fill_redundancy': False,
        'levels': 20,
        'max_witnesses_per_demand': 3,
    },
    'operation_executor': 'off',
    'progressive_topk': {
        'bits': 12,
        'eps_rel': 1e-9,
        'group_size': 128,
        'lm_head': 'off',
        'lm_head_shadow_every': 8,
        'plane_schedule': [1] * 12,
        'retrieval': 'off',
    },
    'stage_profiler': 'off',
    'visibility_protocol': 'history_end',
    'vision_feature_cache': 'off',
    'witness_values': 'off',
    'workload_trace': 'off',
}


class ExtensionsConfigTests(unittest.TestCase):
    def test_defaults_match_plan(self):
        extensions = load_extensions({})
        self.assertEqual(extensions.as_dict(), DEFAULTS)
        self.assertEqual(load_extensions({'extensions': {}}), extensions)

    def test_unknown_keys_are_rejected_at_every_level(self):
        cases = (
            {'extensions': {'unknown': 1}},
            {'extensions': {'progressive_topk': {'unknown': 1}}},
            {'extensions': {'calibration': {'unknown': 1}}},
            {'extensions': {'noisy_or': {'unknown': 1}}},
        )
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_extensions(config)

    def test_plane_schedule_sum_and_types_are_validated(self):
        with self.assertRaises(ValueError):
            load_extensions({'extensions': {'progressive_topk': {
                'bits': 12, 'plane_schedule': [1] * 11}}})
        with self.assertRaises(ValueError):
            load_extensions({'extensions': {'progressive_topk': {
                'bits': 12, 'plane_schedule': [0] * 12}}})
        configured = load_extensions({'extensions': {'progressive_topk': {
            'bits': 12, 'plane_schedule': [4, 2, 2, 2, 2]}}})
        self.assertEqual(configured.progressive_topk.plane_schedule,
                         (4, 2, 2, 2, 2))

    def test_choices_booleans_and_numbers_are_strict(self):
        cases = (
            {'extensions': {'visibility_protocol': 'future'}},
            {'extensions': {'progressive_topk': {'lm_head': 'on'}}},
            {'extensions': {'progressive_topk': {'bits': True}}},
            {'extensions': {'calibration': {'alpha': 1.0}}},
            {'extensions': {'noisy_or': {'fill_redundancy': 1}}},
        )
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_extensions(config)

    def test_conformal_requires_calibrator_identity_and_fixed_width(self):
        digest = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'requires calibrator_path'):
            load_extensions({'extensions': {'calibration': {
                'candidate_policy': 'conformal',
            }}})
        with self.assertRaisesRegex(ValueError, 'min_per_demand must be <= 48'):
            load_extensions({'extensions': {'calibration': {
                'candidate_policy': 'conformal',
                'calibrator_path': '/tmp/calibrator.json',
                'calibrator_sha256': digest,
                'min_per_demand': 49,
            }}})
        configured = load_extensions({'extensions': {'calibration': {
            'candidate_policy': 'conformal',
            'calibrator_path': '/tmp/calibrator.json',
            'calibrator_sha256': digest,
        }}})
        self.assertEqual(configured.calibration.candidate_policy, 'conformal')
        self.assertEqual(configured.calibration.calibrator_sha256, digest)

    def test_yaml_on_off_booleans_are_normalized_for_switches(self):
        extensions = load_extensions({'extensions': {
            'operation_executor': False,
            'witness_values': True,
            'progressive_topk': {'lm_head': False, 'retrieval': False},
        }})
        self.assertEqual(extensions.operation_executor, 'off')
        self.assertEqual(extensions.witness_values, 'on')
        self.assertEqual(extensions.progressive_topk.lm_head, 'off')
        self.assertEqual(extensions.progressive_topk.retrieval, 'off')

    def test_output_contract_hashes_canonical_json(self):
        extensions = Extensions()
        contract = extensions.output_contract()
        canonical = json.dumps(
            extensions.as_dict(), ensure_ascii=False, sort_keys=True,
            separators=(',', ':'))
        self.assertEqual(contract['config'], extensions.as_dict())
        self.assertEqual(contract['sha256'],
                         hashlib.sha256(canonical.encode('utf-8')).hexdigest())


if __name__ == '__main__':
    unittest.main()
