import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import project_hyperpim_sc_closure as projection


class HyperPimScProjectionTests(unittest.TestCase):
    def test_grammar_calls_are_paired_as_calls(self):
        gate0 = {'rows': [{
            'person_id': '0', 'question_id': 'q0',
            'parser': {'prefill_seconds': 1.0, 'decode_seconds': 10.0},
            'grounding_attempts': [
                {'prefill_seconds': 2.0, 'decode_seconds': 20.0},
                {'prefill_seconds': 2.0, 'decode_seconds': 20.0},
            ],
        }]}
        grammar = {'calls': [
            {'kind': 'parser', 'person_id': '0', 'question_id': 'q0',
             'tokens': [{'fill_seconds': 0.1}] * 10},
            {'kind': 'grounding', 'person_id': '0', 'question_id': 'q0',
             'tokens': [{'fill_seconds': 0.1}] * 5},
            {'kind': 'grounding', 'person_id': '0', 'question_id': 'q0',
             'tokens': [{'fill_seconds': 0.2}] * 8},
        ]}
        result = projection.grammar_forward_samples(gate0, grammar)
        self.assertEqual(result['parser']['calls'], 1)
        self.assertEqual(result['grounding']['calls'], 2)
        self.assertAlmostEqual(result['parser']['grammar_ms_per_token_median'], 100.0)
        self.assertAlmostEqual(result['grounding']['grammar_ms_per_token_median'], 150.0)

    def test_calibration_reads_generated_token_profile(self):
        gate0 = {'rows': []}
        grammar = {'calls': [{
            'kind': 'parser', 'person_id': '0', 'question_id': 'q0',
            'tokens': [{'fill_seconds': 1.0}, {'fill_seconds': 1.0},
                       {'fill_seconds': 1.0}, {'fill_seconds': 99.0}],
        }]}
        probe = {
            'question_id': 'q0',
            'constrained': {
                'output_tokens': None,
                'decode_phase_profile': {
                    'generated_tokens': 3, 'grammar_fill_seconds': 4.5,
                },
            },
        }
        path = ROOT / 'artifacts' / 'hyperpim_sc_trace_v1' / 'calibration_fixture.json'
        path.write_text(__import__('json').dumps(probe))
        try:
            result = projection.grammar_fill_calibration(gate0, grammar, path)
        finally:
            path.unlink()
        self.assertEqual(result['tokens'], 3)
        self.assertAlmostEqual(result['replay_seconds'], 3.0)
        self.assertAlmostEqual(result['factor'], 1.5)


if __name__ == '__main__':
    unittest.main()
