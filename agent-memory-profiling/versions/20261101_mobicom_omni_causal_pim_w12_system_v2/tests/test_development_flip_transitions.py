import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_development_flip_transitions as audit


def write(directory, name, value):
    path = Path(directory) / name
    path.write_text(json.dumps(value))
    return path


class DevelopmentFlipTransitionTests(unittest.TestCase):
    def fixture(self, directory):
        flips = []
        r4_rows = []
        paired_rows = []
        for index in range(13):
            question_id = f"{index % 3}_q_{index}"
            person_id = str(index % 3)
            transition = 'candidate_only' if index < 8 else 'reference_only'
            reference = transition == 'reference_only'
            candidate = transition == 'candidate_only'
            r4_candidate = (
                index in (0, 1, 6, 7) if transition == 'candidate_only'
                else index in (9, 10))
            r4_reference = index in (0, 8, 9, 10, 11)
            flips.append({
                'question_id': question_id, 'person_id': person_id,
                'transition': transition,
                'reference_correct': reference, 'candidate_correct': candidate,
            })
            r4_rows.append({
                'question_id': question_id, 'person_id': person_id,
                'reference_correct': r4_reference,
                'candidate_correct': r4_candidate,
                'candidate_parse_valid': True, 'candidate_eos': True,
            })
            paired_rows.append({
                'question_id': question_id, 'person_id': person_id,
                'upstream_identical': True, 'stop_reason_equal': True,
            })
        r2 = {'complete': True, 'flips': flips}
        r4 = {
            'complete': True, 'requests': 13, 'rows': r4_rows,
            'quality': {'reference_correct': sum(row['reference_correct'] for row in r4_rows),
                        'candidate_correct': sum(row['candidate_correct'] for row in r4_rows)},
        }
        paired = {
            'complete': True, 'requests': 13, 'rows': paired_rows,
            'arms': {'candidate': 'w8a16_r4'}, 'candidate_residual_bits': 4,
            'candidate_fallback_scope': 'none',
        }
        return (write(directory, 'r2.json', r2), write(directory, 'r4.json', r4),
                write(directory, 'paired.json', paired))

    def load(self, path):
        value = json.loads(path.read_text())
        value['_path'] = str(path)
        return value

    def test_transition_outcomes_are_joined_without_hiding_lost_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            r2, r4, paired = self.fixture(directory)
            result = audit.build_audit(self.load(r2), self.load(r4), self.load(paired))
            summary = result['summary']
            self.assertEqual(summary['r2_partition'],
                             {'candidate_only': 8, 'reference_only': 5})
            self.assertEqual(summary['candidate_only']['preserved'], 4)
            self.assertEqual(summary['candidate_only']['lost'], 4)
            self.assertEqual(summary['reference_only']['recovered'], 2)
            self.assertEqual(summary['reference_only']['not_recovered'], 3)
            self.assertEqual(summary['net_r2_success_recovery'], -2)
            self.assertEqual(summary['r4_candidate_correct_strict_proxy'], 6)
            self.assertEqual(summary['r4_bf16_correct_strict_proxy'], 5)

    def test_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            r2, r4, paired = self.fixture(directory)
            value = self.load(r4)
            value['rows'][0]['question_id'] = 'other'
            with self.assertRaises(ValueError):
                audit.build_audit(self.load(r2), value, self.load(paired))

    def test_claimed_strict_success_requires_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            r2, r4, paired = self.fixture(directory)
            value = self.load(r4)
            value['rows'][0]['candidate_eos'] = False
            with self.assertRaises(ValueError):
                audit.build_audit(self.load(r2), value, self.load(paired))


if __name__ == '__main__':
    unittest.main()
