import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_precision_transition as audit


def write(directory, name, value):
    path = Path(directory) / name
    path.write_text(json.dumps(value))
    return path


class PrecisionTransitionAuditTests(unittest.TestCase):
    def fixture(self, directory):
        flips = []
        rows = []
        paired_rows = []
        for index in range(13):
            question_id = f"{index % 3}_q_{index}"
            person_id = str(index % 3)
            transition = 'candidate_only' if index < 8 else 'reference_only'
            reference = transition == 'reference_only'
            r2_candidate = transition == 'candidate_only'
            candidate = index in (0, 1, 2, 3, 8)
            flips.append({
                'question_id': question_id, 'person_id': person_id,
                'transition': transition,
                'reference_correct': reference, 'candidate_correct': r2_candidate,
            })
            rows.append({
                'question_id': question_id, 'person_id': person_id,
                'reference_correct': reference, 'candidate_correct': candidate,
                'candidate_parse_valid': True, 'candidate_eos': True,
            })
            paired_rows.append({
                'question_id': question_id, 'person_id': person_id,
                'upstream_identical': True,
                'stop_reason_equal': index % 2 == 0,
            })
        r2 = {'complete': True, 'flips': flips}
        candidate = {
            'complete': True, 'requests': 13, 'rows': rows,
            'serving_pair': {'candidate_quantization': 'w12a16'},
            'quality': {
                'reference_correct': sum(row['reference_correct'] for row in rows),
                'candidate_correct': sum(row['candidate_correct'] for row in rows),
            },
        }
        paired = {
            'complete': True, 'requests': 13, 'rows': paired_rows,
            'arms': {'candidate': 'w12a16'},
            'candidate_residual_bits': 0,
            'candidate_fallback_scope': 'none',
        }
        return (write(directory, 'r2.json', r2),
                write(directory, 'candidate.json', candidate),
                write(directory, 'paired.json', paired))

    def load(self, path):
        value = json.loads(path.read_text())
        value['_path'] = str(path)
        return value

    def test_join_reports_preserved_and_lost_r2_successes(self):
        with tempfile.TemporaryDirectory() as directory:
            r2, candidate, paired = self.fixture(directory)
            result = audit.build_audit(
                self.load(r2), self.load(candidate), self.load(paired),
                candidate_label='w12a16', candidate_residual_bits=0,
                candidate_weight_bits=12)
            summary = result['summary']
            self.assertEqual(summary['r2_partition'],
                             {'candidate_only': 8, 'reference_only': 5})
            self.assertEqual(summary['candidate_only']['preserved'], 4)
            self.assertEqual(summary['candidate_only']['lost'], 4)
            self.assertEqual(summary['reference_only']['recovered'], 1)
            self.assertEqual(summary['reference_only']['not_recovered'], 4)
            self.assertEqual(summary['net_r2_success_recovery'], -3)
            self.assertEqual(summary['candidate_correct_strict_proxy'], 5)
            self.assertEqual(result['candidate_weight_bits'], 12)
            self.assertEqual(result['candidate_residual_bits'], 0)

    def test_strict_success_without_eos_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            r2, candidate, paired = self.fixture(directory)
            value = self.load(candidate)
            value['rows'][0]['candidate_eos'] = False
            with self.assertRaises(ValueError):
                audit.build_audit(
                    self.load(r2), value, self.load(paired),
                    candidate_label='w12a16', candidate_residual_bits=0,
                    candidate_weight_bits=12)

    def test_fallback_scope_must_be_none(self):
        with tempfile.TemporaryDirectory() as directory:
            r2, candidate, paired = self.fixture(directory)
            value = self.load(paired)
            value['candidate_fallback_scope'] = 'reader'
            with self.assertRaises(ValueError):
                audit.build_audit(
                    self.load(r2), self.load(candidate), value,
                    candidate_label='w12a16', candidate_residual_bits=0,
                    candidate_weight_bits=12)


if __name__ == '__main__':
    unittest.main()
