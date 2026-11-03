import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from fit_calibration import (CANDIDATE_WIDTH, COVERAGE_NOTE, SCHEMA,
                             _fit_isotonic, build_calibrator,
                             conformal_tau, validate_calibrator,
                             write_calibrator)


SHA_A = hashlib.sha256(b'image-a').hexdigest()
SHA_B = hashlib.sha256(b'image-b').hexdigest()
REF_A_LABEL = {
    'raw': 'image/uid0/event/0_event_A.png',
    'normalized': 'uid0/event/0_event_A.png',
    'resolved': 'uid0/camera_photos/0_event_A.png',
}
REF_A_SOURCE = {
    'raw': 'uid0/camera_photos/0_event_A.png',
    'normalized': 'uid0/camera_photos/0_event_A.png',
}
REF_B_SOURCE = {
    'raw': 'uid0/money/0_money_B.png',
    'normalized': 'uid0/money/0_money_B.png',
}


def source_objects():
    return {
        'text:negative': {'kind': 'text', 'session_id': 's_negative'},
        'text:positive': {'kind': 'text', 'session_id': 's_positive'},
        'image:a': {
            'kind': 'image',
            'source_sha256': SHA_A,
            'image_ref': REF_A_SOURCE,
            'sources': [{'session_id': 's_image', 'image_ref': REF_A_SOURCE}],
        },
        'image:b': {
            'kind': 'image',
            'source_sha256': SHA_B,
            'image_ref': REF_B_SOURCE,
            'sources': [{'session_id': 's_image_b', 'image_ref': REF_B_SOURCE}],
        },
    }


def candidate(index, modality, object_id, session_id, score):
    return {
        'record_id': f'r{index}',
        'object_ids': [object_id],
        'session_ids': [session_id],
        'modality': modality,
        'retrieval_cosine': 0.5,
        'rerank_score': score,
        'query_rank': index,
        'retrieval_rank': index,
    }


def dump_payload(question_id, *, split='development', positive_text=False):
    rows = []
    for index in range(46):
        if positive_text and index == 45:
            rows.append(candidate(
                index, 'text', 'text:positive', 's_positive', 0.8))
        else:
            rows.append(candidate(
                index, 'text', 'text:negative', 's_negative',
                0.1 + index / 1000.0))
    rows.append(candidate(46, 'image', 'image:a', 's_image', 0.9))
    rows.append(candidate(47, 'image', 'image:b', 's_image_b', 0.2))
    return {
        'schema': 'omni-candidate-dump-v1',
        'person_id': '0',
        'question_id': question_id,
        'split': split,
        'label_free': True,
        'evaluation_label_files_read': [],
        'candidate_width': CANDIDATE_WIDTH,
        'demand_count': 1,
        'demands': [{
            'demand_id': 'd0',
            'candidate_count': CANDIDATE_WIDTH,
            'candidates': rows,
        }],
        'witnesses': [],
    }


def write_jsonl(path, rows):
    path.write_text(''.join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n'
        for row in rows))


def create_fixture(root, *, split='development'):
    prepared = root / 'prepared'
    prepared.mkdir(parents=True)
    labels = [
        {
            'question_id': '0_q_1',
            'person_id': '0',
            'source_session_ids': ['s_positive'],
            'evidence_image_refs': [REF_A_LABEL],
        },
        {
            'question_id': '0_q_2',
            'person_id': '0',
            'source_session_ids': ['s_missing'],
            'evidence_image_refs': [],
        },
    ]
    labels_path = prepared / 'labels.jsonl'
    write_jsonl(labels_path, labels)

    source_dir = root / 'indexes_final' / 'proof_index_user_0'
    source_dir.mkdir(parents=True)
    (source_dir / 'source.json').write_text(json.dumps({
        'schema': 'source-object-container-v2',
        'person_id': '0',
        'objects': source_objects(),
    }))
    (source_dir / 'index.json').write_text('{}')

    dump = root / 'dump' / '0'
    dump.mkdir(parents=True)
    (dump / '0_q_1.json').write_text(json.dumps(dump_payload(
        '0_q_1', split=split, positive_text=True)))
    (dump / '0_q_2.json').write_text(json.dumps(dump_payload(
        '0_q_2', split=split)))
    return labels_path, root / 'dump'


class FitCalibrationTests(unittest.TestCase):
    def test_session_and_image_matches_fit_separate_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, dump = create_fixture(Path(directory))
            calibrator = build_calibrator(dump, labels)

        self.assertEqual(calibrator['schema'], SCHEMA)
        self.assertEqual(calibrator['split'], 'development')
        self.assertEqual(calibrator['users'], ['0'])
        self.assertEqual(calibrator['calibration']['text']['n'], 92)
        self.assertEqual(calibrator['calibration']['text']['positive_count'], 1)
        self.assertEqual(calibrator['calibration']['image']['n'], 4)
        self.assertEqual(calibrator['calibration']['image']['positive_count'], 1)
        self.assertEqual(calibrator['diagnostics']['n'], 2)
        self.assertEqual(
            calibrator['diagnostics']['g_negative_infinity_count'], 1)
        self.assertEqual(
            calibrator['diagnostics']['g_negative_infinity_ratio'], 0.5)
        self.assertEqual(len(
            calibrator['diagnostics']['modalities']['text']['reliability_bins']),
            10)
        self.assertEqual(calibrator['coverage_note'], COVERAGE_NOTE)
        self.assertIn('只针对 48 个候选池', calibrator['coverage_note'])

    def test_unmatched_evidence_unit_forces_negative_infinity(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, dump = create_fixture(Path(directory))
            calibrator = build_calibrator(dump, labels)

        tau = calibrator['tau']
        self.assertTrue(all(value is None for value in tau.values()))
        self.assertEqual(
            set(calibrator['tau_negative_infinity']),
            {'0.05', '0.1', '0.2'})

    def test_alpha_rank_and_small_n(self):
        self.assertEqual(conformal_tau([], 0.05), -math.inf)
        self.assertEqual(conformal_tau([1.0, 2.0], 0.2), -math.inf)
        self.assertEqual(conformal_tau(list(range(10)), 0.05), -math.inf)
        self.assertEqual(conformal_tau(list(range(10)), 0.1), 0.0)
        self.assertEqual(conformal_tau(list(range(10)), 0.2), 1.0)

    def test_probabilities_are_clipped_at_both_ends(self):
        all_positive = _fit_isotonic([(0.1, 1), (0.2, 1)])
        all_negative = _fit_isotonic([(0.1, 0), (0.2, 0)])
        self.assertEqual(
            set(all_positive['breakpoints']['calibrated_probability']),
            {1.0 - 1e-4})
        self.assertEqual(
            set(all_negative['breakpoints']['calibrated_probability']),
            {1e-4})

    def test_non_development_split_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            labels, dump = create_fixture(Path(directory), split='validation')
            with self.assertRaisesRegex(ValueError, 'must be development'):
                build_calibrator(dump, labels)

    def test_output_is_deterministic_and_self_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, dump = create_fixture(root)
            first = build_calibrator(dump, labels)
            second = build_calibrator(dump, labels)
            output = write_calibrator(first, root / 'calibrator.json')
            written = json.loads(output.read_text())

        self.assertEqual(first, second)
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(validate_calibrator(written), written)
        tampered = copy.deepcopy(written)
        tampered['tau']['0.05'] = 123.0
        with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
            validate_calibrator(tampered)


if __name__ == '__main__':
    unittest.main()
