import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

from fit_calibration import _payload_sha256
from pim_backend.candidate_policy import (apply_conformal_candidate_policy,
                                          load_calibrator,
                                          threshold_for_alpha)


def calibrator_payload(tau=None):
    payload = {
        'schema': 'omni-conformal-calibrator-v1',
        'split': 'development',
        'candidate_width': 48,
        'alphas': [0.05, 0.1, 0.2],
        'tau': {'0.05': tau, '0.1': tau, '0.2': tau},
    }
    payload['sha256'] = _payload_sha256(payload)
    return payload


def candidate(index, score, demand_id='d0', text='one two'):
    return {
        'witness_id': f'{demand_id}#{index}',
        'demand_id': demand_id,
        'base_record_id': f'record-{index}',
        'query_rank': index,
        'retrieval_rank': index,
        'score': score,
        'retrieval_text': text,
    }


class CandidatePolicyTests(unittest.TestCase):
    def test_calibrator_identity_and_self_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibrator.json'
            payload = calibrator_payload(tau=0.75)
            path.write_text(json.dumps(payload))
            loaded = load_calibrator(path, payload['sha256'])

            with self.assertRaisesRegex(ValueError, 'configured sha256'):
                load_calibrator(path, 'b' * 64)

            tampered = copy.deepcopy(payload)
            tampered['tau']['0.1'] = 0.1
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                load_calibrator(path, payload['sha256'])

        self.assertEqual(threshold_for_alpha(loaded, 0.1), 0.75)

    def test_null_threshold_denotes_negative_infinity(self):
        self.assertEqual(
            threshold_for_alpha(calibrator_payload(), 0.2), -math.inf)

    def test_threshold_selection_fills_to_minimum_and_counts_tokens(self):
        rows = [
            candidate(0, 0.9), candidate(1, 0.8), candidate(2, 0.7),
            candidate(3, 0.6), candidate(4, 0.5),
        ]
        selected, stats = apply_conformal_candidate_policy(
            rows, demand_ids=['d0'], tau=0.75, min_per_demand=4,
            token_counter=lambda values: sum(
                len(value['retrieval_text'].split()) for value in values))

        self.assertEqual([row['base_record_id'] for row in selected],
                         ['record-0', 'record-1', 'record-2', 'record-3'])
        demand = stats['demands'][0]
        self.assertEqual(demand['candidate_count_before'], 5)
        self.assertEqual(demand['candidate_count_after'], 4)
        self.assertEqual(demand['retained_by_threshold'], 2)
        self.assertEqual(demand['filled_to_min'], 2)
        self.assertEqual(demand['witness_candidate_tokens_before'], 10)
        self.assertEqual(demand['witness_candidate_tokens_after'], 8)
        self.assertEqual(demand['witness_candidate_token_delta'], -2)
        self.assertEqual(stats['witness_candidate_token_delta'], -2)

    def test_negative_infinity_tau_retains_the_whole_pool(self):
        rows = [candidate(index, 1.0 - index / 100.0) for index in range(48)]
        selected, stats = apply_conformal_candidate_policy(
            rows, demand_ids=['d0'], tau=-math.inf, min_per_demand=4)

        self.assertEqual(len(selected), 48)
        self.assertEqual(stats['candidate_count_after'], 48)
        self.assertEqual(stats['tau'], None)
        self.assertEqual(stats['demands'][0]['retained_by_threshold'], 48)

    def test_policy_validates_width_and_minimum(self):
        rows = [candidate(0, 0.9)]
        with self.assertRaisesRegex(ValueError, 'min_per_demand'):
            apply_conformal_candidate_policy(
                rows, demand_ids=['d0'], tau=0.0, min_per_demand=49)
        with self.assertRaisesRegex(ValueError, 'fixed 48-wide'):
            apply_conformal_candidate_policy(
                rows, demand_ids=['d0'], tau=0.0, candidate_width=47)


if __name__ == '__main__':
    unittest.main()
