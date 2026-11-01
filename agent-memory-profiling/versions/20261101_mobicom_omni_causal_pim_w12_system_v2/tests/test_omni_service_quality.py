from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_omni_service_quality import summarize_quality


class ServiceQualityTests(unittest.TestCase):
    def fixture(self):
        predictions = [{'question_id': str(index), 'person_id': '0', 'answer_sha256': 'answer',
                        'stop_reason': 'eos' if index < 2 else 'max_new_tokens'} for index in range(3)]
        requests = [{'question_id': row['question_id'], 'prediction_sha256': 'answer',
                     'prompt_sha256': 'prompt', 'reader_stop_reason': row['stop_reason']}
                    for row in predictions]
        judge = {'complete': True, 'rows': [{'question_id': str(index), 'prediction_sha256': 'answer',
                  'prompt_sha256': 'prompt', 'parse_valid': index != 1,
                  'value': 1. if index != 1 else None} for index in range(3)]}
        return predictions, requests, judge

    def test_invalid_and_truncated_requests_remain_in_denominator(self):
        result = summarize_quality(*self.fixture())
        self.assertEqual(result['requests'], 3)
        self.assertEqual(result['strict_local_proxy_successes'], 1)
        self.assertEqual(result['strict_local_proxy_success_rate'], 1 / 3)

    def test_subset_order_hash_and_invalid_scores_fail(self):
        for mutation in ('subset', 'order', 'hash', 'score'):
            predictions, requests, judge = deepcopy(self.fixture())
            if mutation == 'subset':
                judge['rows'].pop()
            elif mutation == 'order':
                judge['rows'].reverse()
            elif mutation == 'hash':
                judge['rows'][0]['prediction_sha256'] = 'other'
            else:
                judge['rows'][0]['value'] = .5
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_quality(predictions, requests, judge)


if __name__ == '__main__':
    unittest.main()
