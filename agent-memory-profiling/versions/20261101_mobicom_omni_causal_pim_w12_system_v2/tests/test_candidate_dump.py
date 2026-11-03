import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.candidate_dump import (build_candidate_dump,
                                        object_session_ids,
                                        write_candidate_dump)


OBJECTS = {
    'text:a': {'kind': 'text', 'session_id': 's1'},
    'text:b': {'kind': 'text', 'session_id': 's4'},
    'image:x': {'kind': 'image', 'sources': [
        {'session_id': 's2'}, {'session_id': 's3'}]},
}
DEMANDS = [
    {'demand_id': 'd0', 'text': 'first'},
    {'demand_id': 'd1', 'text': 'second'},
]


def candidates():
    rows = []
    for demand_id in ('d0', 'd1'):
        for rank in range(48):
            image = demand_id == 'd0' and rank % 2 == 1
            rows.append({
                'witness_id': f'r{rank}#{demand_id}',
                'demand_id': demand_id,
                'base_record_id': f'r{rank}',
                'kind': 'image_locator' if image else 'source_text',
                'object_ids': ['image:x'] if image else ['text:a'],
                'retrieval_cosine': 1.0 / (rank + 1),
                'score': 5.0 - rank / 100.0,
                'query_rank': rank,
                'retrieval_rank': rank,
            })
    return rows


class CandidateDumpTests(unittest.TestCase):
    def test_export_contains_fixed_width_candidates_and_sessions(self):
        witnesses = [{
            'witness_id': 'w:test',
            'demand_id': 'd0',
            'record_ids': ['a0', 'a1'],
            'retrieval_record_ids': ['r0', 'r1'],
            'object_ids': ['text:a', 'image:x'],
            'base_record_id': 'r0|r1',
            'score': 4.5,
        }]
        payload = build_candidate_dump(
            person_id='1', question_id='1_q_test', split='development',
            demands=DEMANDS, candidates=candidates(), witnesses=witnesses,
            objects=OBJECTS)
        self.assertEqual(payload['schema'], 'omni-candidate-dump-v1')
        self.assertTrue(payload['label_free'])
        self.assertEqual(payload['demand_count'], 2)
        first = payload['demands'][0]
        self.assertEqual(first['candidate_count'], 48)
        self.assertEqual(first['candidates'][0]['record_id'], 'r0')
        self.assertEqual(first['candidates'][0]['modality'], 'text')
        self.assertEqual(first['candidates'][0]['session_ids'], ['s1'])
        self.assertEqual(first['candidates'][1]['modality'], 'image')
        self.assertEqual(first['candidates'][1]['session_ids'], ['s2', 's3'])
        self.assertEqual(
            set(first['candidates'][0]), {
                'record_id', 'object_ids', 'session_ids', 'modality',
                'retrieval_cosine', 'rerank_score', 'query_rank',
                'retrieval_rank'})
        self.assertEqual(payload['witnesses'][0]['record_ids'], ['r0', 'r1'])
        self.assertEqual(payload['witnesses'][0]['demand_id'], 'd0')
        encoded = json.dumps(payload)
        self.assertNotIn('golden_answers', encoded)
        self.assertNotIn('question_type', encoded)

    def test_image_session_union_is_read_from_source(self):
        self.assertEqual(object_session_ids(OBJECTS, ['image:x']), ['s2', 's3'])
        self.assertEqual(
            object_session_ids(OBJECTS, ['text:a', 'text:b']), ['s1', 's4'])

    def test_wrong_candidate_width_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'expected 48'):
            build_candidate_dump(
                person_id='1', question_id='1_q_test', split='development',
                demands=DEMANDS, candidates=candidates()[:-1], witnesses=[],
                objects=OBJECTS)

    def test_duplicate_rank_or_unknown_object_fails_closed(self):
        rows = candidates()
        rows[1]['retrieval_rank'] = 0
        with self.assertRaisesRegex(ValueError, 'retrieval ranks'):
            build_candidate_dump(
                person_id='1', question_id='1_q_test', split='development',
                demands=DEMANDS, candidates=rows, witnesses=[],
                objects=OBJECTS)

        rows = candidates()
        rows[0]['object_ids'] = ['missing']
        with self.assertRaisesRegex(ValueError, 'unknown source object'):
            build_candidate_dump(
                person_id='1', question_id='1_q_test', split='development',
                demands=DEMANDS, candidates=rows, witnesses=[],
                objects=OBJECTS)

    def test_nonfinite_scores_fail_closed(self):
        rows = candidates()
        rows[0]['retrieval_cosine'] = math.inf
        with self.assertRaisesRegex(ValueError, 'retrieval_cosine'):
            build_candidate_dump(
                person_id='1', question_id='1_q_test', split='development',
                demands=DEMANDS, candidates=rows, witnesses=[],
                objects=OBJECTS)

    def test_write_is_fresh_and_atomic(self):
        payload = build_candidate_dump(
            person_id='1', question_id='1_q_test', split='development',
            demands=[], candidates=[], witnesses=[], objects=OBJECTS)
        with tempfile.TemporaryDirectory() as directory:
            path = write_candidate_dump(directory, payload)
            self.assertEqual(path, Path(directory) / '1' / '1_q_test.json')
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            with self.assertRaises(FileExistsError):
                write_candidate_dump(directory, payload)


if __name__ == '__main__':
    unittest.main()
