import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import report_object_reuse as object_reuse


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        self.assert_no_special_tokens(add_special_tokens)
        return list(text)

    @staticmethod
    def assert_no_special_tokens(add_special_tokens):
        if add_special_tokens:
            raise AssertionError('object accounting must not add special tokens')


def proof_row(question_id, *, request_index=0, person_id='0', atoms=(),
              images=(), visual_tokens=None):
    text = ''.join(
        f'[{atom_id.split(":", 1)[0]}][{atom_id}][object-{index}]'
        for index, atom_id in enumerate(atoms))
    if visual_tokens is None:
        visual_tokens = sum(cost for _, cost in images)
    object_ids = ([f'text:{atom_id}' for atom_id in atoms]
                  + [f'image:{identity}' for identity, _ in images])
    return {
        'question_id': question_id,
        'request_index': request_index,
        'person_id': person_id,
        'materialized_object_ids': object_ids,
        'proof_plan': {
            'accepted_atom_ids': list(atoms),
            'accepted_object_ids': object_ids,
            'assessment': {
                'text': text,
                'text_tokens': len(text),
                'visual_tokens': visual_tokens,
                'image_count': len(images),
            },
        },
    }


def shard_payload(person_id, shard_index, shard_count, rows, *, capture_only=False):
    return {
        'schema': 'run-test',
        'complete': True,
        'capacity_capture_only': capture_only,
        'person_id': person_id,
        'shard_index': shard_index,
        'shard_count': shard_count,
        'reader_model': 'reader-under-test',
        'question_ids': [row['question_id'] for row in rows],
        'rows': rows,
    }


def shard_row(question_id, request_index, person_id='0'):
    return {
        'question_id': question_id,
        'request_index': request_index,
        'person_id': person_id,
        'materialized_object_ids': [],
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload))


class ObjectReuseAnalysisTests(unittest.TestCase):
    def test_text_markers_are_counted_per_object(self):
        row = proof_row('q-text', atoms=('s1:a', 's2:b'))

        costs = object_reuse.extract_text_costs(row, CharacterTokenizer())

        self.assertEqual(costs, {
            'text:s1:a': len('[s1][s1:a][object-0]'),
            'text:s2:b': len('[s2][s2:b][object-1]'),
        })
        self.assertEqual(sum(costs.values()), row['proof_plan']['assessment']['text_tokens'])

    def test_exact_image_costs_are_used_when_complete(self):
        first, second = 'a' * 64, 'b' * 64
        row = proof_row('q-image', atoms=('s1:a',),
                        images=((first, 7), (second, 11)))

        result = object_reuse.analyze_question(
            row, CharacterTokenizer(), {first: 7, second: 11})

        by_identity = {item['object_id']: item['tokens'] for item in result['objects']}
        self.assertEqual(by_identity[f'image:{first}'], 7)
        self.assertEqual(by_identity[f'image:{second}'], 11)
        self.assertEqual(result['image_token_cost_source'], 'image_costs_json')
        self.assertEqual(result['missing_image_cost_count'], 0)

    def test_missing_image_costs_use_marked_equal_share_estimate(self):
        first, second = 'a' * 64, 'b' * 64
        row = proof_row('q-estimate', images=((first, 0), (second, 0)),
                        visual_tokens=5)

        result = object_reuse.analyze_question(row, CharacterTokenizer(), {})

        by_identity = {item['object_id']: item['tokens'] for item in result['objects']}
        self.assertEqual(by_identity[f'image:{first}'], 3)
        self.assertEqual(by_identity[f'image:{second}'], 2)
        self.assertEqual(result['image_token_cost_source'], 'equal_share_estimate')
        self.assertEqual(result['missing_image_cost_count'], 2)

    def test_duplicate_ratio_ideal_bound_and_lru_known_sequence(self):
        questions = [
            {'question_id': 'q1', 'objects': [
                {'object_id': 'text:a', 'kind': 'text', 'tokens': 10},
                {'object_id': 'text:b', 'kind': 'text', 'tokens': 20}]},
            {'question_id': 'q2', 'objects': [
                {'object_id': 'text:a', 'kind': 'text', 'tokens': 10},
                {'object_id': 'text:c', 'kind': 'text', 'tokens': 30}]},
            {'question_id': 'q3', 'objects': [
                {'object_id': 'text:b', 'kind': 'text', 'tokens': 20},
                {'object_id': 'text:a', 'kind': 'text', 'tokens': 10}]},
        ]

        result = object_reuse.analyze_user('0', questions, [2, 3])

        self.assertEqual(result['selected_token_volume'], 100)
        self.assertEqual(result['repeated_object_occurrences'], 5)
        self.assertEqual(result['repeated_object_token_volume'], 70)
        self.assertEqual(result['duplicate_object_token_ratio'], 0.7)
        self.assertEqual(result['ideal_object_kv_reuse'], {
            'saved_tokens_upper_bound': 40,
            'saved_token_ratio': 0.4,
            'cold_first_occurrence_tokens': 60,
        })
        capacity_two, capacity_three = result['lru_object_capacity']
        self.assertEqual(capacity_two['capacity_objects'], 2)
        self.assertEqual(capacity_two['saved_tokens'], 10)
        self.assertEqual(capacity_two['hit_occurrences'], 1)
        self.assertEqual(capacity_two['miss_occurrences'], 5)
        self.assertEqual(capacity_two['evictions'], 3)
        self.assertEqual(capacity_two['final_resident_objects'], 2)
        self.assertTrue(capacity_three['unlimited_capacity_equivalent'])
        self.assertEqual(capacity_three['saved_tokens'], 40)
        self.assertEqual(capacity_three['evictions'], 0)

    def test_single_and_multi_shard_question_order_are_reconstructed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_json(run_dir / 'user_0_shard_0.json', shard_payload(
                '0', 0, 1, [shard_row('q0', 0), shard_row('q1', 1)]))
            single, reader_model, _ = object_reuse.load_run_shards(run_dir)
            self.assertEqual(single['0']['question_ids'], ['q0', 'q1'])
            self.assertEqual(reader_model, 'reader-under-test')

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            write_json(run_dir / 'user_0_shard_0.json', shard_payload(
                '0', 0, 2, [shard_row('q0', 0), shard_row('q2', 1)]))
            write_json(run_dir / 'user_0_shard_1.json', shard_payload(
                '0', 1, 2, [shard_row('q1', 0), shard_row('q3', 1)]))
            write_json(run_dir / 'user_0_shard_0.partial.json', shard_payload(
                '0', 0, 2, [shard_row('partial', 0)]))

            multi, _, files = object_reuse.load_run_shards(run_dir)

            self.assertEqual(multi['0']['question_ids'], ['q0', 'q1', 'q2', 'q3'])
            self.assertEqual(len(files), 2)
            self.assertEqual(
                [row['question_id'] for row in multi['0']['questions']],
                ['q0', 'q1', 'q2', 'q3'])

    def test_invalid_shard_fields_raise_value_errors(self):
        invalid_updates = {
            'none shard index': {'shard_index': None},
            'string shard index': {'shard_index': '0'},
            'duplicate request index': None,
            'row user mismatch': None,
        }
        for name, update in invalid_updates.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                rows = [shard_row('q0', 0), shard_row('q1', 1)]
                payload = shard_payload('0', 0, 1, rows)
                if name == 'duplicate request index':
                    rows[1]['request_index'] = 0
                elif name == 'row user mismatch':
                    rows[1]['person_id'] = '1'
                else:
                    payload.update(update)
                write_json(Path(directory) / 'user_0_shard_0.json', payload)

                with self.assertRaises(ValueError):
                    object_reuse.load_run_shards(directory)

    def test_capture_only_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / 'user_0_shard_0.json', shard_payload(
                '0', 0, 1, [shard_row('q0', 0)], capture_only=True))

            with self.assertRaisesRegex(ValueError, 'capture-only'):
                object_reuse.load_run_shards(directory)

    def test_cli_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'object_reuse.json'
            output.write_text('{"existing": true}\n')
            argv = ['report_object_reuse.py', str(Path(directory) / 'run'), '--output', str(output)]

            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                object_reuse.main()


if __name__ == '__main__':
    unittest.main()
