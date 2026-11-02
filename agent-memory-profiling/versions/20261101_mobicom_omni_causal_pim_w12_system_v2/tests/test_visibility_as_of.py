from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from omni_proof_sources import (image_source_header, materialize,
                                object_is_visible, visible_sessions)


def session(session_id, end_time):
    return {'session_id': session_id, 'start_time': '2025-01-01 08:00:00',
            'end_time': end_time}


def text_object(session_id, ordinal=0):
    return {'object_id': f'text:{session_id}', 'kind': 'text',
            'atom_id': f'{session_id}:turn:0', 'session_id': session_id,
            'ordinal': ordinal, 'text': session_id}


def image_object(sessions):
    refs = []
    sources = []
    for session_id in sessions:
        ref = {'normalized': f'user/{session_id}.png', 'status': 'exact'}
        refs.append(ref)
        sources.append({'atom_id': f'{session_id}:turn:0', 'session_id': session_id,
                        'role': 'user', 'ordinal': len(sources), 'time': [],
                        'image_ref': ref})
    return {'object_id': 'image:same', 'kind': 'image', 'source_sha256': 'a' * 64,
            'image_ref': refs[0], 'sources': sources}


class VisibilityAsOfTests(unittest.TestCase):
    def test_visible_sessions_uses_complete_session_end_before_question(self):
        graph = {'sessions': [
            session('early', '2025-01-01 10:00:00'),
            session('boundary', '2025-01-01 12:00:00'),
            session('late', '2025-01-01 13:00:00'),
        ]}
        question = {'effective_timestamp': '2025-01-01 12:00:00'}

        visible = visible_sessions(graph, question)

        self.assertEqual(visible, frozenset({'early', 'boundary'}))
        self.assertEqual(visible_sessions(
            graph, {'effective_timestamp': '2025-01-01 09:00:00'}), frozenset())

    def test_visible_sessions_rejects_bad_timestamp_forms(self):
        graph = {'sessions': [session('early', '2025-01-01 10:00:00')]}
        with self.assertRaises(ValueError):
            visible_sessions(graph, {
                'effective_timestamp': '2025-01-01T10:00:00+08:00'})
        with self.assertRaises(ValueError):
            visible_sessions({'sessions': [
                session('early', '2025-01-01T10:00:00+08:00')]},
                {'effective_timestamp': '2025-01-01 12:00:00'})

    def test_image_is_visible_when_any_source_session_is_visible(self):
        image = image_object(['past', 'future'])

        self.assertTrue(object_is_visible(image, frozenset({'past'})))
        self.assertFalse(object_is_visible(image, frozenset({'unrelated'})))

    def test_materialize_rejects_invisible_objects(self):
        objects = {'text:past': text_object('past'),
                   'image:same': image_object(['future'])}

        with self.assertRaisesRegex(ValueError, 'invisible object'):
            materialize(objects, ['text:past'],
                        visible_session_ids=frozenset({'future'}))
        with self.assertRaisesRegex(ValueError, 'invisible object'):
            materialize(objects, ['image:same'],
                        visible_session_ids=frozenset({'past'}))

    def test_image_header_and_hint_only_expose_visible_sources(self):
        image = image_object(['past', 'future'])
        objects = {'image:same': image}
        hidden_hint = image['sources'][1]

        materialized = materialize(
            objects, ['image:same'], source_hints={'image:same': hidden_hint},
            visible_session_ids=frozenset({'past'}))

        ref = materialized['images'][0]
        self.assertEqual(ref['source_atom_id'], 'past:turn:0')
        self.assertEqual([source['session_id'] for source in ref['all_observable_sources']],
                         ['past'])
        header = image_source_header(ref)
        self.assertIn('past:turn:0', header)
        self.assertNotIn('future:turn:0', header)

    def test_no_visible_history_row_has_empty_answer_contract(self):
        from run_proof_mainline import no_visible_history_row

        row = no_visible_history_row(
            person_id='0', question_id='q0', query='question', request_index=0,
            service_seconds=0.1, visibility_protocol='as_of_question')

        self.assertEqual(row['status'], 'no_visible_history')
        self.assertEqual(row['answer'], '')
        self.assertEqual(row['stop_reason'], 'no_visible_history')
        self.assertEqual(row['visibility_protocol'], 'as_of_question')
        self.assertEqual(row['n_visible_sessions'], 0)
        self.assertEqual(row['proof_plan']['proof_status'], 'no_visible_history')


if __name__ == '__main__':
    unittest.main()
