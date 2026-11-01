from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_image_selection import select_dialogue_images


class OmniImageSelectionTests(unittest.TestCase):
    def atom(self, session, member):
        return {'session_id': session, 'atom_id': f'{session}:{member}',
                'image_refs': [{'normalized': member, 'status': 'exact'}]}

    def test_round_robin_retains_observable_provenance_and_no_session_metadata(self):
        atoms = {'a': [self.atom('a', 'one'), self.atom('a', 'two')],
                 'b': [self.atom('b', 'three')]}
        selected = select_dialogue_images(['a', 'b'], atoms, 2)
        self.assertEqual([ref['normalized'] for ref in selected], ['one', 'three'])
        self.assertEqual([ref['source_session_id'] for ref in selected], ['a', 'b'])
        self.assertTrue(all(ref['selection_source'] == 'observable_dialogue_turn' for ref in selected))

    def test_duplicates_and_missing_do_not_consume_budget(self):
        missing = self.atom('a', 'missing')
        missing['image_refs'][0]['status'] = 'missing'
        atoms = {'a': [self.atom('a', 'same'), missing],
                 'b': [self.atom('b', 'same'), self.atom('b', 'other')]}
        self.assertEqual([ref['normalized'] for ref in select_dialogue_images(['a', 'b'], atoms, 2)],
                         ['same', 'other'])

    def test_invalid_budget_and_identity_fail_closed(self):
        for sessions, atoms, limit in [(['a'], {}, 0), (['a', 'a'], {}, 1),
                                       (['a'], {'a': [self.atom('b', 'wrong')]}, 1)]:
            with self.assertRaises(ValueError):
                select_dialogue_images(sessions, atoms, limit)


if __name__ == '__main__':
    unittest.main()
