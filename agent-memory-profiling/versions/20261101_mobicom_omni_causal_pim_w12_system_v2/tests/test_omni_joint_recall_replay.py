from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_omni_joint_recall import ordered_queries


class JointRecallReplayTests(unittest.TestCase):
    def test_full_development_order_and_sealed_boundary(self):
        manifests = [dict(person_id=person, split=split, question_id=identity,
                          visibility_protocol='history_end') for person, split, identity in
                     [('0', 'development', 'a'), ('3', 'validation', 'v'),
                      ('0', 'development', 'b'), ('6', 'holdout', 'h')]]
        split = {'development_people': ['0']}
        reference = dict(complete=True, question_ids=['a', 'b'], evaluation_label_files_read=[])
        self.assertEqual([row['question_id'] for row in ordered_queries(manifests, split, reference)], ['a', 'b'])
        for identities in (['b', 'a'], ['a'], ['a', 'b', 'v']):
            changed = {**reference, 'question_ids': identities}
            with self.assertRaises(ValueError):
                ordered_queries(manifests, split, changed)
        changed = deepcopy(manifests)
        changed[0]['visibility_protocol'] = 'future'
        with self.assertRaises(ValueError):
            ordered_queries(changed, split, reference)
