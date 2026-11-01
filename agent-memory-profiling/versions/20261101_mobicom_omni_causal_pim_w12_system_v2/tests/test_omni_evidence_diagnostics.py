from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_omni_cohort_evidence import diagnose_row


class EvidenceDiagnosticsTests(unittest.TestCase):
    def test_distinguishes_retrieval_pruning_and_selection_reference_gaps(self):
        atoms = {'image': {'session_id': 's', 'image_refs': [{'normalized': 'image.png', 'status': 'exact'}]},
                 'text': {'session_id': 's', 'image_refs': []}}
        row = {'question_id': 'q', 'retrieved_session_ids': ['s'], 'images': [],
               'hypergraph_evidence_plan': {'accepted_atom_ids': ['text'],
                                           'candidate_replay': [{'atom_ids': ['text', 'image']}]}}
        label = {'source_session_ids': ['s'], 'evidence_image_refs': [{'normalized': 'image.png'}]}
        result = diagnose_row(row, label, atoms)
        self.assertEqual(result['image_reference_gap_category'], 'image_reference_not_selected_within_budget')
        self.assertEqual(result['source_session_recall_selected'], 1.0)
        self.assertEqual(result['label_image_recall_candidates'], 1.0)
        row['hypergraph_evidence_plan']['candidate_replay'] = [{'atom_ids': ['text']}]
        self.assertEqual(diagnose_row(row, label, atoms)['image_reference_gap_category'],
                         'image_reference_missing_from_bounded_candidates')
        row['retrieved_session_ids'] = []
        self.assertEqual(diagnose_row(row, label, atoms)['image_reference_gap_category'],
                         'image_reference_missing_from_retrieved_dialogue')
        label['evidence_image_refs'] = []
        self.assertIsNone(diagnose_row(row, label, atoms)['label_image_recall_selected'])


if __name__ == '__main__':
    unittest.main()
