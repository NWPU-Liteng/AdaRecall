from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_dense_contract import event_ranks


class DenseDiagnosisTests(unittest.TestCase):
    def test_event_rank_deduplicates_windows_without_losing_atom_rank(self):
        atoms = [{'event_id': 'older'}, {'event_id': 'newer'}, {'event_id': 'older'},
                 {'event_id': 'third'}]
        self.assertEqual(event_ranks(atoms, [2, 0, 3, 1]), {
            'older': {'event_rank': 1, 'best_atom_rank': 1},
            'third': {'event_rank': 2, 'best_atom_rank': 3},
            'newer': {'event_rank': 3, 'best_atom_rank': 4}})

    def test_empty_ranking_has_no_fabricated_events(self):
        self.assertEqual(event_ranks([], []), {})


if __name__ == '__main__':
    unittest.main()
