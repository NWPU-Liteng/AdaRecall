import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_linear_joint_requalification import validate_run


class LinearRequalificationTests(unittest.TestCase):
    def fixture(self):
        protocol = {'max_images': 4, 'max_new_tokens': 512, 'input_budget': 8192,
            'text_budget': 3600, 'reader_image_max_pixels': 262144, 'reranker_batch_size': 2,
            'visual_recall_count': 16, 'text_recall_count': 16, 'index_image_max_pixels': 65536,
            'cohort_sha256': 'cohort', 'sources_sha256': {'scripts/reader.py': 'source'}}
        arguments = {'max_images': 4, 'max_new_tokens': 512, 'input_budget': 8192,
            'total_text_tokens': 3600, 'image_max_pixels': 262144, 'user_id': '0',
            'evidence_policy': 'provenance_milp', 'disable_edge_reuse': True,
            'visual_cache_bytes': 0, 'profile_reranker': False, 'verify_generation': False,
            'validate_bank_gemv': False, 'relevance_modality': 'joint_index',
            'joint_patch_projection': 'linear', 'relevance_batch_size': 2,
            'joint_recall_count': 16, 'joint_text_recall_count': 16,
            'joint_index_image_max_pixels': 65536}
        registration = {'arguments': arguments, 'cohort_sha256': 'cohort',
            'source_files_sha256': {'scripts/reader.py': 'source'}, 'question_ids': ['q']}
        run = {'complete': True, 'screening_run': True, 'evaluation_label_files_read': [],
            'embedding_state_reuse_enabled': False, 'visual_state_reuse': None,
            'source_files_sha256': {'scripts/reader.py': 'source'},
            'joint_patch_projection': 'linear', 'relevance_modality': 'joint_index',
            'rows': [{'question_id': 'q', 'person_id': '0', 'answer': 'answer',
                'answer_sha256': hashlib.sha256(b'answer').hexdigest(), 'service_seconds': 1.,
                'service_stage_ledger': {'disjoint_seconds': {'reader': 1.}}}]}
        return run, registration, protocol

    def check(self, run, registration, protocol):
        validate_run(run, registration, protocol, ['q'], '0', 'provenance_milp')

    def test_registered_contract_passes_without_qualifying_performance(self):
        self.check(*self.fixture())

    def test_changed_projection_and_budget_rejected(self):
        for name, value in [('joint_patch_projection', 'conv'), ('max_new_tokens', 8),
                            ('input_budget', 4096), ('profile_reranker', True)]:
            with self.subTest(name=name):
                run, registration, protocol = self.fixture()
                registration['arguments'][name] = value
                with self.assertRaises(ValueError):
                    self.check(run, registration, protocol)

    def test_denominator_source_and_answer_mutations_rejected(self):
        run, registration, protocol = self.fixture()
        variants = []
        changed = copy.deepcopy(run)
        changed['rows'] = []
        variants.append(changed)
        changed = copy.deepcopy(run)
        changed['source_files_sha256']['scripts/reader.py'] = 'changed'
        variants.append(changed)
        changed = copy.deepcopy(run)
        changed['rows'][0]['answer'] = 'changed'
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                self.check(changed, registration, protocol)

    def test_label_reads_or_unaccounted_reuse_rejected(self):
        for name, value in [('evaluation_label_files_read', ['labels.jsonl']),
                            ('embedding_state_reuse_enabled', True),
                            ('visual_state_reuse', {'hits': 1})]:
            run, registration, protocol = self.fixture()
            run[name] = value
            with self.assertRaises(ValueError):
                self.check(run, registration, protocol)

    def test_nonfinite_and_unclosed_stage_ledgers_rejected(self):
        for value in (float('nan'), -1., 2.):
            run, registration, protocol = self.fixture()
            run['rows'][0]['service_stage_ledger']['disjoint_seconds']['reader'] = value
            with self.assertRaises(ValueError):
                self.check(run, registration, protocol)


if __name__ == '__main__':
    unittest.main()
