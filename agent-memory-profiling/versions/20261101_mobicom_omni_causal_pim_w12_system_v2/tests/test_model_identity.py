import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model_identity import EMBEDDING_WEIGHT_SHA256, RERANKER_WEIGHT_SHA256, model_role_checks, require_model_roles


class ModelIdentityTests(unittest.TestCase):
    def valid_provenance(self):
        return {'embedding_weights': {'model.safetensors': EMBEDDING_WEIGHT_SHA256},
                'reranker_weights': {'model.safetensors': RERANKER_WEIGHT_SHA256},
                'embedding_pooling_contract': {
                    'word_embedding_dimension': 1024, 'pooling_mode_lasttoken': True,
                    'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False,
                    'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False,
                    'pooling_mode_weightedmean_tokens': False}}

    def test_pinned_content_and_pooling_pass(self):
        self.assertTrue(all(require_model_roles(self.valid_provenance())['checks'].values()))

    def test_renamed_reranker_does_not_become_embedding(self):
        provenance = self.valid_provenance()
        provenance['embedding_weights'] = copy.copy(provenance['reranker_weights'])
        self.assertFalse(model_role_checks(provenance)['distinct_embedding_and_reranker'])
        with self.assertRaises(ValueError):
            require_model_roles(provenance)

    def test_missing_pooling_or_weights_fails_closed(self):
        for field in ('embedding_weights', 'reranker_weights', 'embedding_pooling_contract'):
            provenance = self.valid_provenance()
            del provenance[field]
            with self.assertRaises(ValueError):
                require_model_roles(provenance)

    def test_other_pooling_cannot_silently_replace_lasttoken(self):
        provenance = self.valid_provenance()
        provenance['embedding_pooling_contract']['pooling_mode_mean_tokens'] = True
        with self.assertRaises(ValueError):
            require_model_roles(provenance)


if __name__ == '__main__':
    unittest.main()
