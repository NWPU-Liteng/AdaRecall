"""Content-based roles for the frozen neural models, independent of folder names."""
from __future__ import annotations


EMBEDDING_WEIGHT_SHA256 = '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
RERANKER_WEIGHT_SHA256 = '27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b'


def model_role_checks(provenance):
    embedding = provenance.get('embedding_weights', {})
    reranker = provenance.get('reranker_weights', {})
    pooling = provenance.get('embedding_pooling_contract', {})
    return {
        'embedding_weight_identity': embedding.get('model.safetensors') == EMBEDDING_WEIGHT_SHA256,
        'reranker_weight_identity': reranker.get('model.safetensors') == RERANKER_WEIGHT_SHA256,
        'distinct_embedding_and_reranker': bool(embedding.get('model.safetensors')) and
            embedding.get('model.safetensors') != reranker.get('model.safetensors'),
        'embedding_lasttoken_pooling': pooling.get('word_embedding_dimension') == 1024
            and pooling.get('pooling_mode_lasttoken') is True
            and all(pooling.get(name) is False for name in (
                'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens',
                'pooling_mode_mean_sqrt_len_tokens', 'pooling_mode_weightedmean_tokens')),
    }


def require_model_roles(provenance):
    checks = model_role_checks(provenance)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f'frozen model role mismatch: {failed}')
    return {'schema': 'frozen-model-roles-v1', 'checks': checks}
