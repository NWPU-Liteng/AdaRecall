"""Bind reconstructed Reader inputs to evidence and original serving tensors."""
from __future__ import annotations

import torch

from omni_visual_feature_store import VisualFeatureStore


def execution_input_contract(inputs, original):
    hashes = {name: VisualFeatureStore.tensor_digest(value) for name, value in inputs.items()
              if isinstance(value, torch.Tensor)}
    if hashes != original['processor_inputs_sha256']:
        raise ValueError('reconstructed full processor operands differ from serving')
    return {'processor_inputs_sha256': hashes, 'images': original['images'],
        'embedding_producer_outputs': [{name: record[name] for name in ('producer_sha256', 'output_sha256')}
                                       for record in original['embedding_producers']],
        'independent_image_materialization_executed': False,
        'reader_state_cross_request_reuse_executed': False}
