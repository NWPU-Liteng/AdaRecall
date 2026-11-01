"""Strict comparison of reranker visual outputs across image partitions."""
from __future__ import annotations

import torch

from run_omni_vision_partition_probe import compare_features


def compare_partition_outputs(reference, partitions):
    if not partitions or len(reference.pooler_output) != len(partitions):
        raise ValueError('reference and partition image counts differ')
    if reference.deepstack_features is None:
        raise ValueError('DeepStack outputs are required')
    stage_count = len(reference.deepstack_features)
    for output in partitions:
        if len(output.pooler_output) != 1:
            raise ValueError('one image per partition is required')
        if output.deepstack_features is None or len(output.deepstack_features) != stage_count:
            raise ValueError('reference and partition DeepStack stage counts differ')
    pooled = [compare_features(first, output.pooler_output[0])
              for first, output in zip(reference.pooler_output, partitions, strict=True)]
    intermediate = [compare_features(first, torch.cat(
        [output.deepstack_features[stage] for output in partitions], dim=0))
        for stage, first in enumerate(reference.deepstack_features)]
    return pooled, intermediate
