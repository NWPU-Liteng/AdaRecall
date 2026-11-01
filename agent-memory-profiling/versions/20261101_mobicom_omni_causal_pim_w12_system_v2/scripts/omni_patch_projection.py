"""Explicit full-patch Conv3d implementation choice, not a numerical equivalence claim."""
from __future__ import annotations

import torch
import torch.nn.functional as functional


def configure_patch_projection(visual, mode):
    if mode not in {'conv', 'linear'}:
        raise ValueError('patch projection must be conv or linear')
    projection = visual.patch_embed.proj
    if mode == 'conv':
        return
    if (not isinstance(projection, torch.nn.Conv3d) or projection.groups != 1
            or tuple(projection.padding) != (0, 0, 0)
            or tuple(projection.dilation) != (1, 1, 1)
            or tuple(projection.stride) != tuple(projection.kernel_size)):
        raise ValueError('linear patch projection requires nonoverlapping full-patch Conv3d')

    def linear_projection(hidden_states):
        flattened = hidden_states.reshape(-1, projection.weight[0].numel())
        return functional.linear(flattened.to(dtype=projection.weight.dtype),
                                 projection.weight.flatten(1), projection.bias)

    projection.forward = linear_projection
