"""Bounded exact-prefix snapshots for hybrid attention and recurrent state."""
from __future__ import annotations

import copy
import hashlib
import time

import torch


def map_tensors(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, list):
        return [map_tensors(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensors(item, transform) for item in value)
    if isinstance(value, dict):
        return {key: map_tensors(item, transform) for key, item in value.items()}
    return value


def state_bytes(cache) -> int:
    tensors = []
    for layer in cache.layers:
        map_tensors(vars(layer), lambda tensor: tensors.append(tensor))
    unique = {tensor.data_ptr(): tensor for tensor in tensors if tensor.numel()}
    return sum(tensor.numel() * tensor.element_size() for tensor in unique.values())


class PrefixStateSnapshot:
    def __init__(self, cache, input_ids: torch.Tensor, rope_deltas, *, capacity_bytes: int):
        started = time.perf_counter()
        self.bytes = state_bytes(cache) + input_ids.numel() * input_ids.element_size()
        if rope_deltas is not None:
            self.bytes += rope_deltas.numel() * rope_deltas.element_size()
        if self.bytes > capacity_bytes:
            raise ValueError("prefix state exceeds byte capacity")
        self.tokens = int(input_ids.shape[1])
        if cache.get_seq_length() != self.tokens:
            raise ValueError("cache sequence does not match immutable prefix")
        self.input_ids = input_ids.detach().cpu().clone()
        self.prefix_fingerprint = hashlib.sha256(self.input_ids.numpy().tobytes()).hexdigest()
        self.rope_deltas = rope_deltas.detach().cpu().clone() if rope_deltas is not None else None
        self.cache = copy.copy(cache)
        self.cache.layers = []
        for layer in cache.layers:
            cloned = copy.copy(layer)
            cloned.__dict__.update(map_tensors(vars(layer), lambda tensor: tensor.detach().to("cpu", copy=True)))
            self.cache.layers.append(cloned)
        self.snapshot_seconds = time.perf_counter() - started

    def compatible(self, input_ids: torch.Tensor, *, has_suffix_images: bool = False) -> bool:
        return (not has_suffix_images and input_ids.shape[1] >= self.tokens
                and torch.equal(input_ids[:, :self.tokens].detach().cpu(), self.input_ids))

    def restore(self, device):
        cache = copy.copy(self.cache)
        cache.layers = []
        for layer in self.cache.layers:
            cloned = copy.copy(layer)
            cloned.__dict__.update(map_tensors(
                vars(layer), lambda tensor: tensor.to(device=device, copy=True),
            ))
            if hasattr(cloned, "device"):
                cloned.device = torch.device(device)
            cache.layers.append(cloned)
        rope_deltas = self.rope_deltas.to(device) if self.rope_deltas is not None else None
        return cache, rope_deltas
