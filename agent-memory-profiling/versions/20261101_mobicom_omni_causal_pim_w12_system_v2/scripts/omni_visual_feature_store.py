"""Bounded, inference-only reuse of ordered Qwen vision feature batches."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import time

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


class VisualFeatureStore:
    def __init__(self, capacity_bytes, model_identity, processor_identity,
                 output_dtype=torch.bfloat16, spatial_merge_size=2):
        if capacity_bytes <= 0 or not model_identity or not processor_identity or spatial_merge_size < 1:
            raise ValueError('positive visual capacity and immutable identities required')
        self.capacity_bytes = capacity_bytes
        self.model_identity = model_identity
        self.processor_identity = processor_identity
        self.output_dtype = output_dtype
        self.spatial_merge_size = spatial_merge_size
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected = 0
        self.bypasses = 0
        self.integrity_failures = 0
        self.image_hits = 0
        self.hit_transfer_seconds = 0.0
        self.miss_compute_seconds = 0.0
        self.resolve_seconds = 0.0
        self.entries = OrderedDict()
        self.last_feature_hashes = None
        self.last_bypass_kwargs = None

    @staticmethod
    def tensor_digest(tensor):
        value = tensor.detach().contiguous().cpu()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def key(self, image_sha256s, pixel_values, grid_thw):
        if (not image_sha256s or any(not value for value in image_sha256s)
                or pixel_values.ndim != 2 or not pixel_values.is_floating_point()
                or grid_thw.ndim != 2 or grid_thw.shape != (len(image_sha256s), 3)
                or grid_thw.dtype not in (torch.int32, torch.int64)):
            raise ValueError('ordered image hashes, floating patch tensor and integer grids required')
        grids = grid_thw.detach().cpu()
        counts = grids.prod(dim=-1)
        if (torch.any(grids <= 0) or counts.sum().item() != pixel_values.shape[0]
                or torch.any(counts % self.spatial_merge_size**2)):
            raise ValueError('processor patch layout differs from merge/grid identity')
        return (tuple(image_sha256s), self.tensor_digest(pixel_values), self.tensor_digest(grids),
                str(self.output_dtype), self.spatial_merge_size,
                self.model_identity, self.processor_identity)

    def get(self, key, device):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        values, hashes, size = entry
        if tuple(self.tensor_digest(value) for value in values) != hashes:
            del self.entries[key]
            self.resident_bytes -= size
            self.integrity_failures += 1
            self.misses += 1
            return None
        started = time.perf_counter()
        result = tuple(value.to(device=device, non_blocking=False, copy=True) for value in values)
        synchronize(device)
        self.hit_transfer_seconds += time.perf_counter() - started
        self.hits += 1
        self.image_hits += len(values)
        self.entries.move_to_end(key)
        return result

    def put(self, key, values):
        size = sum(value.numel() * value.element_size() for value in values)
        if size > self.capacity_bytes:
            self.rejected += 1
            return False
        stored = tuple(value.detach().to(device='cpu', copy=True).contiguous() for value in values)
        hashes = tuple(self.tensor_digest(value) for value in stored)
        old = self.entries.pop(key, None)
        if old is not None:
            self.resident_bytes -= old[2]
        while self.entries and self.resident_bytes + size > self.capacity_bytes:
            _, old = self.entries.popitem(last=False)
            self.resident_bytes -= old[2]
            self.evictions += 1
        self.entries[key] = (stored, hashes, size)
        self.resident_bytes += size
        self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
        return True

    def resolve_batch(self, pixel_values, grid_thw, image_sha256s, original_get_image_features, **kwargs):
        if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
            raise ValueError('visual feature reuse requires inference_mode')
        if (kwargs.get('return_dict', True) is not True
                or set(kwargs) - {'return_dict', 'use_cache'}
                or ('use_cache' in kwargs and not isinstance(kwargs['use_cache'], bool))):
            self.bypasses += 1
            self.last_bypass_kwargs = {name: repr(value) for name, value in kwargs.items()}
            return original_get_image_features(pixel_values, grid_thw, **kwargs)
        synchronize(pixel_values.device)
        started = time.perf_counter()
        key = self.key(image_sha256s, pixel_values, grid_thw) + (kwargs.get('use_cache'),)
        cached = self.get(key, pixel_values.device)
        if cached is not None:
            output = BaseModelOutputWithPooling(last_hidden_state=None, pooler_output=cached)
            self.last_feature_hashes = self.entries[key][1]
        else:
            compute_started = time.perf_counter()
            output = original_get_image_features(pixel_values, grid_thw, **kwargs)
            synchronize(pixel_values.device)
            self.miss_compute_seconds += time.perf_counter() - compute_started
            values = output.pooler_output
            sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
            if (not isinstance(values, (list, tuple)) or len(values) != len(image_sha256s)
                    or any(value.ndim != 2 or value.shape[0] != size
                           or value.dtype != self.output_dtype or not torch.isfinite(value).all()
                           for value, size in zip(values, sizes))):
                raise ValueError('vision encoder output differs from pinned pooled-feature contract')
            self.put(key, values)
            self.last_feature_hashes = tuple(self.tensor_digest(value) for value in values)
        synchronize(pixel_values.device)
        self.resolve_seconds += time.perf_counter() - started
        return output

    @contextmanager
    def activate(self, model, image_sha256s):
        if model.training:
            raise ValueError('visual feature reuse requires an immutable eval model')
        original = model.get_image_features
        previous = model.__dict__.get('get_image_features')

        def cached_get_image_features(pixel_values, image_grid_thw=None, **kwargs):
            if model.training:
                raise ValueError('model changed to training during visual reuse')
            return self.resolve_batch(pixel_values, image_grid_thw, image_sha256s, original, **kwargs)

        model.get_image_features = cached_get_image_features
        try:
            yield
        finally:
            if previous is None:
                del model.get_image_features
            else:
                model.get_image_features = previous

    def snapshot(self):
        result = {name: getattr(self, name) for name in (
            'capacity_bytes', 'resident_bytes', 'peak_resident_bytes', 'hits', 'misses',
            'evictions', 'rejected', 'bypasses', 'integrity_failures', 'image_hits',
            'hit_transfer_seconds', 'miss_compute_seconds', 'resolve_seconds')}
        result.update(policy='ordered-whole-batch-exact-v1', entries=len(self.entries),
                      model_identity=self.model_identity, processor_identity=self.processor_identity,
                      output_dtype=str(self.output_dtype), spatial_merge_size=self.spatial_merge_size,
                      last_bypass_kwargs=self.last_bypass_kwargs)
        return result
