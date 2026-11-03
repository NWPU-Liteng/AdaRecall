"""Immutable Reader identity and activation boundary for exact visual feature reuse."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from omni_visual_feature_store import VisualFeatureStore


DEFAULT_CAPACITY_BYTES = 512 * 1024**2
IDENTITY_SUFFIXES = frozenset({'.json', '.safetensors', '.txt', '.py', '.jinja'})
PROCESSOR_FILES = (
    'config.json',
    'preprocessor_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja',
    'vocab.json',
    'merges.txt',
    'video_preprocessor_config.json',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def reader_file_hashes(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'Reader model directory does not exist: {path}')
    hashes = {
        str(item.relative_to(path)): file_sha256(item)
        for item in sorted(path.rglob('*'))
        if item.is_file() and item.suffix in IDENTITY_SUFFIXES
    }
    if 'config.json' not in hashes:
        raise ValueError('Reader identity requires config.json')
    if not any(name.endswith('.safetensors') for name in hashes):
        raise ValueError('Reader identity requires immutable safetensors weights')
    return hashes


def _processor_runtime_config(processor):
    try:
        processor_config = processor.to_dict()
        image_processor_config = processor.image_processor.to_dict()
    except (AttributeError, TypeError) as error:
        raise ValueError('processor must expose serializable processor/image_processor configs') from error
    return {'processor': processor_config, 'image_processor': image_processor_config}


def visual_feature_identities(reader_path, processor, file_hashes=None):
    hashes = reader_file_hashes(reader_path) if file_hashes is None else dict(file_hashes)
    if 'config.json' not in hashes or not any(
            name.endswith('.safetensors') for name in hashes):
        raise ValueError('Reader identity requires config.json and safetensors weights')
    runtime_config = _processor_runtime_config(processor)
    model_identity = canonical_sha256({
        'files': hashes,
        'runtime': {
            'attention': 'sdpa',
            'dtype': 'torch.bfloat16',
            'patch_projection': 'linear',
        },
    })
    processor_identity = canonical_sha256({
        'files': {name: hashes[name] for name in PROCESSOR_FILES if name in hashes},
        'runtime': runtime_config,
    })
    return model_identity, processor_identity, hashes, runtime_config


def create_visual_feature_cache(enabled, reader_path, processor, model,
                                capacity_bytes=DEFAULT_CAPACITY_BYTES):
    if enabled not in {'off', 'on'}:
        raise ValueError("vision_feature_cache must be 'off' or 'on'")
    if enabled == 'off':
        return None
    return VisualFeatureCache(reader_path, processor, model,
                              capacity_bytes=capacity_bytes)


def _visual_module(model):
    for candidate in (model, getattr(model, 'model', None)):
        visual = getattr(candidate, 'visual', None)
        if visual is not None:
            return visual
    raise ValueError('Reader must expose a visual encoder at model.visual or model.model.visual')


def _feature_boundary(model):
    for candidate in (getattr(model, 'model', None), model):
        if callable(getattr(candidate, 'get_image_features', None)):
            return candidate
    raise ValueError('Reader must expose get_image_features at model or model.model')


class VisualFeatureCache:
    """Process-local exact cache around the final Reader visual encoder boundary."""

    def __init__(self, reader_path, processor, model,
                 capacity_bytes=DEFAULT_CAPACITY_BYTES):
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int):
            raise ValueError('visual feature cache capacity must be an integer byte count')
        if model.training:
            raise ValueError('visual feature reuse requires an eval Reader')
        self.reader_path = str(Path(reader_path).resolve())
        model_identity, processor_identity, file_hashes, runtime_config = (
            visual_feature_identities(self.reader_path, processor))
        self.model_identity = model_identity
        self.processor_identity = processor_identity
        self.file_hashes = file_hashes
        self.processor_runtime_config = runtime_config
        self.capacity_bytes = capacity_bytes
        self.no_image_requests = 0
        visual = _visual_module(model)
        self.store = VisualFeatureStore(
            capacity_bytes,
            model_identity,
            processor_identity,
            output_dtype=visual.dtype,
            spatial_merge_size=visual.spatial_merge_size,
        )

    @contextmanager
    def activate(self, model, image_sha256s):
        ordered = tuple(image_sha256s)
        if not ordered:
            self.no_image_requests += 1
            yield
            return
        with self.store.activate(_feature_boundary(model), ordered):
            yield

    def snapshot(self):
        return {
            **self.store.snapshot(),
            'no_image_requests': self.no_image_requests,
            'reader_path': self.reader_path,
            'reader_files_sha256': self.file_hashes,
            'processor_runtime_config': self.processor_runtime_config,
            'key_scope': ('ordered image-byte sha256 + processor config/runtime identity + '
                          'Reader file identity'),
            'boundary': ('process-local CPU bounded LRU over whole-batch '
                         'get_image_features pooler_output; text embeddings, 3D position ids, '
                         'image-token scatter, and language-model execution unchanged'),
        }


def tagged_visual_feature_cache(cache):
    return None if cache is None else cache.snapshot()


def cache_counter_delta(before, after):
    if before is None or after is None:
        return None
    names = ('hits', 'misses', 'evictions', 'rejected', 'bypasses',
             'integrity_failures', 'image_hits', 'no_image_requests')
    return {name: int(after[name]) - int(before[name]) for name in names}
