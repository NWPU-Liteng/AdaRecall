"""Typed parsing for default-off v2 experiment extensions."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import re


_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class ProgressiveTopk:
    lm_head: str = 'off'
    lm_head_shadow_every: int = 8
    retrieval: str = 'off'
    bits: int = 12
    group_size: int = 128
    plane_schedule: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    eps_rel: float = 1.0e-9


@dataclass(frozen=True)
class Calibration:
    calibrator_path: str | None = None
    calibrator_sha256: str | None = None
    candidate_policy: str = 'fixed48'
    alpha: float = 0.1
    min_per_demand: int = 4


@dataclass(frozen=True)
class NoisyOr:
    max_witnesses_per_demand: int = 3
    levels: int = 20
    fill_redundancy: bool = False


@dataclass(frozen=True)
class Extensions:
    visibility_protocol: str = 'history_end'
    as_of_rule: str = 'session_end_before_question'
    operation_executor: str = 'off'
    witness_values: str = 'off'
    stage_profiler: str = 'off'
    workload_trace: str = 'off'
    progressive_topk: ProgressiveTopk = field(default_factory=ProgressiveTopk)
    candidate_dump: str | None = None
    calibration: Calibration = field(default_factory=Calibration)
    join_algorithm: str = 'dfs_maxmin'
    noisy_or: NoisyOr = field(default_factory=NoisyOr)
    vision_feature_cache: str = 'off'

    def to_json(self) -> str:
        return json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False)

    def as_dict(self) -> dict:
        return json.loads(self.to_json())

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode('utf-8')).hexdigest()

    def output_contract(self) -> dict:
        return {'config': self.as_dict(), 'sha256': self.sha256()}


_EXTENSION_KEYS = frozenset({
    'visibility_protocol', 'as_of_rule', 'operation_executor', 'witness_values',
    'stage_profiler', 'workload_trace', 'progressive_topk', 'candidate_dump',
    'calibration', 'join_algorithm', 'noisy_or', 'vision_feature_cache',
})
_PROGRESSIVE_KEYS = frozenset({
    'lm_head', 'lm_head_shadow_every', 'retrieval', 'bits', 'group_size',
    'plane_schedule', 'eps_rel',
})
_CALIBRATION_KEYS = frozenset({
    'calibrator_path', 'calibrator_sha256', 'candidate_policy', 'alpha',
    'min_per_demand',
})
_NOISY_OR_KEYS = frozenset({
    'max_witnesses_per_demand', 'levels', 'fill_redundancy',
})


def _mapping(value, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f'{path} must be a mapping')
    return value


def _reject_unknown(value: Mapping, allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f'unknown keys in {path}: {unknown}')


def _choice(value: Mapping, key: str, default: str, choices: tuple[str, ...],
            path: str) -> str:
    result = value.get(key, default)
    if isinstance(result, bool):
        if result is False and 'off' in choices:
            return 'off'
        if result is True and 'on' in choices:
            return 'on'
    if not isinstance(result, str) or result not in choices:
        raise ValueError(f'{path}.{key} must be one of {choices}')
    return result


def _integer(value: Mapping, key: str, default: int, path: str,
             minimum: int = 1) -> int:
    result = value.get(key, default)
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise ValueError(f'{path}.{key} must be an integer >= {minimum}')
    return result


def _number(value: Mapping, key: str, default: float, path: str,
            *, minimum: float | None = None,
            maximum_exclusive: float | None = None) -> float:
    result = value.get(key, default)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f'{path}.{key} must be numeric')
    result = float(result)
    if minimum is not None and result < minimum:
        raise ValueError(f'{path}.{key} must be >= {minimum}')
    if maximum_exclusive is not None and result >= maximum_exclusive:
        raise ValueError(f'{path}.{key} must be < {maximum_exclusive}')
    return result


def _boolean(value: Mapping, key: str, default: bool, path: str) -> bool:
    result = value.get(key, default)
    if not isinstance(result, bool):
        raise ValueError(f'{path}.{key} must be boolean')
    return result


def _optional_string(value: Mapping, key: str, path: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str) or not result:
        raise ValueError(f'{path}.{key} must be a non-empty string or null')
    return result


def _optional_sha256(value: Mapping, key: str, path: str) -> str | None:
    result = _optional_string(value, key, path)
    if result is not None and not _SHA256_RE.fullmatch(result):
        raise ValueError(f'{path}.{key} must be a lowercase sha256 hex digest or null')
    return result


def _plane_schedule(value: Mapping, bits: int, path: str) -> tuple[int, ...]:
    result = value.get('plane_schedule', (1,) * bits)
    if (not isinstance(result, (list, tuple)) or isinstance(result, (str, bytes))
            or not result):
        raise ValueError(f'{path}.plane_schedule must be a non-empty integer list')
    planes = []
    for item in result:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f'{path}.plane_schedule entries must be positive integers')
        planes.append(item)
    if sum(planes) != bits:
        raise ValueError(f'{path}.plane_schedule must sum to bits={bits}')
    return tuple(planes)


def load_extensions(cfg_dict: Mapping) -> Extensions:
    """Parse and validate the optional extensions block."""
    if not isinstance(cfg_dict, Mapping):
        raise ValueError('configuration must be a mapping')
    raw = cfg_dict.get('extensions', {})
    value = _mapping(raw, 'extensions')
    _reject_unknown(value, _EXTENSION_KEYS, 'extensions')

    progressive = _mapping(value.get('progressive_topk', {}),
                           'extensions.progressive_topk')
    _reject_unknown(progressive, _PROGRESSIVE_KEYS, 'extensions.progressive_topk')
    bits = _integer(progressive, 'bits', 12, 'extensions.progressive_topk')
    progressive_config = ProgressiveTopk(
        lm_head=_choice(progressive, 'lm_head', 'off', ('off', 'shadow'),
                        'extensions.progressive_topk'),
        lm_head_shadow_every=_integer(
            progressive, 'lm_head_shadow_every', 8, 'extensions.progressive_topk'),
        retrieval=_choice(progressive, 'retrieval', 'off',
                          ('off', 'shadow', 'replace'), 'extensions.progressive_topk'),
        bits=bits,
        group_size=_integer(progressive, 'group_size', 128,
                            'extensions.progressive_topk'),
        plane_schedule=_plane_schedule(progressive, bits,
                                       'extensions.progressive_topk'),
        eps_rel=_number(progressive, 'eps_rel', 1.0e-9,
                        'extensions.progressive_topk', minimum=0.0),
    )

    calibration = _mapping(value.get('calibration', {}), 'extensions.calibration')
    _reject_unknown(calibration, _CALIBRATION_KEYS, 'extensions.calibration')
    calibration_config = Calibration(
        calibrator_path=_optional_string(
            calibration, 'calibrator_path', 'extensions.calibration'),
        calibrator_sha256=_optional_sha256(
            calibration, 'calibrator_sha256', 'extensions.calibration'),
        candidate_policy=_choice(
            calibration, 'candidate_policy', 'fixed48',
            ('fixed48', 'conformal'), 'extensions.calibration'),
        alpha=_number(calibration, 'alpha', 0.1, 'extensions.calibration',
                      minimum=0.0, maximum_exclusive=1.0),
        min_per_demand=_integer(
            calibration, 'min_per_demand', 4, 'extensions.calibration'),
    )
    if calibration_config.candidate_policy == 'conformal':
        if (calibration_config.calibrator_path is None
                or calibration_config.calibrator_sha256 is None):
            raise ValueError(
                'extensions.calibration conformal policy requires '
                'calibrator_path and calibrator_sha256')
        if calibration_config.min_per_demand > 48:
            raise ValueError(
                'extensions.calibration.min_per_demand must be <= 48')

    noisy_or = _mapping(value.get('noisy_or', {}), 'extensions.noisy_or')
    _reject_unknown(noisy_or, _NOISY_OR_KEYS, 'extensions.noisy_or')
    noisy_or_config = NoisyOr(
        max_witnesses_per_demand=_integer(
            noisy_or, 'max_witnesses_per_demand', 3, 'extensions.noisy_or'),
        levels=_integer(noisy_or, 'levels', 20, 'extensions.noisy_or'),
        fill_redundancy=_boolean(
            noisy_or, 'fill_redundancy', False, 'extensions.noisy_or'),
    )
    join_algorithm = _choice(
        value, 'join_algorithm', 'dfs_maxmin',
        ('dfs_maxmin', 'noisy_or_saturate'), 'extensions')
    if (join_algorithm == 'noisy_or_saturate'
            and (calibration_config.calibrator_path is None
                 or calibration_config.calibrator_sha256 is None)):
        raise ValueError(
            'extensions.join_algorithm noisy_or_saturate requires '
            'calibrator_path and calibrator_sha256')

    return Extensions(
        visibility_protocol=_choice(
            value, 'visibility_protocol', 'history_end',
            ('history_end', 'as_of_question'), 'extensions'),
        as_of_rule=_choice(
            value, 'as_of_rule', 'session_end_before_question',
            ('session_end_before_question',), 'extensions'),
        operation_executor=_choice(
            value, 'operation_executor', 'off', ('off', 'on'), 'extensions'),
        witness_values=_choice(
            value, 'witness_values', 'off', ('off', 'on'), 'extensions'),
        stage_profiler=_choice(
            value, 'stage_profiler', 'off', ('off', 'on'), 'extensions'),
        workload_trace=_choice(
            value, 'workload_trace', 'off', ('off', 'on'), 'extensions'),
        progressive_topk=progressive_config,
        candidate_dump=_optional_string(value, 'candidate_dump', 'extensions'),
        calibration=calibration_config,
        join_algorithm=join_algorithm,
        noisy_or=noisy_or_config,
        vision_feature_cache=_choice(
            value, 'vision_feature_cache', 'off', ('off', 'on'), 'extensions'),
    )
