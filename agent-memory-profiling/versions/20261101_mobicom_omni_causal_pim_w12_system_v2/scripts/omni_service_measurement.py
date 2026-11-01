"""Disjoint service timing with explicitly nested vision accounting."""
from __future__ import annotations

from contextlib import contextmanager
import math
import time


@contextmanager
def measure_vision(model, synchronize):
    original = model.get_image_features
    previous = model.__dict__.get('get_image_features')
    measurement = {'vision_in_prefill_seconds': 0., 'vision_calls': 0}

    def measured(pixel_values, image_grid_thw=None, **kwargs):
        synchronize()
        started = time.perf_counter()
        result = original(pixel_values, image_grid_thw, **kwargs)
        synchronize()
        measurement['vision_in_prefill_seconds'] += time.perf_counter() - started
        measurement['vision_calls'] += 1
        return result

    model.get_image_features = measured
    try:
        yield measurement
    finally:
        if previous is None:
            del model.get_image_features
        else:
            model.get_image_features = previous


def service_stage_ledger(service_seconds, *, evidence_seconds, input_preparation_seconds,
                         prefill_seconds, decode_seconds, answer_conversion_seconds,
                         vision_in_prefill_seconds):
    stages = {'evidence_seconds': evidence_seconds, 'input_preparation_seconds': input_preparation_seconds,
        'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
        'answer_conversion_seconds': answer_conversion_seconds}
    values = [service_seconds, vision_in_prefill_seconds, *stages.values()]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('finite nonnegative service timers required')
    residual = service_seconds - sum(stages.values())
    if residual < -1e-8 or vision_in_prefill_seconds > prefill_seconds + 1e-8:
        raise ValueError('disjoint stages and nested vision-within-prefill timers required')
    stages['timer_bracket_seconds'] = max(0., residual)
    return {'disjoint_seconds': stages,
        'nested_seconds': {'vision_in_prefill_seconds': vision_in_prefill_seconds,
                           'language_and_wrapper_prefill_seconds': prefill_seconds - vision_in_prefill_seconds},
        'total_service_seconds': service_seconds,
        'boundary': 'nested vision is already inside prefill, never add it a second time; CUDA synchronized coarse timers, same instrumentation in both serving paths'}
