from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def paired_bootstrap(values: Iterable[float], *, seed: int, samples: int = 10000) -> dict[str, float | list[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("paired bootstrap requires at least one value")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[start : start + count] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def win_loss_tie(values: Iterable[float], threshold: float) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "win_rate": float(np.mean(array > threshold)),
        "loss_rate": float(np.mean(array < -threshold)),
        "tie_rate": float(np.mean(np.abs(array) <= threshold)),
    }


def strict_dominance(quality: Mapping[str, float], latency_ms: float, baseline_quality: Mapping[str, float], baseline_latency_ms: float) -> bool:
    keys = sorted(set(quality) & set(baseline_quality))
    no_worse = all(float(baseline_quality[key]) >= float(quality[key]) for key in keys) and baseline_latency_ms <= latency_ms
    strictly_better = any(float(baseline_quality[key]) > float(quality[key]) for key in keys) or baseline_latency_ms < latency_ms
    return bool(no_worse and strictly_better)
