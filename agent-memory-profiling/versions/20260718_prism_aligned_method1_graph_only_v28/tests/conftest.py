from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "graph_only_main.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def synthetic_data() -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    rng = np.random.default_rng(20260718)
    centers = rng.normal(size=(8, 16)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows = []
    for index in range(160):
        value = centers[index % len(centers)] + 0.18 * rng.normal(size=16)
        rows.append(value)
    embeddings = np.asarray(rows, dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = [f"m{index:03d}" for index in range(len(embeddings))]
    namespaces = ["a"] * 128 + ["b"] * 32
    arrivals = np.asarray(list(range(1, 129)) + list(range(1, 33)), dtype=np.int64)
    return embeddings, ids, namespaces, arrivals
