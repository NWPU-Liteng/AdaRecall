from __future__ import annotations

import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        str(PROJECT / "models" / "models" / "Qwen3-Reranker-0.6B"),
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )
