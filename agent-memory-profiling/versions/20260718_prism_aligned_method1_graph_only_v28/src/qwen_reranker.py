from __future__ import annotations

import sys
from pathlib import Path

V24_SRC = Path(__file__).resolve().parents[2] / "20260718_prism_aligned_method1_closure_v24" / "src"
if str(V24_SRC) not in sys.path:
    sys.path.insert(0, str(V24_SRC))

from v24.reranker import Qwen3Reranker  # noqa: E402,F401

__all__ = ["Qwen3Reranker"]
