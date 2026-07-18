from __future__ import annotations

import argparse
import json
import time

from common import PROJECT_ROOT, VERSION_ROOT, load_yaml, write_json
from src.qwen_reranker import Qwen3SuffixPreservingReranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_yaml(VERSION_ROOT / "configs" / "qwen_suffix_preserving.yaml")
    model_cfg = config["model"]
    reranker = Qwen3SuffixPreservingReranker(
        PROJECT_ROOT / model_cfg["path"],
        device=args.device,
        dtype=str(model_cfg["dtype"]),
        max_length=int(model_cfg["max_length"]),
        batch_size=int(model_cfg["batch_size"]),
        instruction=str(model_cfg["instruction"]),
        document_head_fraction=float(config["packing"]["document_head_fraction"]),
    )
    base = reranker.smoke()
    long_document = "unrelated history " * 2600 + "Valsartan was prescribed for hypertension."
    started = time.perf_counter_ns()
    long_output = reranker.score_pairs("Which medication treats hypertension?", [long_document])
    reranker.torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    packing = long_output.packing[0]
    result = {
        "schema_version": "method1-v30-corrected-qwen-cuda-smoke-1",
        "passed": bool(
            base["passed"]
            and len(long_output.scores) == 1
            and 0.0 <= long_output.scores[0] <= 1.0
            and packing.truncated
            and packing.packed_tokens == int(model_cfg["max_length"])
        ),
        "base_smoke": base,
        "long_prompt": {
            "score": long_output.scores[0],
            "elapsed_ms": elapsed_ms,
            "original_tokens": packing.original_tokens,
            "packed_tokens": packing.packed_tokens,
            "document_tokens": packing.document_tokens,
            "document_tokens_kept": packing.document_tokens_kept,
            "truncated": packing.truncated,
            "suffix_tokens": packing.suffix_tokens,
        },
        "device": reranker.device,
        "peak_gpu_memory_bytes": int(reranker.torch.cuda.max_memory_allocated()),
    }
    write_json(VERSION_ROOT / "artifacts" / "frozen" / "qwen_control_cuda_smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
