#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = VERSION_ROOT / "scripts" / "84_launch_v21_uot_recipe_shards.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preinitialize CUDA then run one v21 UOT launcher recipe in-process.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--cuda-init-retries", type=int, default=5)
    args, launcher_args = parser.parse_known_args()
    if launcher_args[:1] == ["--"]:
        launcher_args = launcher_args[1:]
    if not launcher_args:
        raise RuntimeError("provide one launcher invocation after --")
    if not args.device.lower().startswith("cuda"):
        raise RuntimeError("the CUDA bootstrap requires a CUDA device")
    target = torch.device(args.device)
    last_error = "CUDA context was not initialized"
    for attempt in range(1, args.cuda_init_retries + 1):
        try:
            torch.cuda.init()
            torch.empty(1, device=target)
            torch.cuda.synchronize(target)
            print(json.dumps({"cuda_bootstrap_preinitialized": str(target), "cuda_bootstrap_attempt": attempt}), flush=True)
            break
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt == args.cuda_init_retries:
                raise RuntimeError(f"CUDA bootstrap failed after {args.cuda_init_retries} attempts: {last_error}")
            time.sleep(1.0)
    previous_argv = sys.argv
    try:
        sys.argv = [str(LAUNCHER), "--in-process", "--device", args.device, *launcher_args]
        runpy.run_path(str(LAUNCHER), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
