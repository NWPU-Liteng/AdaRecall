from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from pathlib import Path

import torch


VERSION_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = VERSION_ROOT / "scripts" / "84_launch_v21_uot_recipe_shards.py"


def _initialize_context(device: str, retries: int) -> int:
    if not device.lower().startswith("cuda"):
        raise ValueError("the v21 UOT CUDA bootstrap requires a CUDA device")
    if retries < 1:
        raise ValueError("cuda-init-retries must be positive")
    target = torch.device(device)
    last_error = "CUDA context was not initialized"
    for attempt in range(1, retries + 1):
        try:
            torch.cuda.init()
            torch.empty(1, device=target)
            torch.cuda.synchronize(target)
            return attempt
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(1.0)
    raise RuntimeError(f"CUDA bootstrap failed after {retries} attempts for {device}: {last_error}")


def main(*, context_already_initialized: bool = False, context_attempt: int = 0) -> None:
    parser = argparse.ArgumentParser(description="v21 CUDA -c bootstrap for one isolated UOT recipe.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--cuda-init-retries", type=int, default=5)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-matrix-size", type=int, default=1024)
    args, launcher_args = parser.parse_known_args()
    if launcher_args[:1] == ["--"]:
        launcher_args = launcher_args[1:]
    attempt = context_attempt if context_already_initialized else _initialize_context(args.device, args.cuda_init_retries)
    if args.probe_only:
        if args.probe_matrix_size < 1:
            raise ValueError("probe-matrix-size must be positive")
        values = torch.randn((args.probe_matrix_size, args.probe_matrix_size), device=args.device)
        result = values @ values
        torch.cuda.synchronize(torch.device(args.device))
        print(json.dumps({
            "v21_cuda_bootstrap_probe": "passed",
            "device": args.device,
            "init_attempt": attempt,
            "device_name": torch.cuda.get_device_name(torch.device(args.device)),
            "matrix_size": args.probe_matrix_size,
            "sample": float(result[0, 0].item()),
        }, sort_keys=True))
        return
    if not launcher_args:
        raise RuntimeError("provide exactly one v21 launcher recipe invocation after --")
    previous_argv = sys.argv
    try:
        sys.argv = [str(LAUNCHER), "--in-process", "--device", args.device, *launcher_args]
        runpy.run_path(str(LAUNCHER), run_name="__main__")
    finally:
        sys.argv = previous_argv
