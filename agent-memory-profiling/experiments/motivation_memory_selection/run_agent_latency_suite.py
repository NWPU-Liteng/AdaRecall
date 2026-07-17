#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = PROJECT_ROOT / "experiments" / "motivation_memory_selection" / "configs" / "agent_latency_suite_20260615"


def _load_manifest() -> list[dict]:
    manifest_path = SUITE_DIR / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tier", choices=["smoke", "formal"])
    parser.add_argument("--agent-variant", choices=["mock_agent", "real_agent", "real_agent_v2"])
    parser.add_argument("--dataset", choices=["locomo", "longmemeval", "msc"])
    parser.add_argument("--mode", choices=["no_rerank", "heuristic_rerank", "cross_encoder_rerank"])
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest()
    selected = [
        item
        for item in manifest
        if (args.run_tier is None or item["run_tier"] == args.run_tier)
        and (args.agent_variant is None or item["agent_variant"] == args.agent_variant)
        and (args.dataset is None or item["dataset"] == args.dataset)
        and (args.mode is None or item["mode"] == args.mode)
    ]

    if not selected:
        raise SystemExit("No suite runs matched the requested filters.")

    runner = PROJECT_ROOT / "experiments" / "motivation_memory_selection" / "run_motivation_profile.py"
    for item in selected:
        cmd = [sys.executable, str(runner), "--config", str(PROJECT_ROOT / item["config_path"])]
        if args.quick:
            cmd.append("--quick")
        env = os.environ.copy()
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            env["LOCAL_QWEN_CUDA_DEVICE"] = "0"
        print(
            f"START run_tier={item['run_tier']} agent_variant={item['agent_variant']} "
            f"dataset={item['dataset']} mode={item['mode']} gpu={args.gpu}"
        )
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
        print(
            f"DONE run_tier={item['run_tier']} agent_variant={item['agent_variant']} "
            f"dataset={item['dataset']} mode={item['mode']} gpu={args.gpu}"
        )


if __name__ == "__main__":
    main()
