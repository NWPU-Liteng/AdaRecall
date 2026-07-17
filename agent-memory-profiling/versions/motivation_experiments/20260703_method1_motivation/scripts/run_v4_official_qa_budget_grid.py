#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parent.parent
CONFIG_DIR = VERSION_ROOT / "config" / "v4_official_qa_budget_grid"
RUNNER = PROJECT_ROOT / "experiments" / "motivation_memory_selection" / "profile_real_agent_v4.py"


def _load_manifest() -> list[dict]:
    return json.loads((CONFIG_DIR / "manifest.json").read_text(encoding="utf-8"))


def _output_dir(tier: str, dataset: str, budget: int) -> Path:
    return VERSION_ROOT / "runs" / f"v4_{tier}_{dataset}_budget{budget}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V4 official-QA budget grid.")
    parser.add_argument("--tier", choices=["smoke", "formal"])
    parser.add_argument("--dataset", choices=["locomo", "longmemeval_s"])
    parser.add_argument("--budget", type=int, nargs="+")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    manifest = _load_manifest()
    selected = [
        item
        for item in manifest
        if (args.tier is None or item["tier"] == args.tier)
        and (args.dataset is None or item["dataset"] == args.dataset)
        and (args.budget is None or item["budget"] in args.budget)
    ]
    if not selected:
        raise SystemExit("No budget-grid runs matched the requested filters.")

    for item in selected:
        config_path = PROJECT_ROOT / item["config_path"]
        output_dir = _output_dir(item["tier"], item["dataset"], item["budget"])
        cmd = [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--manifest",
            str(PROJECT_ROOT / item["runtime_manifest"]),
            "--output-dir",
            str(output_dir),
        ]
        env = os.environ.copy()
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            env["LOCAL_QWEN_CUDA_DEVICE"] = "0"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        print(f"START tier={item['tier']} dataset={item['dataset']} budget={item['budget']} output={output_dir}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
        print(f"DONE tier={item['tier']} dataset={item['dataset']} budget={item['budget']} output={output_dir}")


if __name__ == "__main__":
    main()
