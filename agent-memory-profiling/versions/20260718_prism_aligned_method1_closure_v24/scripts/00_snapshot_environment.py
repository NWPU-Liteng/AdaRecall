from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from common import VERSION_ROOT, add_common_args, load_config, save_resolved_config
from v24.io import write_json


def run(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    save_resolved_config(config, VERSION_ROOT / "artifacts" / "environment" / "resolved_config.json")
    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "config_hash": __import__("common").config_hash(config),
    }
    commands = {
        "conda_env_list": ["conda", "env", "list"],
        "conda_explicit": ["conda", "list", "-n", "agent-memory-profiling", "--explicit"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "gpu": ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
    }
    for name, command in commands.items():
        code, output = run(command)
        environment[name] = {"returncode": code, "command": command, "output": output}
        if name == "conda_explicit":
            (VERSION_ROOT / "artifacts" / "environment" / "conda_explicit_before.txt").write_text(output, encoding="utf-8")
        elif name == "pip_freeze":
            (VERSION_ROOT / "artifacts" / "environment" / "pip_freeze_before.txt").write_text(output, encoding="utf-8")
        elif name == "gpu":
            (VERSION_ROOT / "artifacts" / "environment" / "gpu_before.txt").write_text(output, encoding="utf-8")
    # No package mutation is performed by v24. Keep an explicit after snapshot so a later
    # dependency change can be audited against this baseline.
    for before, after in (("conda_explicit_before.txt", "conda_explicit_after.txt"), ("pip_freeze_before.txt", "pip_freeze_after.txt"), ("gpu_before.txt", "gpu_after.txt")):
        source = VERSION_ROOT / "artifacts" / "environment" / before
        target = VERSION_ROOT / "artifacts" / "environment" / after
        if source.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        import torch

        environment["torch_runtime"] = {
            "module": str(torch.__file__), "version": torch.__version__, "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()), "cuda_device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        environment["torch_runtime"] = {"error": repr(exc)}
    try:
        import transformers

        environment["transformers_runtime"] = {"module": str(transformers.__file__), "version": transformers.__version__}
    except Exception as exc:
        environment["transformers_runtime"] = {"error": repr(exc)}
    write_json(VERSION_ROOT / "artifacts" / "environment" / "snapshot.json", environment)
    print(json.dumps({"status": "completed", "torch_runtime": environment.get("torch_runtime")}, sort_keys=True))


if __name__ == "__main__":
    main()
