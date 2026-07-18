from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone

from common import REPOSITORY_ROOT, VERSION_ROOT, write_json


def capture(command: list[str]) -> str:
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


def capture_optional(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.returncode, completed.stdout


def main() -> None:
    import numpy
    import sklearn
    import torch
    import transformers

    if os.environ.get("CONDA_DEFAULT_ENV") != "agent-memory-profiling":
        raise RuntimeError("environment snapshot must run inside agent-memory-profiling")
    target = VERSION_ROOT / "artifacts" / "environment"
    target.mkdir(parents=True, exist_ok=True)
    (target / "conda_explicit_before.txt").write_text(capture(["conda", "list", "--explicit"]), encoding="utf-8")
    (target / "pip_freeze_before.txt").write_text(capture(["python", "-m", "pip", "freeze"]), encoding="utf-8")
    nvidia_smi_returncode, nvidia_smi_output = capture_optional(["nvidia-smi"])
    (target / "gpu_before.txt").write_text(nvidia_smi_output, encoding="utf-8")
    write_json(
        target / "snapshot.json",
        {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "nvidia_smi_returncode": nvidia_smi_returncode,
            "attention_implementation": "eager",
            "core_dependencies_changed": False,
        },
    )
    print("GATE_A_ENVIRONMENT_SNAPSHOT_COMPLETE")


if __name__ == "__main__":
    main()
