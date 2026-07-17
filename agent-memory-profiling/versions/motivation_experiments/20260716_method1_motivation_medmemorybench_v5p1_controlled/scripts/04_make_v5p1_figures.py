#!/usr/bin/env python3
"""Compatibility wrapper for the revised standalone figure set."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISED = ROOT / "runs/v5p1_figures_revised"


def main() -> None:
    subprocess.run([sys.executable, str(REVISED / "run_all.py")], check=True, cwd=REVISED)
    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / "figures/revised").glob("*"):
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)
    print(f"revised_figures={ROOT / 'figures/revised'}")


if __name__ == "__main__":
    main()
