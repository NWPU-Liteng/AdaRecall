#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-medmmb-v4")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/medmmb-v4-cache")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"


def save_figure(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.pdf")

