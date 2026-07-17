#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memorycd.protocol_builder import build_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["official_single", "official_cross", "causal_cross"], required=True)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=3)
    args = parser.parse_args()

    summary = build_manifests(
        users_path=args.data_root / "users" / "cross_domain_users_sampled.jsonl.gz",
        output_root=args.version_dir,
        protocol=args.protocol,
        max_users=args.max_users,
        num_test=args.num_test,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
