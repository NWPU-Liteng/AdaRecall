#!/usr/bin/env python3
import argparse
from pathlib import Path

from profile_longgraph_memory_selection import run_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    paths = run_profile(Path(args.config), quick=args.quick)
    print(f"raw_profile={paths['raw_profile']}")
    print(f"summary_csv={paths['summary_csv']}")
    print(f"summary_md={paths['summary_md']}")
    print(f"figures_dir={paths['figures_dir']}")


if __name__ == "__main__":
    main()
