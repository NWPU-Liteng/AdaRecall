#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.synthetic_data import generate_memories, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/generate_memories.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    for size in args.sizes:
        rows = generate_memories(size=size, seed=args.seed)
        output = Path("data") / f"synthetic_memories_{size}.jsonl"
        write_jsonl(str(output), rows)
        logging.info("wrote %s rows to %s", size, output)
        print(f"Wrote {size} memories to {output}")


if __name__ == "__main__":
    main()
