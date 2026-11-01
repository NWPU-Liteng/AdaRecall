from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.microbatch_opportunity import audit_microbatch_opportunity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing audit")

    source_bytes = args.input.read_bytes()
    source = json.loads(source_bytes)
    result = audit_microbatch_opportunity(source)
    result["input_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    result["input_path"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "qualified", "status", "has_natural_microbatch_opportunity", "paired_smoke_allowed",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
