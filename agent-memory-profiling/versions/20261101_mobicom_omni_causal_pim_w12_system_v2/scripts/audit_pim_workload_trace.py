from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pim_backend.workload_trace import audit_workload_trace


def main():
    parser = argparse.ArgumentParser(description="Qualify observed memory waits on a request critical path")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-critical-wait-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output path required")
    raw = args.trace.read_bytes()
    result = audit_workload_trace(
        json.loads(raw),
        minimum_critical_wait_fraction=args.minimum_critical_wait_fraction,
    )
    result["trace_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "qualified_for_candidate_design", "critical_path_memory_wait_fraction",
        "pim_eligible_read_bytes", "pim_speedup_claim_allowed") if key in result}, indent=2))


if __name__ == "__main__":
    main()
