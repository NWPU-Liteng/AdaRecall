#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memorycd.loader import iter_user_records, load_metadata_sample
from src.memorycd.schema import DOMAINS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    users_path = args.data_root / "users" / "cross_domain_users_sampled.jsonl.gz"
    meta_dir = args.data_root / "meta"
    if not users_path.exists():
        raise SystemExit(f"Missing users file: {users_path}")

    files = {
        "users/cross_domain_users_sampled.jsonl.gz": users_path,
        **{f"meta/{domain}.jsonl.gz": meta_dir / f"{domain}.jsonl.gz" for domain in DOMAINS},
    }
    file_info = {
        rel: {"bytes": path.stat().st_size, "sha256": sha256_file(path), "exists": path.exists()}
        for rel, path in files.items()
        if path.exists()
    }

    users = 0
    interactions_by_domain: dict[str, int] = defaultdict(int)
    per_user_counts: dict[str, list[int]] = defaultdict(list)
    min_ts = None
    max_ts = None
    schema_keys: set[str] = set()
    ranking_with_negatives = 0
    for record in iter_user_records(users_path):
        users += 1
        by_domain = record.get("interactions") or {}
        for domain in DOMAINS:
            interactions = by_domain.get(domain, [])
            per_user_counts[domain].append(len(interactions))
            interactions_by_domain[domain] += len(interactions)
            for inter in interactions:
                schema_keys.update(inter.keys())
                ts = int(inter.get("timestamp") or 0)
                min_ts = ts if min_ts is None else min(min_ts, ts)
                max_ts = ts if max_ts is None else max(max_ts, ts)
                if inter.get("negative_parent_asins"):
                    ranking_with_negatives += 1
        if args.max_users is not None and users >= args.max_users:
            break

    def percentiles(values: list[int]) -> dict[str, int | None]:
        if not values:
            return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
        vals = sorted(values)
        def pct(p: float) -> int:
            return vals[min(len(vals) - 1, int(round((len(vals) - 1) * p)))]
        return {"p50": pct(0.50), "p90": pct(0.90), "p95": pct(0.95), "p99": pct(0.99), "max": vals[-1]}

    inventory = {
        "data_root": str(args.data_root),
        "users_file": str(users_path),
        "files": file_info,
        "users_scanned": users,
        "domains": DOMAINS,
        "interactions_by_domain": dict(interactions_by_domain),
        "per_user_interactions": {domain: percentiles(vals) for domain, vals in per_user_counts.items()},
        "time_span": {"min_timestamp": min_ts, "max_timestamp": max_ts},
        "interaction_schema_keys": sorted(schema_keys),
        "ranking_interactions_with_negatives": ranking_with_negatives,
        "metadata_sample_counts": {domain: len(rows) for domain, rows in load_metadata_sample(meta_dir, 2).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
