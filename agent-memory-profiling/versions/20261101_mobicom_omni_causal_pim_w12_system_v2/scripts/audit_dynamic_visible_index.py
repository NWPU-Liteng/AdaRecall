"""Audit dynamic visible-index working sets against a persistent replay."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _record_sessions(record: dict) -> set[str]:
    sessions = set()
    if record.get("session_id") is not None:
        sessions.add(str(record["session_id"]))
    for atom_id in record.get("source_atom_ids", []) or []:
        value = str(atom_id)
        if ":turn:" in value:
            sessions.add(value.split(":turn:", 1)[0])
    for object_id in record.get("object_ids", []) or []:
        value = str(object_id)
        if value.startswith("text:") and ":turn:" in value:
            sessions.add(value.split(":turn:", 1)[0].removeprefix("text:"))
    return sessions


def audit_visible_index(replay: Path, index_root: Path, output: Path, users: set[str] | None = None) -> dict:
    manifest = json.loads((replay / "manifest.json").read_text())
    queries = [
        json.loads(line)
        for line in (replay / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    queries_by_user: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        user = str(query["person_id"])
        if users is None or user in users:
            queries_by_user[user].append(query)
    if not queries_by_user:
        raise ValueError("no selected users have replay queries")

    trace_rows = []
    user_summaries = []
    total_full_records = 0
    total_visible_records = 0
    total_full_vector_bytes = 0
    total_visible_vector_bytes = 0
    for user in sorted(queries_by_user, key=int):
        index = index_root / f"proof_index_user_{user}"
        records_path = index / "records.json"
        vectors_path = index / "vectors.npy"
        if not records_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(f"missing static index for user {user}: {index}")
        records = json.loads(records_path.read_text())
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.ndim != 2 or len(vectors) != len(records):
            raise ValueError(f"record/vector shape mismatch for user {user}")
        record_sessions = [_record_sessions(record) for record in records]
        dimension = int(vectors.shape[1])
        itemsize = int(vectors.dtype.itemsize)
        full_vector_bytes = int(vectors.nbytes)
        full_record_count = len(records)
        user_rows = []
        max_visible = 0
        min_visible = full_record_count
        for query in queries_by_user[user]:
            visible_sessions = set(query["visible_session_ids"])
            visible_indices = [
                index for index, sessions in enumerate(record_sessions)
                if sessions & visible_sessions
            ]
            visible_count = len(visible_indices)
            visible_bytes = visible_count * dimension * itemsize
            max_visible = max(max_visible, visible_count)
            min_visible = min(min_visible, visible_count)
            row = {
                "question_id": query["question_id"],
                "person_id": user,
                "checkpoint_time": query["checkpoint_time"],
                "visible_sessions": len(visible_sessions),
                "full_record_count": full_record_count,
                "visible_record_count": visible_count,
                "visible_record_fraction": visible_count / full_record_count if full_record_count else None,
                "full_vector_bytes": full_vector_bytes,
                "visible_vector_bytes": visible_bytes,
                "vector_dimension": dimension,
                "vector_dtype": str(vectors.dtype),
                "runtime_labels_read": False,
            }
            trace_rows.append(row)
            user_rows.append(row)
        total_full_records += full_record_count * len(user_rows)
        total_visible_records += sum(row["visible_record_count"] for row in user_rows)
        total_full_vector_bytes += full_vector_bytes * len(user_rows)
        total_visible_vector_bytes += sum(row["visible_vector_bytes"] for row in user_rows)
        user_summaries.append({
            "person_id": user,
            "queries": len(user_rows),
            "full_record_count": full_record_count,
            "min_visible_record_count": min_visible,
            "max_visible_record_count": max_visible,
            "mean_visible_record_count": sum(row["visible_record_count"] for row in user_rows) / len(user_rows),
            "full_vector_bytes": full_vector_bytes,
            "vector_dimension": dimension,
            "vector_dtype": str(vectors.dtype),
        })

    trace_rows.sort(key=lambda row: (int(row["person_id"]), row["checkpoint_time"], row["question_id"]))
    output.mkdir(parents=True)
    (output / "query_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace_rows)
    )
    summary = {
        "schema": "mobilemem-omni-dynamic-visible-index-audit-v1",
        "replay_manifest_artifacts_sha256": manifest["artifacts_sha256"],
        "index_root": str(index_root),
        "users": len(user_summaries),
        "queries": len(trace_rows),
        "total_query_full_records": total_full_records,
        "total_query_visible_records": total_visible_records,
        "total_query_full_vector_bytes": total_full_vector_bytes,
        "total_query_visible_vector_bytes": total_visible_vector_bytes,
        "mean_visible_record_fraction": (
            total_visible_records / total_full_records if total_full_records else None
        ),
        "mean_visible_vector_fraction": (
            total_visible_vector_bytes / total_full_vector_bytes if total_full_vector_bytes else None
        ),
        "boundary": "visible index working-set audit only; no query embedding, scan latency, PIM timing, or E2E claim",
        "runtime_labels_read": False,
        "user_summaries": user_summaries,
        "status": "pass",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", nargs="*")
    args = parser.parse_args()
    summary = audit_visible_index(
        args.replay,
        args.index_root,
        args.output,
        set(args.users) if args.users else None,
    )
    print(json.dumps({
        "status": summary["status"],
        "users": summary["users"],
        "queries": summary["queries"],
        "mean_visible_record_fraction": summary["mean_visible_record_fraction"],
        "mean_visible_vector_fraction": summary["mean_visible_vector_fraction"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
