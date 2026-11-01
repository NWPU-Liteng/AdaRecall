"""Audit logical working-set and reuse properties of a persistent session replay."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _logical_atom_bytes(atom: dict) -> int:
    content = str(atom.get("content", "")).encode("utf-8")
    image_bytes = sum(
        len(json.dumps(ref, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        for ref in atom.get("image_refs", [])
    )
    metadata = {
        key: atom.get(key)
        for key in (
            "atom_id", "session_id", "ordinal", "session_ordinal", "turn_ordinal",
            "role", "content_type", "visibility_start", "visibility_end",
        )
    }
    return len(content) + image_bytes + len(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def audit_replay(replay: Path, output: Path) -> dict:
    manifest = json.loads((replay / "manifest.json").read_text())
    sessions = [json.loads(line) for line in (replay / "sessions.jsonl").read_text().splitlines() if line.strip()]
    queries = [json.loads(line) for line in (replay / "queries.jsonl").read_text().splitlines() if line.strip()]
    schedule_audit = [
        json.loads(line)
        for line in (replay / "schedule_audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    audit_by_question = {row["question_id"]: row for row in schedule_audit}
    sessions_by_user: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        sessions_by_user[str(session["person_id"])].append(session)
    queries_by_user: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        queries_by_user[str(query["person_id"])].append(query)

    user_rows = []
    all_reuse_distances = []
    total_ingest_bytes = 0
    total_visible_bytes = 0
    total_source_occurrences = 0
    total_source_reuses = 0
    for user in sorted(sessions_by_user, key=int):
        user_sessions = sorted(sessions_by_user[user], key=lambda row: row["session_ordinal"])
        session_bytes = {
            session["session_id"]: sum(_logical_atom_bytes(atom) for atom in session["atoms"])
            for session in user_sessions
        }
        cumulative_bytes = 0
        cumulative_atoms = 0
        cumulative_sessions = 0
        session_state = {}
        for session in user_sessions:
            session_state[session["session_id"]] = {
                "bytes": session_bytes[session["session_id"]],
                "atoms": len(session["atoms"]),
                "ordinal": session["session_ordinal"],
            }
        query_rows = []
        last_source_query = {}
        user_reuse_distances = []
        for query_index, query in enumerate(queries_by_user[user]):
            visible_ids = query["visible_session_ids"]
            visible_set = set(visible_ids)
            newly_visible = [
                session_id for session_id in visible_ids
                if session_state[session_id]["ordinal"] >= cumulative_sessions
            ]
            # The replay is monotonic in checkpoint time. Count each session once.
            if newly_visible:
                cumulative_sessions = max(
                    session_state[session_id]["ordinal"] + 1 for session_id in newly_visible
                )
                cumulative_atoms += sum(
                    session_state[session_id]["atoms"] for session_id in newly_visible
                )
                cumulative_bytes += sum(
                    session_state[session_id]["bytes"] for session_id in newly_visible
                )
            source = audit_by_question[query["question_id"]]
            source_ids = source["resolved_source_session_ids"]
            total_source_occurrences += len(source_ids)
            reused_source_ids = []
            for source_id in source_ids:
                if source_id in last_source_query:
                    distance = query_index - last_source_query[source_id]
                    user_reuse_distances.append(distance)
                    all_reuse_distances.append(distance)
                    reused_source_ids.append(source_id)
                last_source_query[source_id] = query_index
            total_source_reuses += len(reused_source_ids)
            total_visible_bytes += cumulative_bytes
            total_ingest_bytes += sum(
                session_state[session_id]["bytes"] for session_id in newly_visible
            )
            query_rows.append({
                "question_id": query["question_id"],
                "checkpoint_time": query["checkpoint_time"],
                "visible_sessions": len(visible_set),
                "visible_atoms": cumulative_atoms,
                "visible_logical_bytes": cumulative_bytes,
                "newly_ingested_sessions": len(newly_visible),
                "newly_ingested_logical_bytes": sum(
                    session_state[session_id]["bytes"] for session_id in newly_visible
                ),
                "source_session_count_for_audit": len(source_ids),
                "reused_source_sessions_for_audit": len(reused_source_ids),
                "runtime_labels_read": False,
            })
        user_rows.append({
            "person_id": user,
            "sessions": len(user_sessions),
            "atoms": sum(len(session["atoms"]) for session in user_sessions),
            "ingested_logical_bytes": sum(session_bytes.values()),
            "max_visible_logical_bytes": max((row["visible_logical_bytes"] for row in query_rows), default=0),
            "queries": len(query_rows),
            "source_session_occurrences_for_audit": sum(
                row["source_session_count_for_audit"] for row in query_rows
            ),
            "source_session_reuses_for_audit": sum(
                row["reused_source_sessions_for_audit"] for row in query_rows
            ),
            "source_reuse_distance_median": statistics.median(user_reuse_distances)
            if user_reuse_distances else None,
            "source_reuse_distance_p90": statistics.quantiles(user_reuse_distances, n=10)[8]
            if len(user_reuse_distances) >= 10 else None,
        })

    # Query-level rows are intentionally not retained in the summary object; write a compact trace separately.
    query_trace = []
    for user in sorted(queries_by_user, key=int):
        user_sessions = sorted(sessions_by_user[user], key=lambda row: row["session_ordinal"])
        session_state = {
            session["session_id"]: {
                "bytes": sum(_logical_atom_bytes(atom) for atom in session["atoms"]),
                "atoms": len(session["atoms"]),
                "ordinal": session["session_ordinal"],
            }
            for session in user_sessions
        }
        seen_sessions = set()
        visible_bytes = 0
        visible_atoms = 0
        for query in queries_by_user[user]:
            new_ids = [sid for sid in query["visible_session_ids"] if sid not in seen_sessions]
            seen_sessions.update(new_ids)
            visible_bytes += sum(session_state[sid]["bytes"] for sid in new_ids)
            visible_atoms += sum(session_state[sid]["atoms"] for sid in new_ids)
            query_trace.append({
                "question_id": query["question_id"],
                "person_id": user,
                "checkpoint_time": query["checkpoint_time"],
                "visible_sessions": len(seen_sessions),
                "visible_atoms": visible_atoms,
                "visible_logical_bytes": visible_bytes,
                "newly_ingested_sessions": len(new_ids),
                "newly_ingested_logical_bytes": sum(session_state[sid]["bytes"] for sid in new_ids),
                "runtime_labels_read": False,
            })

    summary = {
        "schema": "mobilemem-omni-persistent-session-replay-audit-v1",
        "input_manifest_sha256": manifest["artifacts_sha256"],
        "users": len(user_rows),
        "sessions": len(sessions),
        "atoms": sum(row["atoms"] for row in user_rows),
        "queries": len(queries),
        "logical_byte_definition": "UTF-8 atom content + serialized observable atom metadata + serialized image references; not DRAM transactions",
        "total_ingested_logical_bytes": total_ingest_bytes,
        "sum_query_visible_logical_bytes": total_visible_bytes,
        "source_session_occurrences_for_audit": total_source_occurrences,
        "source_session_reuses_for_audit": total_source_reuses,
        "source_session_reuse_fraction_for_audit": (
            total_source_reuses / total_source_occurrences if total_source_occurrences else None
        ),
        "source_reuse_distance_median": statistics.median(all_reuse_distances)
        if all_reuse_distances else None,
        "source_reuse_distance_p90": statistics.quantiles(all_reuse_distances, n=10)[8]
        if len(all_reuse_distances) >= 10 else None,
        "pim_qualification": "not_started; logical trace only",
        "runtime_labels_read": False,
        "user_rows": user_rows,
        "status": "pass",
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "query_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in query_trace)
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_replay(args.replay, args.output)
    print(json.dumps({
        "status": summary["status"],
        "users": summary["users"],
        "sessions": summary["sessions"],
        "atoms": summary["atoms"],
        "queries": summary["queries"],
        "total_ingested_logical_bytes": summary["total_ingested_logical_bytes"],
        "source_session_reuse_fraction_for_audit": summary["source_session_reuse_fraction_for_audit"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
