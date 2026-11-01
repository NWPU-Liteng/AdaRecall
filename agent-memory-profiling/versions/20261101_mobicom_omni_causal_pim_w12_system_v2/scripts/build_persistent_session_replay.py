"""Build a label-separated persistent MobileMem-Omni session replay.

The Omni questions do not contain natural query-release timestamps. This tool
therefore creates an explicit dataset-derived checkpoint replay: a question is
scheduled after the end of its resolved source sessions, or at the user's
history end when no complete source-session boundary is available. The
serving trace never contains answers, question types, or evidence labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from omni_prepare import ImageResolver, build_graph, digest, parse_time


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return digest(path)


def _session_end_index(graph: dict) -> dict[str, dict]:
    return {str(session["session_id"]): session for session in graph["sessions"]}


def _visible_sessions(sessions: list[dict], checkpoint: str) -> list[str]:
    checkpoint_time = parse_time(checkpoint)
    return [
        str(session["session_id"])
        for session in sessions
        if parse_time(session["end_time"]) <= checkpoint_time
    ]


def build_replay(
    data_path: Path,
    questions_path: Path,
    output: Path,
    image_archive: Path | None = None,
    users: set[str] | None = None,
) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite replay output: {output}")
    records = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    question_records = {
        str(row["uuid"]): row.get("questions", [])
        for row in (json.loads(line) for line in questions_path.read_text().splitlines() if line.strip())
    }
    selected_records = [
        record for record in records
        if users is None or str(record["uuid"]) in users
    ]
    if not selected_records:
        raise ValueError("no users selected for replay")

    resolver = ImageResolver(image_archive)
    session_rows: list[dict] = []
    query_rows: list[dict] = []
    schedule_audit_rows: list[dict] = []
    user_summaries: list[dict] = []
    seen_question_ids: set[str] = set()
    total_atoms = 0

    for record in sorted(selected_records, key=lambda item: int(item["uuid"])):
        person_id = str(record["uuid"])
        graph, identity = build_graph(record, resolver)
        sessions = list(graph["sessions"])
        atoms_by_session: dict[str, list[dict]] = defaultdict(list)
        for atom in graph["atoms"]:
            atoms_by_session[str(atom["session_id"])].append(atom)
        session_index = _session_end_index(graph)
        history_end = max((session["end_time"] for session in sessions), default=None)
        if history_end is None:
            raise ValueError(f"user {person_id} has no sessions")

        for session in sessions:
            session_id = str(session["session_id"])
            session_rows.append({
                "person_id": person_id,
                "session_id": session_id,
                "event_id": session["event_id"],
                "session_ordinal": session["ordinal"],
                "start_time": session["start_time"],
                "end_time": session["end_time"],
                "image_refs": session["image_refs"],
                "atoms": atoms_by_session[session_id],
            })
            total_atoms += len(atoms_by_session[session_id])

        user_queries = []
        for source_order, question in enumerate(question_records.get(person_id, [])):
            question_id = str(question["question_id"])
            if question_id in seen_question_ids:
                raise ValueError(f"duplicate question identity: {question_id}")
            seen_question_ids.add(question_id)
            raw_source_sessions = [str(value) for value in question.get("source_session_ids", [])]
            resolved_source_sessions = sorted({
                source_id for source_id in raw_source_sessions
                if source_id in identity["session_ids"]
            }, key=lambda value: session_index[value]["ordinal"])
            unresolved_source_sessions = sorted(set(raw_source_sessions) - set(resolved_source_sessions))
            if resolved_source_sessions and not unresolved_source_sessions:
                checkpoint = max(
                    (session_index[value]["end_time"] for value in resolved_source_sessions),
                    key=parse_time,
                )
                checkpoint_reason = "max_resolved_source_session_end"
                temporal_eligible = True
            else:
                checkpoint = history_end
                checkpoint_reason = (
                    "history_end_no_source_session"
                    if not raw_source_sessions else
                    "history_end_unresolved_source_session"
                )
                temporal_eligible = False
            visible_sessions = _visible_sessions(sessions, checkpoint)
            visible_set = set(visible_sessions)
            missing_visible_sources = sorted(set(resolved_source_sessions) - visible_set)
            if missing_visible_sources:
                raise ValueError(
                    f"checkpoint does not expose source sessions for {question_id}: "
                    f"{missing_visible_sources}"
                )
            row = {
                "question_id": question_id,
                "person_id": person_id,
                "source_order": source_order,
                "query_text": str(question["question"]),
                "checkpoint_time": checkpoint,
                "checkpoint_reason": checkpoint_reason,
                "visible_session_count": len(visible_sessions),
                "visible_session_ids": visible_sessions,
                "runtime_labels_read": False,
                "serving_boundary": "persistent_session_replay; no answers/types/evidence labels",
            }
            query_rows.append(row)
            user_queries.append(row)
            schedule_audit_rows.append({
                "question_id": question_id,
                "person_id": person_id,
                "raw_source_session_ids": raw_source_sessions,
                "resolved_source_session_ids": resolved_source_sessions,
                "unresolved_source_session_ids": unresolved_source_sessions,
                "checkpoint_time": checkpoint,
                "checkpoint_reason": checkpoint_reason,
                "temporal_replay_eligible": temporal_eligible,
                "history_end": history_end,
                "audit_boundary": "offline schedule construction only; never read at serving time",
            })
        user_summaries.append({
            "person_id": person_id,
            "sessions": len(sessions),
            "atoms": sum(len(values) for values in atoms_by_session.values()),
            "queries": len(user_queries),
            "history_start": min(session["start_time"] for session in sessions),
            "history_end": history_end,
            "temporal_eligible_queries": sum(
                row["checkpoint_reason"] == "max_resolved_source_session_end"
                for row in user_queries
            ),
        })

    session_rows.sort(key=lambda row: (int(row["person_id"]), row["session_ordinal"]))
    query_rows.sort(key=lambda row: (int(row["person_id"]), parse_time(row["checkpoint_time"]), row["question_id"]))
    schedule_audit_rows.sort(key=lambda row: (int(row["person_id"]), row["question_id"]))
    output.mkdir(parents=True)
    session_digest = _write_jsonl(output / "sessions.jsonl", session_rows)
    query_digest = _write_jsonl(output / "queries.jsonl", query_rows)
    audit_digest = _write_jsonl(output / "schedule_audit.jsonl", schedule_audit_rows)
    manifest = {
        "schema": "mobilemem-omni-persistent-session-replay-v1",
        "source_sha256": {
            "data": digest(data_path),
            "questions": digest(questions_path),
        },
        "image_archive_sha256": digest(image_archive) if image_archive else None,
        "users": len(user_summaries),
        "sessions": len(session_rows),
        "atoms": total_atoms,
        "queries": len(query_rows),
        "query_checkpoint_policy": (
            "max resolved source-session end; history_end fallback for missing or unresolved sources"
        ),
        "query_arrival_contract": "not provided by MobileMem-Omni; checkpoint is a derived replay schedule",
        "runtime_labels_read": False,
        "serving_files": ["sessions.jsonl", "queries.jsonl"],
        "evaluation_only_files": ["schedule_audit.jsonl"],
        "artifacts_sha256": {
            "sessions.jsonl": session_digest,
            "queries.jsonl": query_digest,
            "schedule_audit.jsonl": audit_digest,
        },
        "users_summary": user_summaries,
        "status": "pass",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", nargs="*", help="Optional numeric user IDs")
    args = parser.parse_args()
    result = build_replay(
        args.data,
        args.questions,
        args.output,
        args.image_archive,
        set(args.users) if args.users else None,
    )
    print(json.dumps({
        "status": result["status"],
        "users": result["users"],
        "sessions": result["sessions"],
        "atoms": result["atoms"],
        "queries": result["queries"],
        "temporal_eligible_queries": sum(
            item["temporal_eligible_queries"] for item in result["users_summary"]
        ),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
