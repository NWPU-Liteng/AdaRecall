"""Prepare MobileMem-Omni without leaking QA labels into serving inputs.

The Omni release does not provide a query-time cutoff for every question.
This adapter therefore supports an explicit ``history_end`` protocol only:
the complete observable history for a user is visible to every question.
It must not be described as an online temporal replay until a release-time
contract is supplied separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import zipfile


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def normalize_time(value: str) -> tuple[str, str | None]:
    value = str(value)
    if re.fullmatch(r"\d{4}-\d{2}:\d{2} \d{2}:\d{2}:\d{2}", value):
        repaired = value[:7] + "-" + value[8:]
        return repaired, "month_day_separator"
    return value, None


def parse_time(value: str) -> datetime:
    normalized, _ = normalize_time(value)
    return datetime.strptime(normalized, TIME_FORMAT)


def normalize_image_ref(value: str) -> str:
    value = str(value).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.removeprefix("image/")
    if value.startswith('/') or '..' in value.split('/'):
        raise ValueError('unsafe image reference')
    return value


class ImageResolver:
    def __init__(self, archive: Path | None):
        self.archive = archive
        self.members = set()
        self.by_user_basename = defaultdict(list)
        if archive is None:
            return
        with zipfile.ZipFile(archive, metadata_encoding='gb18030') as handle:
            for name in handle.namelist():
                normalized = normalize_image_ref(name)
                if Path(normalized).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                self.members.add(normalized)
                parts = normalized.split("/", 1)
                if len(parts) == 2:
                    self.by_user_basename[(parts[0], posixpath.basename(normalized))].append(normalized)

    def resolve(self, value: str | None) -> dict | None:
        if not value:
            return None
        raw = str(value)
        normalized = normalize_image_ref(raw)
        if normalized in self.members:
            return {"raw": raw, "normalized": normalized, "status": "exact"}
        parts = normalized.split("/", 1)
        candidates = []
        if len(parts) == 2:
            candidates = self.by_user_basename.get((parts[0], posixpath.basename(normalized)), [])
        if len(candidates) == 1:
            return {"raw": raw, "normalized": normalized, "resolved": candidates[0],
                    "status": "unique_user_basename"}
        return {"raw": raw, "normalized": normalized,
                "status": "missing" if not candidates else "ambiguous",
                "candidates": candidates}


def iter_image_refs(value):
    if isinstance(value, str):
        if Path(value).suffix.lower() in IMAGE_SUFFIXES:
            yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_image_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_image_refs(child)


def canonical_event_key(session: dict) -> str:
    return str(session["session_id"])


def build_graph(record: dict, resolver: ImageResolver) -> tuple[dict, dict]:
    person_id = str(record["uuid"])
    sessions = sorted(record.get("sessions", []),
                      key=lambda item: (parse_time(item["event_start_time"]), str(item["session_id"])))
    event_index = defaultdict(list)
    atoms = []
    session_nodes = []
    image_audit = Counter()
    timestamp_repairs = []
    for session_ordinal, session in enumerate(sessions):
        session_id = canonical_event_key(session)
        event_id = str(session.get("event_id", session_id))
        if session_id in {node['session_id'] for node in session_nodes}:
            raise ValueError('duplicate session identity')
        event_index[event_id].append(session_id)
        raw_start = session["event_start_time"]
        raw_end = session["event_end_time"]
        start, start_repair = normalize_time(raw_start)
        end, end_repair = normalize_time(raw_end)
        if start_repair:
            timestamp_repairs.append({"session_id": session_id, "field": "event_start_time",
                                      "raw": raw_start, "normalized": start, "rule": start_repair})
        if end_repair:
            timestamp_repairs.append({"session_id": session_id, "field": "event_end_time",
                                      "raw": raw_end, "normalized": end, "rule": end_repair})
        if parse_time(end) < parse_time(start):
            raise ValueError('session end precedes start')
        session_images = []
        for image_ref in iter_image_refs(session.get("image_refs", {})):
            resolved = resolver.resolve(image_ref)
            session_images.append(resolved)
            image_audit[resolved["status"]] += 1
        session_nodes.append({
            "session_id": session_id, "event_id": event_id,
            "start_time": start, "end_time": end,
            "ordinal": session_ordinal,
            "image_refs": session_images,
        })
        for turn_ordinal, turn in enumerate(session.get("dialogue", [])):
            atom_id = f"{session_id}:turn:{turn_ordinal}"
            atom = {
                "atom_id": atom_id, "event_id": event_id, "session_id": session_id,
                "atom_kind": "dialogue_turn", "ordinal": len(atoms),
                "session_ordinal": session_ordinal, "turn_ordinal": turn_ordinal,
                "visibility_start": start, "visibility_end": end,
                "role": turn.get("role"), "content_type": turn.get("content_type", "text"),
                "content": turn.get("content", ""),
                "image_refs": [],
            }
            if turn.get("image_inline"):
                resolved = resolver.resolve(turn["image_inline"])
                atom["image_refs"].append(resolved)
                image_audit[resolved["status"]] += 1
            atoms.append(atom)
    parent_index = defaultdict(list)
    for session in sessions:
        session_id = canonical_event_key(session)
        parent = session.get("parent_event_id")
        if parent is None:
            continue
        parent_index[str(parent)].append(session_id)
    graph = {
        "schema_version": "mobilemem-omni-observable-graph-v1",
        "person_id": person_id,
        "visibility_protocol": "history_end",
        "visibility_boundary": "all sessions visible; Omni questions have no query release timestamp",
        "label_free": True,
        "sessions": session_nodes,
        "atoms": atoms,
        "dependencies": {},
        "dependency_boundary": "synthetic parent events are excluded; dialogue context is grouped by observed turns only",
        "hyperedges": [
            {"kind": "session", "identity": node["session_id"],
             "members": [atom["atom_id"] for atom in atoms if atom["session_id"] == node["session_id"]]}
            for node in session_nodes
        ],
        "image_audit": dict(image_audit),
        "timestamp_repairs": timestamp_repairs,
    }
    return graph, {"session_ids": {node["session_id"] for node in session_nodes},
                   "event_ids": {node["event_id"] for node in session_nodes},
                   "event_sessions": dict(event_index), "parent_event_sessions": dict(parent_index)}


def source_session_candidates(raw: str, session_ids: set[str]) -> list[str]:
    if raw in session_ids:
        return [raw]
    return []


def source_event_candidates(raw, identity: dict) -> list[str]:
    value = str(raw)
    exact = sorted(identity['event_sessions'].get(value, []))
    if exact:
        return exact
    return sorted(identity['parent_event_sessions'].get(value, []))


def prepare(data_path: Path, questions_path: Path, output: Path, image_archive: Path | None,
            split_reference: Path | None = None) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    data_records = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    question_rows = [json.loads(line) for line in questions_path.read_text().splitlines() if line.strip()]
    question_records = {str(row['uuid']): row for row in question_rows}
    if len(question_rows) != len(question_records):
        raise ValueError('duplicate QA user')
    people = sorted({str(row['uuid']) for row in data_records}, key=int)
    if len(people) != len(data_records) or not set(question_records).issubset(people):
        raise ValueError('duplicate history user or unknown QA user')
    split_people = sorted(question_records, key=int)
    dev_count = max(1, len(split_people) // 5)
    validation_count = max(1, len(split_people) // 5)
    development_people = split_people[:dev_count]
    validation_people = split_people[dev_count:dev_count + validation_count]
    if split_reference is not None:
        reference = json.loads(split_reference.read_text())
        development_people = reference['development_people']
        validation_people = reference['validation_people']
        groups = [reference[name] for name in ('development_people', 'validation_people', 'holdout_people')]
        flattened = [person for group in groups for person in group]
        if len(flattened) != len(set(flattened)) or not set(flattened).issubset(split_people):
            raise ValueError('reference split must contain disjoint known QA users')
    resolver = ImageResolver(image_archive)
    graphs = []
    manifests = []
    labels = []
    reference_audit = []
    question_ids = set()
    mapping_audit = Counter()
    for record in data_records:
        person_id = str(record["uuid"])
        graph, identity = build_graph(record, resolver)
        graphs.append(graph)
        question_record = question_records.get(person_id, {"questions": []})
        history_end = max((session["end_time"] for session in graph["sessions"]), default=None)
        for question_index, question in enumerate(question_record.get("questions", [])):
            qid = str(question["question_id"])
            if qid in question_ids:
                raise ValueError('duplicate question identity')
            question_ids.add(qid)
            if history_end is None:
                raise ValueError('QA user has no visible history')
            resolved_sessions = []
            for raw in question.get("source_session_ids", []):
                candidates = source_session_candidates(str(raw), identity["session_ids"])
                mapping_audit["source_session_exact" if str(raw) in identity["session_ids"] else
                             "source_session_derived"] += bool(candidates)
                mapping_audit["source_session_unresolved"] += not bool(candidates)
                resolved_sessions.extend(candidates)
            resolved_events = []
            for raw in question.get("source_event_ids", []):
                candidates = source_event_candidates(raw, identity)
                mapping_audit["source_event_resolved"] += bool(candidates)
                mapping_audit["source_event_unresolved"] += not bool(candidates)
                resolved_events.extend(candidates)
            image_refs = [resolver.resolve(value) for value in question.get("image_refs", []) or []]
            mapping_audit["question_image_missing"] += any(item["status"] == "missing" for item in image_refs)
            manifests.append({
                "question_id": qid, "person_id": person_id, "question": question["question"],
                "question_index": question_index, "visibility_protocol": "history_end",
                "effective_timestamp": history_end,
                "split": 'development' if person_id in development_people else
                         'validation' if person_id in validation_people else 'holdout',
            })
            labels.append({
                "question_id": qid, "person_id": person_id,
                "golden_answers": [str(question.get("answer", ""))],
                "question_type": question.get("question_type"), "difficulty": question.get("difficulty"),
                "source_session_ids": sorted(set(resolved_sessions)),
                "source_event_session_ids": sorted(set(resolved_events)),
                "evidence_image_refs": image_refs,
                "raw_source_session_ids": question.get("source_session_ids", []),
                "raw_source_event_ids": question.get("source_event_ids", []),
                "raw_evidence": question.get('evidence', []),
            })
            reference_audit.append({
                'question_id': qid, 'person_id': person_id,
                'unresolved_session_ids': [value for value in question.get('source_session_ids', [])
                                           if str(value) not in identity['session_ids']],
                'unresolved_event_ids': [value for value in question.get('source_event_ids', [])
                                         if not source_event_candidates(value, identity)],
                'unresolved_images': [item for item in image_refs if item['status'] in {'missing', 'ambiguous'}],
                'mapping_boundary': 'exact session/event or explicit parent-group metadata; no answer-based or prefix guessing',
            })
    split = {
        "protocol": "user_disjoint_static_history_end_v1",
        "development_people": development_people,
        "validation_people": validation_people,
        "holdout_people": [person for person in split_people
                           if person not in development_people and person not in validation_people],
        "assignment": ("preserved prior development/validation; all newly introduced users sealed as holdout"
                       if split_reference is not None else "numeric user IDs sorted before quality evaluation"),
        "split_reference_sha256": digest(split_reference) if split_reference is not None else None,
        "warning": "history_end is not an online temporal cutoff; no query release timestamps are present",
    }
    output.mkdir(parents=True)
    files = {
        "omni_graph.json": json.dumps(graphs, ensure_ascii=False),
        "manifests.jsonl": "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifests),
        "labels.jsonl": "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in labels),
        "split.json": json.dumps(split, ensure_ascii=False, indent=2),
        "reference_audit.jsonl": ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in reference_audit),
    }
    for name, content in files.items():
        (output / name).write_text(content)
    audit = {
        "schema_version": "mobilemem-omni-prepared-v1",
        "source_sha256": {"data": digest(data_path), "questions": digest(questions_path)},
        "image_archive_sha256": digest(image_archive) if image_archive else None,
        "users": len(people), "sessions": sum(len(graph["sessions"]) for graph in graphs),
        "atoms": sum(len(graph["atoms"]) for graph in graphs), "questions": len(manifests),
        "timestamp_repairs": [repair for graph in graphs for repair in graph["timestamp_repairs"]],
        "mapping": dict(mapping_audit), "visibility_protocol": "history_end",
        "serving_exclusions": ["golden_answers", "question_type", "difficulty", "source evidence labels"],
        "artifacts_sha256": {name: digest(output / name) for name in files},
        "status": "qualified_with_reference_gaps" if any(
            item['unresolved_session_ids'] or item['unresolved_event_ids'] or item['unresolved_images']
            for item in reference_audit) else "pass",
        "quality_evaluated": False,
        "serving_files": ['omni_graph.json', 'manifests.jsonl', 'split.json'],
        "evaluation_only_files": ['labels.jsonl', 'reference_audit.jsonl'],
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-reference", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.questions, args.output, args.image_archive, args.split_reference),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
