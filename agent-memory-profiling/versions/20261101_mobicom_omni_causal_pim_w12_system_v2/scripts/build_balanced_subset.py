#!/usr/bin/env python3
"""Freeze a deterministic user-and-type-balanced development/validation subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_SPLITS = {"development", "validation"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_key(salt: str, user_id: str, question_type: str, question_id: str) -> str:
    material = "\0".join((salt, user_id, question_type, question_id))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def select(manifests: list[dict], labels: dict[str, dict], source_cohort: dict,
           users: list[str], splits: set[str], per_user_per_type: int,
           salt: str) -> dict[str, Any]:
    manifest_by_id = {row["question_id"]: row for row in manifests}
    if len(manifest_by_id) != len(manifests):
        raise ValueError("duplicate question IDs in manifests")
    registered = source_cohort["question_ids_by_user"]
    unknown = sorted(set(users) - set(registered))
    if unknown:
        raise ValueError(f"users absent from source cohort: {unknown}")

    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for user_id in users:
        for question_id in registered[user_id]:
            manifest = manifest_by_id.get(question_id)
            label = labels.get(question_id)
            if manifest is None or label is None:
                raise ValueError(f"missing manifest or label for {question_id}")
            if str(manifest["person_id"]) != user_id:
                raise ValueError(f"manifest person mismatch for {question_id}")
            if manifest["split"] not in splits:
                continue
            pools[user_id][label["question_type"]].append({
                "question_id": question_id,
                "effective_timestamp": manifest["effective_timestamp"],
                "split": manifest["split"],
                "difficulty": label.get("difficulty"),
            })

    selected_by_user: dict[str, list[dict]] = {}
    for user_id in users:
        selected: list[dict] = []
        types = sorted(pools[user_id])
        if not types:
            raise ValueError(f"user {user_id} has no questions in {sorted(splits)}")
        for question_type in types:
            candidates = pools[user_id][question_type]
            if len(candidates) < per_user_per_type:
                raise ValueError(
                    f"user {user_id} type {question_type} has {len(candidates)} questions; "
                    f"need {per_user_per_type}")
            candidates.sort(key=lambda row: selection_key(
                salt, user_id, question_type, row["question_id"]))
            chosen = candidates[:per_user_per_type]
            for row in chosen:
                selected.append({**row, "question_type": question_type})
        selected.sort(key=lambda row: (row["effective_timestamp"], row["question_id"]))
        selected_by_user[user_id] = selected

    return {"pools": pools, "selected_by_user": selected_by_user}


def build(prepared: Path, source_cohort_path: Path, output_dir: Path,
          users: list[str], splits: set[str], per_user_per_type: int,
          salt: str, name: str, force: bool) -> dict[str, Any]:
    if not splits <= ALLOWED_SPLITS:
        raise ValueError("balanced debugging subsets may only use development and validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = output_dir / "cohort.json"
    audit_path = output_dir / "subset_audit.json"
    if not force and (cohort_path.exists() or audit_path.exists()):
        raise ValueError(f"refusing to overwrite frozen subset in {output_dir}")

    source_cohort = json.loads(source_cohort_path.read_text())
    manifests = load_jsonl(prepared / "manifests.jsonl")
    labels = {row["question_id"]: row for row in load_jsonl(prepared / "labels.jsonl")}
    result = select(manifests, labels, source_cohort, users, splits,
                    per_user_per_type, salt)
    selected_by_user = result["selected_by_user"]
    question_ids_by_user = {
        user_id: [row["question_id"] for row in selected_by_user[user_id]]
        for user_id in sorted(users, key=int)
    }
    selected_rows = [row for user_id in question_ids_by_user
                     for row in selected_by_user[user_id]]
    selected_ids = sorted(row["question_id"] for row in selected_rows)
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("subset contains duplicate question IDs")

    cohort = {
        "schema": "mobilemem-omni-main-cohort-v1",
        "prepared_audit_sha256": digest(prepared / "audit.json"),
        "manifests_sha256": digest(prepared / "manifests.jsonl"),
        "serving_splits": sorted(splits),
        "question_ids_by_user": question_ids_by_user,
        "question_count_by_user": {key: len(value)
                                   for key, value in question_ids_by_user.items()},
        "questions": len(selected_ids),
        "evaluation_label_files_read": [],
        "boundary": ("question IDs and timestamps only; no answer, type, evidence, "
                     "or label fields are copied"),
        "subset": {
            "name": name,
            "selection_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
            "selection_salt": salt,
            "per_user_per_type": per_user_per_type,
        },
    }

    selected_type = Counter(row["question_type"] for row in selected_rows)
    full_population_type: Counter = Counter()
    manifest_by_id = {row["question_id"]: row for row in manifests}
    for user_id in users:
        for question_id in source_cohort["question_ids_by_user"][user_id]:
            manifest = manifest_by_id.get(question_id)
            if manifest is not None and manifest["split"] in splits:
                full_population_type[labels[question_id]["question_type"]] += 1
    selected_total = len(selected_rows)
    population_total = sum(full_population_type.values())
    type_audit = {}
    for question_type in sorted(selected_type):
        selected_count = selected_type[question_type]
        population_count = full_population_type[question_type]
        type_audit[question_type] = {
            "selected": selected_count,
            "population": population_count,
            "selected_share": selected_count / selected_total,
            "population_share": population_count / population_total,
            "natural_accuracy_weight": population_count / selected_count,
        }
    audit = {
        "schema": "mobilemem-omni-balanced-subset-audit-v1",
        "name": name,
        "created_from": {
            "prepared": str(prepared),
            "source_cohort": str(source_cohort_path),
            "source_cohort_sha256": digest(source_cohort_path),
            "manifests_sha256": cohort["manifests_sha256"],
        },
        "selection": {
            "method": "lowest SHA-256 rank within each user x question_type cell",
            "salt": salt,
            "users": list(question_ids_by_user),
            "splits": sorted(splits),
            "per_user_per_type": per_user_per_type,
            "selection_sha256": cohort["subset"]["selection_sha256"],
        },
        "counts": {
            "questions": selected_total,
            "users": len(question_ids_by_user),
            "types": len(selected_type),
            "by_user": {key: len(value) for key, value in question_ids_by_user.items()},
            "by_type": dict(sorted(selected_type.items())),
            "by_split": dict(sorted(Counter(row["split"] for row in selected_rows).items())),
            "by_difficulty": dict(sorted(Counter(
                row["difficulty"] for row in selected_rows).items())),
        },
        "population": {
            "questions": population_total,
            "by_type": dict(sorted(full_population_type.items())),
        },
        "type_audit": type_audit,
        "boundary": "development/validation only; holdout is never read or sampled",
    }
    atomic_write(cohort_path, cohort)
    atomic_write(audit_path, audit)
    return {"cohort": str(cohort_path), "audit": str(audit_path),
            "questions": selected_total, "by_type": dict(sorted(selected_type.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--source-cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--users", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["development", "validation"],
                        choices=sorted(ALLOWED_SPLITS))
    parser.add_argument("--per-user-per-type", type=int, default=15)
    parser.add_argument("--salt", default="core840_v1")
    parser.add_argument("--name", default="core840_v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(
        args.prepared.resolve(), args.source_cohort.resolve(), args.output_dir.resolve(),
        args.users, set(args.splits), args.per_user_per_type, args.salt, args.name,
        args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
