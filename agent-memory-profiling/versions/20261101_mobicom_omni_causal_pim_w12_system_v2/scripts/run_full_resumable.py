#!/usr/bin/env python3
"""Run the frozen main experiment with atomic per-row checkpoints and seed reuse.

The experiment is user-sharded.  Every worker owns one GPU and takes the next
unfinished user from a queue, so a slow user does not leave the other GPUs idle.
Existing complete rows are imported before launch; the unchanged
run_proof_mainline.py then resumes from the new atomic partial checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import yaml

from run_main_experiment import serve_one


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("main experiment configuration must be a mapping")
    return value


def resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else (ROOT / result)


def candidate_sources(source_root: Path, user_id: str) -> list[Path]:
    stem = source_root / f"user_{user_id}_shard_0"
    return [path for path in (stem.with_suffix(".json"), stem.with_suffix(".partial.json"))
            if path.exists()]


def collect_seed_rows(source_roots: list[Path], user_id: str,
                      question_ids: list[str]) -> tuple[dict[str, dict], list[str]]:
    allowed = set(question_ids)
    by_question: dict[str, dict] = {}
    origins: list[str] = []
    for root in source_roots:
        for path in candidate_sources(root, user_id):
            try:
                document = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if document.get("person_id") != user_id:
                raise ValueError(f"seed person mismatch: {path}")
            accepted = 0
            for row in document.get("rows") or []:
                identity = row.get("question_id")
                if identity not in allowed:
                    continue
                if row.get("person_id") != user_id:
                    raise ValueError(f"seed row person mismatch: {path} {identity}")
                previous = by_question.get(identity)
                if previous is not None:
                    if previous.get("answer_sha256") != row.get("answer_sha256"):
                        raise ValueError(
                            f"conflicting seed answers for {user_id}/{identity}: {path}")
                    continue
                by_question[identity] = row
                accepted += 1
            if accepted:
                origins.append(f"{path}:{accepted}")
    ordered = {identity: by_question[identity]
               for identity in question_ids if identity in by_question}
    return ordered, origins


def checkpoint_document(user_id: str, question_ids: list[str], rows: list[dict],
                        cohort_sha256: str, origins: list[str], *,
                        pim: dict | None = None) -> dict:
    pim = pim or {}
    w12 = pim.get('w12', {}) if isinstance(pim, dict) else {}
    functional_route = str(w12.get('functional_route', 'none'))
    reader_quantization = 'w12a16' if functional_route != 'none' else 'bf16'
    return {
        "complete": False,
        "person_id": user_id,
        "question_ids": question_ids,
        "rows": rows,
        "shard_index": 0,
        "shard_count": 1,
        "pre_reader_operator_profile": False,
        "reader_quantization": reader_quantization,
        "w12_functional_route": functional_route,
        "w12_numeric_mode": str(w12.get('numeric_mode', 'bf16_reference')),
        "w12_bank_count": int(w12.get('bank_count', 256)),
        "w12_tile_rows": int(w12.get('tile_rows', 16)),
        "w12_host_contract_rows": int(w12.get(
            'host_contract_rows', int(w12.get('bank_count', 256))
            * int(w12.get('tile_rows', 16)))),
        "w12a16_group_size": int(w12.get('group_size', 128)),
        "w12_release_source": bool(w12.get('release_source', False)),
        "cohort_sha256": cohort_sha256,
        "evaluation_label_files_read": [],
        "resume_export": {
            "schema": "mobilemem-main-resume-export-v1",
            "created_at": now(),
            "sources": origins,
            "imported_rows": len(rows),
        },
    }


def seed_checkpoints(config: dict) -> dict:
    paths = config["paths"]
    cohort_path = resolve(paths["cohort"])
    cohort = json.loads(cohort_path.read_text())
    cohort_sha = digest(cohort_path)
    run_root = resolve(paths["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    source_roots = [resolve(item) for item in config.get("resume", {}).get("seed_roots", [])]
    pim = config.get("pim", {})
    summary = {"schema": "mobilemem-main-seed-summary-v1", "created_at": now(),
               "cohort": str(cohort_path), "cohort_sha256": cohort_sha,
               "source_roots": [str(item) for item in source_roots], "users": {}}
    for user_id, raw_question_ids in sorted(
            cohort["question_ids_by_user"].items(), key=lambda item: int(item[0])):
        question_ids = list(raw_question_ids)
        partial_path = run_root / f"user_{user_id}_shard_0.partial.json"
        final_path = run_root / f"user_{user_id}_shard_0.json"
        seed_rows, origins = collect_seed_rows(source_roots, user_id, question_ids)
        if final_path.exists():
            existing = json.loads(final_path.read_text())
            if not existing.get("complete"):
                raise ValueError(f"non-complete final output exists: {final_path}")
            summary["users"][user_id] = {
                "question_ids": len(question_ids),
                "already_complete": True,
                "rows": len(existing.get("rows") or []),
            }
            continue
        if partial_path.exists():
            checkpoint = json.loads(partial_path.read_text())
            if (checkpoint.get("person_id") != user_id
                    or checkpoint.get("question_ids") != question_ids
                    or checkpoint.get("cohort_sha256") != cohort_sha):
                raise ValueError(f"existing checkpoint contract differs: {partial_path}")
            merged = {row["question_id"]: row for row in checkpoint.get("rows") or []}
            for identity, row in seed_rows.items():
                previous = merged.get(identity)
                if previous is not None and previous.get("answer_sha256") != row.get("answer_sha256"):
                    raise ValueError(f"checkpoint conflicts with seed for {user_id}/{identity}")
                merged.setdefault(identity, row)
            ordered_rows = [merged[identity] for identity in question_ids if identity in merged]
            document = checkpoint_document(user_id, question_ids, ordered_rows,
                                            cohort_sha, origins, pim=pim)
            document["resume_export"]["previous_rows"] = len(checkpoint.get("rows") or [])
        else:
            ordered_rows = list(seed_rows.values())
            document = checkpoint_document(user_id, question_ids, ordered_rows,
                                            cohort_sha, origins, pim=pim)
        atomic_write(partial_path, document)
        summary["users"][user_id] = {
            "question_ids": len(question_ids),
            "already_complete": False,
            "rows": len(ordered_rows),
            "remaining": len(question_ids) - len(ordered_rows),
            "sources": origins,
        }
    summary["imported_rows"] = sum(item.get("rows", 0) for item in summary["users"].values())
    summary["remaining_rows"] = sum(item.get("remaining", 0) for item in summary["users"].values())
    atomic_write(run_root / "resume_seed_summary.json", summary)
    return summary


def run(config_path: Path, *, seed_only: bool = False) -> int:
    config = load_config(config_path)
    run_root = resolve(config["paths"]["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".orchestrator.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another full-experiment orchestrator owns {lock_path}")

    if not seed_only:
        atomic_write(run_root / "orchestrator.pid", os.getpid())

    summary = seed_checkpoints(config)
    print(json.dumps({"seed": summary}, ensure_ascii=False), flush=True)
    if seed_only:
        return 0

    reader_gpus = [str(item) for item in
                   config["execution"].get("reader_gpus", config["execution"]["gpus"])]
    worker_count = int(config["execution"].get("worker_count", len(reader_gpus)))
    if worker_count <= 0 or worker_count != len(reader_gpus):
        raise ValueError("worker_count must equal the number of reader GPUs")
    cohort = json.loads(resolve(config["paths"]["cohort"]).read_text())
    users = sorted(cohort["question_ids_by_user"], key=int)
    pending: queue.Queue[str] = queue.Queue()
    for user_id in users:
        pending.put(user_id)
    retries = max(0, int(config["execution"].get("retries", 2)))
    status_path = run_root / "orchestrator_status.json"
    status_lock = threading.Lock()
    started_at = now()
    status: dict[str, Any] = {
        "schema": "mobilemem-main-orchestrator-v1",
        "started_at": started_at,
        "updated_at": started_at,
        "complete": False,
        "users_total": len(users),
        "workers": {gpu: {"gpu": gpu, "state": "starting"} for gpu in reader_gpus},
        "completed_users": [],
        "failed_users": [],
    }

    def write_status() -> None:
        with status_lock:
            status["updated_at"] = now()
            atomic_write(status_path, status)

    write_status()

    def worker(gpu: str) -> None:
        while True:
            try:
                user_id = pending.get_nowait()
            except queue.Empty:
                return
            with status_lock:
                status["workers"][gpu] = {"gpu": gpu, "state": "running",
                                          "user_id": user_id, "started_at": now()}
            write_status()
            code = 1
            attempts = 0
            for attempt in range(retries + 1):
                attempts = attempt + 1
                try:
                    code = int(serve_one(config, user_id, gpu, shard_index=0, shard_count=1))
                except Exception as error:  # keep the other workers alive
                    print(json.dumps({"time": now(), "gpu": gpu, "user_id": user_id,
                                      "attempt": attempts, "error": repr(error)}), flush=True)
                    code = 1
                if code == 0:
                    break
                time.sleep(min(30 * attempts, 120))
            with status_lock:
                record = {"user_id": user_id, "gpu": gpu, "returncode": code,
                          "attempts": attempts, "finished_at": now()}
                if code == 0:
                    status["completed_users"].append(user_id)
                else:
                    status["failed_users"].append(user_id)
                status["completed_users"] = sorted(set(status["completed_users"]), key=int)
                status["failed_users"] = sorted(set(status["failed_users"]), key=int)
                status["workers"][gpu] = {"gpu": gpu, "state": "idle", "last": record}
            write_status()
            print(json.dumps({"time": now(), **record}, ensure_ascii=False), flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False)
               for gpu in reader_gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with status_lock:
        status["complete"] = not status["failed_users"]
        status["finished_at"] = now()
    write_status()
    print(json.dumps({"complete": status["complete"],
                      "completed_users": status["completed_users"],
                      "failed_users": status["failed_users"]}, ensure_ascii=False), flush=True)
    return 0 if status["complete"] else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.config.resolve(), seed_only=args.seed_only))


if __name__ == "__main__":
    main()
