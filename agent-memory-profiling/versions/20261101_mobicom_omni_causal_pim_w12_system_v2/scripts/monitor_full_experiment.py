#!/usr/bin/env python3
"""Terminal and JSON monitor for the resumable full main experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else (ROOT / result)


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("main experiment configuration must be a mapping")
    return value


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class ReaderCache:
    def __init__(self, run_root: Path, users: list[str]):
        self.run_root = run_root
        self.users = users
        self.cache: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}

    @staticmethod
    def file_key(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def rows_for_user(self, user_id: str) -> dict[str, str]:
        stem = self.run_root / f"user_{user_id}_shard_0"
        paths = [stem.with_suffix(".partial.json"), stem.with_suffix(".json")]
        key = tuple(self.file_key(path) or (0, 0) for path in paths)
        cached = self.cache.get(user_id)
        if cached is not None and cached[0] == key:
            return cached[1]
        result: dict[str, str] = {}
        for path in paths:
            document = load_json(path)
            for row in document.get("rows") or []:
                identity = row.get("question_id")
                answer = row.get("answer_sha256")
                if identity:
                    result[identity] = answer
        self.cache[user_id] = (key, result)
        return result

    def snapshot(self) -> dict[str, dict[str, str]]:
        return {user: self.rows_for_user(user) for user in self.users}


class JudgeCache:
    def __init__(self, rows_root: Path):
        self.rows_root = rows_root
        self.signature: tuple[int, int] | None = None
        self.rows: dict[str, dict] = {}

    def snapshot(self) -> dict[str, dict]:
        try:
            stat = self.rows_root.stat()
            signature = stat.st_mtime_ns, stat.st_size
        except OSError:
            signature = None
        if signature is not None and signature == self.signature:
            return self.rows
        rows: dict[str, dict] = {}
        if self.rows_root.exists():
            for path in self.rows_root.glob("*/*.json"):
                document = load_json(path)
                identity = document.get("question_id") or path.stem
                if identity:
                    rows[identity] = document
        self.rows = rows
        self.signature = signature
        return rows


def gpu_rows() -> list[dict]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4:
            continue
        result.append({"index": int(values[0]), "memory_used_mib": int(values[1]),
                       "memory_total_mib": int(values[2]), "utilization_pct": int(values[3])})
    return result


def pid_status(path: Path) -> dict:
    if not path.exists():
        return {"pid": None, "alive": False}
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return {"pid": None, "alive": False}
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    return {"pid": pid, "alive": alive}


def rate_eta(done: int, total: int, previous: tuple[float, int] | None,
             current_time: float) -> tuple[float | None, float | None]:
    if previous is None:
        return None, None
    elapsed = current_time - previous[0]
    delta = done - previous[1]
    if elapsed <= 0 or delta <= 0:
        return None, None
    rate = delta / elapsed
    eta = (total - done) / rate if rate > 0 else None
    return rate, eta


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def snapshot(config: dict, reader_cache: ReaderCache, judge_cache: JudgeCache,
             previous_reader: tuple[float, int] | None,
             previous_judge: tuple[float, int] | None) -> dict:
    paths = config["paths"]
    run_root = resolve(paths["run_root"])
    prepared = resolve(paths["prepared"])
    cohort = load_json(resolve(paths["cohort"]))
    users = sorted(cohort.get("question_ids_by_user", {}), key=int)
    expected_by_user = {user: set(cohort["question_ids_by_user"][user]) for user in users}
    expected_total = sum(len(value) for value in expected_by_user.values())
    split_by_question: dict[str, str] = {}
    with (prepared / "manifests.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split_by_question[row["question_id"]] = row["split"]
    expected_ids = {identity for values in expected_by_user.values() for identity in values}
    expected_by_split: dict[str, int] = {}
    for identity in expected_ids:
        split = split_by_question[identity]
        expected_by_split[split] = expected_by_split.get(split, 0) + 1

    reader = reader_cache.snapshot()
    reader_ids = {identity for values in reader.values() for identity in values}
    reader_count = len(reader_ids)
    judged_rows = judge_cache.snapshot()
    judged_ids = set(judged_rows) & set(split_by_question)
    correct = sum(row.get("judge_label") == "CORRECT" and bool(row.get("parse_valid"))
                  for identity, row in judged_rows.items() if identity in judged_ids)
    invalid = sum(not bool(row.get("parse_valid")) for identity, row in judged_rows.items()
                  if identity in judged_ids)
    judged_count = len(judged_ids)
    split_stats: dict[str, dict] = {}
    for split, expected in sorted(expected_by_split.items()):
        split_judged = {identity for identity in judged_ids
                        if split_by_question.get(identity) == split}
        split_correct = sum(judged_rows[identity].get("judge_label") == "CORRECT"
                            and bool(judged_rows[identity].get("parse_valid"))
                            for identity in split_judged)
        split_reader = {identity for identity in reader_ids
                        if split_by_question.get(identity) == split}
        split_stats[split] = {
            "expected": expected,
            "reader_completed": len(split_reader),
            "judged": len(split_judged),
            "correct": split_correct,
            "success_rate": split_correct / len(split_judged) if split_judged else None,
        }

    current_time = time.time()
    reader_rate, reader_eta = rate_eta(reader_count, expected_total, previous_reader, current_time)
    judge_rate, judge_eta = rate_eta(judged_count, expected_total, previous_judge, current_time)
    per_user = {}
    for user in users:
        expected = expected_by_user[user]
        done = len(set(reader[user]) & expected)
        per_user[user] = {"completed": done, "expected": len(expected),
                          "remaining": len(expected) - done}
    orchestrator = load_json(run_root / "orchestrator_status.json")
    judge_status = load_json(run_root / "judge_status.json")
    result = {
        "schema": "mobilemem-full-monitor-v1",
        "time": now(),
        "run_root": str(run_root),
        "reader": {
            "completed": reader_count,
            "expected": expected_total,
            "fraction": reader_count / expected_total if expected_total else None,
            "rate_per_second": reader_rate,
            "eta_seconds": reader_eta,
        },
        "judge": {
            "judged": judged_count,
            "expected": expected_total,
            "correct": correct,
            "invalid": invalid,
            "success_rate_judged": correct / judged_count if judged_count else None,
            "all_question_lower_bound": correct / expected_total if expected_total else None,
            "rate_per_second": judge_rate,
            "eta_seconds": judge_eta,
        },
        "by_split": split_stats,
        "by_user": per_user,
        "orchestrator": orchestrator,
        "judge_status": judge_status,
        "processes": {
            "orchestrator": pid_status(run_root / "orchestrator.pid"),
            "judge": pid_status(run_root / "judge.pid"),
        },
        "gpus": gpu_rows(),
    }
    atomic_write(run_root / "monitor_snapshot.json", result)
    return result


def render(value: dict) -> str:
    reader = value["reader"]
    judge = value["judge"]
    lines = [
        "=" * 88,
        f"MobileMem full experiment  {value['time']}",
        "=" * 88,
        (f"Reader  {reader['completed']:5d}/{reader['expected']} "
         f"({percent(reader['completed'], reader['expected'])})"
         f"  rate={reader['rate_per_second'] or 0:.2f}/s"
         f"  ETA={duration(reader['eta_seconds'])}"),
        (f"Judge   {judge['judged']:5d}/{judge['expected']} "
         f"({percent(judge['judged'], judge['expected'])})"
         f"  correct={judge['correct']}  invalid={judge['invalid']}"
         f"  success={percent(judge['correct'], judge['judged'])}"
         f"  all-lower={percent(judge['correct'], judge['expected'])}"),
        (f"Judge rate={judge['rate_per_second'] or 0:.2f}/s"
         f"  ETA={duration(judge['eta_seconds'])}"
         f"  state={value.get('judge_status', {}).get('state', 'not-started')}"),
        "",
        "Split                  Reader          Judged        Correct    Success",
    ]
    for split, stats in value["by_split"].items():
        lines.append(
            f"{split:20s} {stats['reader_completed']:5d}/{stats['expected']:<5d}"
            f" {stats['judged']:6d}/{stats['expected']:<5d}"
            f" {stats['correct']:7d}"
            f" {percent(stats['correct'], stats['judged']):>9s}")
    lines.extend(["", "Active workers:"])
    workers = value.get("orchestrator", {}).get("workers", {})
    if workers:
        for gpu, state in sorted(workers.items(), key=lambda item: int(item[0])):
            lines.append(f"  GPU {gpu}: {state.get('state')} user={state.get('user_id', '-')}"
                         f" last={state.get('last', {}).get('user_id', '-')}")
    else:
        lines.append("  not started or no status yet")
    lines.append("GPU memory/utilization:")
    for gpu in value.get("gpus", []):
        lines.append(f"  GPU {gpu['index']}: {gpu['memory_used_mib']:5d}/"
                     f"{gpu['memory_total_mib']} MiB, util={gpu['utilization_pct']}%")
    processes = value["processes"]
    lines.append(
        f"Processes: orchestrator={processes['orchestrator']['alive']} "
        f"(pid {processes['orchestrator']['pid']}), "
        f"judge={processes['judge']['alive']} (pid {processes['judge']['pid']})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0.0,
                        help="seconds between refreshes; zero prints one snapshot")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    run_root = resolve(config["paths"]["run_root"])
    cohort = load_json(resolve(config["paths"]["cohort"]))
    users = sorted(cohort.get("question_ids_by_user", {}), key=int)
    reader_cache = ReaderCache(run_root, users)
    judge_cache = JudgeCache(run_root / "judge_rows")
    previous_reader = None
    previous_judge = None
    while True:
        value = snapshot(config, reader_cache, judge_cache, previous_reader, previous_judge)
        previous_reader = time.time(), value["reader"]["completed"]
        previous_judge = time.time(), value["judge"]["judged"]
        if args.watch > 0:
            print("\033[2J\033[H", end="")
        print(render(value), flush=True)
        if args.watch <= 0:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
