#!/usr/bin/env python3
"""Incrementally judge completed Reader rows with crash-safe per-question files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml


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


def parse_label(response: str) -> str | None:
    try:
        parsed = json.loads(response)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or parsed.get("label") not in ("CORRECT", "WRONG"):
        return None
    return parsed["label"]


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("main experiment configuration must be a mapping")
    return value


def resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else (ROOT / result)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_run_rows(run_root: Path, user_id: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for suffix in (".partial.json", ".json"):
        path = run_root / f"user_{user_id}_shard_0{suffix}"
        if not path.exists():
            continue
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("person_id") != user_id:
            raise ValueError(f"Reader checkpoint person mismatch: {path}")
        for row in document.get("rows") or []:
            identity = row.get("question_id")
            previous = rows.get(identity)
            if previous is not None and previous.get("answer_sha256") != row.get("answer_sha256"):
                raise ValueError(f"Reader answer changed for {user_id}/{identity}")
            rows[identity] = row
    return rows


def ledger_path(rows_root: Path, question_id: str) -> Path:
    user_id = question_id.split("_", 1)[0]
    return rows_root / user_id / f"{question_id}.json"


def build_prompt(template: str, manifest: dict, label: dict, row: dict) -> str:
    replacements = {
        "[Question]": manifest["question"],
        "[Gold Answer]": json.dumps(label["golden_answers"], ensure_ascii=False),
        "[Evidences]": json.dumps(label["raw_evidence"], ensure_ascii=False),
        "[Generated Answer]": row["answer"],
    }
    sections = template.rsplit("QUESTION TO EVALUATE", 1)
    if len(sections) != 2:
        raise ValueError("unexpected Omni evaluation prompt structure")
    prompt = sections[0] + "QUESTION TO EVALUATE\n" + "\n".join(
        field + ": " + replacements[placeholder] for field, placeholder in (
            ("Question", "[Question]"),
            ("Gold answer", "[Gold Answer]"),
            ("Evidence", "[Evidences]"),
            ("Generated answer", "[Generated Answer]")))
    return prompt + ('\nReturn only one JSON object with keys "reason" and "label". '
                     'The label must be CORRECT or WRONG.')


def import_judge_seeds(config: dict, manifests: dict[str, dict], labels: dict[str, dict],
                       reader_rows: dict[str, dict], rows_root: Path) -> tuple[int, list[str]]:
    imported = 0
    warnings: list[str] = []
    for raw_path in config.get("resume", {}).get("judge_seed_results", []):
        path = resolve(raw_path)
        if not path.exists():
            continue
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            warnings.append(f"unreadable judge seed: {path}")
            continue
        for seed in document.get("rows") or []:
            identity = seed.get("question_id")
            if identity not in manifests or identity not in reader_rows:
                continue
            row = reader_rows[identity]
            if seed.get("prediction_sha256") != row.get("answer_sha256"):
                continue
            destination = ledger_path(rows_root, identity)
            if destination.exists():
                continue
            result = {
                "schema": "mobilemem-incremental-judge-row-v1",
                "person_id": manifests[identity]["person_id"],
                "question_id": identity,
                "split": manifests[identity]["split"],
                "question_type": labels[identity].get("question_type"),
                "answer_sha256": row.get("answer_sha256"),
                "stop_reason": row.get("stop_reason"),
                "judge_label": seed.get("judge_label"),
                "parse_valid": bool(seed.get("parse_valid")),
                "response": seed.get("response"),
                "judge_seconds": seed.get("judge_seconds"),
                "judged_at": seed.get("judged_at") or now(),
                "source": "seed",
                "seed_source": str(path),
            }
            atomic_write(destination, result)
            imported += 1
    return imported, warnings


def gpu_free_gib() -> tuple[float | None, float | None, str | None]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, "cuda unavailable"
        free, total = torch.cuda.mem_get_info()
        return free / 1024**3, total / 1024**3, None
    except Exception as error:
        return None, None, repr(error)


def judge(config_path: Path) -> int:
    config = load_config(config_path)
    paths = config["paths"]
    run_root = resolve(paths["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_root / ".judge.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another incremental judge owns {run_root / '.judge.lock'}")
    atomic_write(run_root / "judge.pid", os.getpid())

    prepared = resolve(paths["prepared"])
    cohort = json.loads(resolve(paths["cohort"]).read_text())
    audit = json.loads((prepared / "audit.json").read_text())
    for name in ("labels.jsonl", "manifests.jsonl"):
        if digest(prepared / name) != audit["artifacts_sha256"][name]:
            raise ValueError(f"prepared evaluation input hash mismatch: {name}")
    manifests = {row["question_id"]: row for row in load_jsonl(prepared / "manifests.jsonl")}
    labels = {row["question_id"]: row for row in load_jsonl(prepared / "labels.jsonl")}
    order = {identity: index for index, identity in enumerate(
        identity for user in sorted(cohort["question_ids_by_user"], key=int)
        for identity in cohort["question_ids_by_user"][user])}
    expected = set(order)
    prompt_source = resolve(config["evaluation"]["judge_prompt_source"])
    marker = "Prompt for LLM-as-a-Judge Evaluation"
    source = prompt_source.read_text()
    if source.count(marker) != 1:
        raise ValueError("unexpected Omni prompt schema")
    template = source.split(marker, 1)[1].strip()
    rows_root = run_root / "judge_rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "judge_status.json"
    model_path = resolve(config["models"]["judge"])
    registration_path = run_root / "judge_registration.json"
    if not registration_path.exists():
        model_hashes = {
            path.name: digest(path) for path in sorted(model_path.iterdir())
            if path.is_file() and path.suffix in (".json", ".safetensors", ".txt")
        }
        atomic_write(registration_path, {
            "schema": "mobilemem-incremental-judge-registration-v1",
            "created_at": now(),
            "judge_model": str(model_path),
            "model_files_sha256": model_hashes,
            "prepared_audit_sha256": digest(prepared / "audit.json"),
            "prompt_source_sha256": digest(prompt_source),
            "script_sha256": digest(Path(__file__)),
            "temperature": 0.0,
            "max_new_tokens": int(config["evaluation"].get("judge_max_new_tokens", 512)),
            "protocol": "Omni strict coverage/evidence rubric + JSON-only output clarification",
            "boundary": ("fixed local Qwen3-VL-8B proxy; not the official Qwen3-14B score; "
                         "Reader labels are read only by this evaluation process"),
        })

    min_free = float(config["execution"].get("judge_min_free_gib", 20))
    status: dict[str, Any] = {
        "schema": "mobilemem-incremental-judge-status-v1",
        "started_at": now(),
        "updated_at": now(),
        "state": "starting",
        "expected_questions": len(expected),
        "judged": 0,
        "correct": 0,
        "invalid": 0,
        "model_loaded": False,
    }

    def scan_reader_rows() -> dict[str, dict]:
        result: dict[str, dict] = {}
        for user_id in sorted(cohort["question_ids_by_user"], key=int):
            for identity, row in read_run_rows(run_root, user_id).items():
                if identity in expected:
                    result[identity] = row
        return result

    def judged_identities() -> set[str]:
        return {path.stem for path in rows_root.glob("*/*.json") if path.is_file()}

    def update_status(state: str | None = None, **extra: Any) -> None:
        if state is not None:
            status["state"] = state
        status.update(extra)
        status["updated_at"] = now()
        atomic_write(status_path, status)

    reader_rows = scan_reader_rows()
    imported, warnings = import_judge_seeds(config, manifests, labels, reader_rows, rows_root)
    judged = judged_identities()
    update_status("waiting_for_gpu" if imported == 0 else "seed_imported",
                  judged=len(judged), seed_imported=imported, warnings=warnings)

    while True:
        if (run_root / "STOP_JUDGE").exists():
            update_status("stopped_by_user")
            return 0
        free, total, error = gpu_free_gib()
        if free is not None and free >= min_free:
            break
        update_status("waiting_for_gpu", gpu_free_gib=free, gpu_total_gib=total,
                      gpu_error=error, required_free_gib=min_free)
        time.sleep(30)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.set_num_threads(2)
    torch.manual_seed(0)
    update_status("loading_model", gpu_free_gib=free, gpu_total_gib=total)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda:0").eval()
    update_status("running", model_loaded=True)
    max_new_tokens = int(config["evaluation"].get("judge_max_new_tokens", 512))

    while True:
        if (run_root / "STOP_JUDGE").exists():
            update_status("stopped_by_user")
            return 0
        reader_rows = scan_reader_rows()
        imported_now, warnings = import_judge_seeds(
            config, manifests, labels, reader_rows, rows_root)
        imported += imported_now
        judged = judged_identities()
        remaining = sorted(expected - judged, key=lambda identity: order[identity])
        available = [identity for identity in remaining if identity in reader_rows]
        correct = sum(json.loads(ledger_path(rows_root, identity).read_text()).get("judge_label") == "CORRECT"
                      for identity in judged)
        invalid = sum(not json.loads(ledger_path(rows_root, identity).read_text()).get("parse_valid", False)
                      for identity in judged)
        update_status("running", judged=len(judged), correct=correct, invalid=invalid,
                      available_unjudged=len(available), seed_imported=imported)
        if len(judged) >= len(expected):
            update_status("complete", judged=len(judged), correct=correct, invalid=invalid)
            atomic_write(run_root / "judge_complete.json", {
                "schema": "mobilemem-incremental-judge-complete-v1",
                "complete": True,
                "finished_at": now(),
                "questions": len(judged),
                "correct": correct,
                "invalid": invalid,
                "success_rate": correct / len(judged) if judged else None,
                "boundary": "fixed local Qwen3-VL-8B proxy; invalid Judge output counts as failure",
            })
            return 0
        if not available:
            time.sleep(15)
            continue
        identity = available[0]
        row = reader_rows[identity]
        manifest = manifests[identity]
        label = labels[identity]
        prompt = build_prompt(template, manifest, label, row)
        started = time.perf_counter()
        response = None
        error = None
        try:
            messages = [
                {"role": "system", "content": "You evaluate answer correctness. Follow the supplied evidence rubric."},
                {"role": "user", "content": prompt},
            ]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = processor(text=[rendered], return_tensors="pt").to("cuda:0")
            if inputs.input_ids.shape[1] > 32768:
                raise ValueError("judge input exceeds registered context")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            response = processor.tokenizer.decode(
                output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        except Exception as caught:
            error = repr(caught)
        label_value = parse_label(response) if response is not None else None
        result = {
            "schema": "mobilemem-incremental-judge-row-v1",
            "person_id": manifest["person_id"],
            "question_id": identity,
            "split": manifest["split"],
            "question_type": label.get("question_type"),
            "answer_sha256": row.get("answer_sha256"),
            "stop_reason": row.get("stop_reason"),
            "judge_label": label_value,
            "parse_valid": label_value is not None,
            "response": response,
            "error": error,
            "judge_seconds": time.perf_counter() - started,
            "judged_at": now(),
            "source": "incremental",
        }
        atomic_write(ledger_path(rows_root, identity), result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(judge(args.config.resolve()))


if __name__ == "__main__":
    main()
