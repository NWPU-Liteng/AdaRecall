#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import re
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, resolve, sha256_file, write_json_new  # noqa: E402
from v21.uot_recipes import build_preregistered_recipes, canonical_json_sha256  # noqa: E402


WORKER_SCRIPT = VERSION_ROOT / "scripts" / "82_run_v21_uot_numerical_audit.py"
RUN_ID_PATTERN = re.compile(r"^v21_[A-Za-z0-9_.-]+$")
ATTEMPT_PATTERN = re.compile(r"^attempt_(\d+)$")
CUDA_RUNPY_BOOTSTRAP = """
import json
import runpy
import sys
import time
import torch

device = sys.argv[1]
retries = int(sys.argv[2])
target = torch.device(device)
last_error = "CUDA context was not initialized"
for attempt in range(1, retries + 1):
    try:
        torch.cuda.init()
        torch.empty(1, device=target)
        torch.cuda.synchronize(target)
        print(json.dumps({"cuda_bootstrap_preinitialized": str(target), "cuda_bootstrap_attempt": attempt}), flush=True)
        break
    except Exception as error:
        last_error = f"{type(error).__name__}: {error}"
        if attempt == retries:
            raise RuntimeError(f"CUDA bootstrap failed after {retries} attempts for {device}: {last_error}")
        time.sleep(1.0)
script = sys.argv[3]
sys.argv = [script, *sys.argv[4:]]
runpy.run_path(script, run_name="__main__")
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bool_arg(value: str) -> bool:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _run_directory(run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run-id must start with v21_ and contain only letters, digits, _, ., or -")
    return VERSION_ROOT / "artifacts" / "v21_uot" / run_id


def _load_completed_attempts(recipe_directory: Path, recipe_id: int) -> list[tuple[int, Path, dict]]:
    completed: list[tuple[int, Path, dict]] = []
    attempts_directory = recipe_directory / "attempts"
    if not attempts_directory.is_dir():
        return completed
    for attempt_directory in sorted(attempts_directory.iterdir()):
        match = ATTEMPT_PATTERN.fullmatch(attempt_directory.name)
        if not match or not attempt_directory.is_dir():
            continue
        attempt = int(match.group(1))
        result_path = attempt_directory / f"v21_uot_recipe_{recipe_id:02d}_attempt_{attempt:03d}.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("execution_status") == "completed":
            completed.append((attempt, result_path, result))
    return completed


def _next_attempt(recipe_directory: Path) -> int:
    attempts_directory = recipe_directory / "attempts"
    if not attempts_directory.is_dir():
        return 1
    values = []
    for child in attempts_directory.iterdir():
        match = ATTEMPT_PATTERN.fullmatch(child.name)
        if match and child.is_dir():
            values.append(int(match.group(1)))
    return max(values, default=0) + 1


def _preinitialize_cuda(device: str, retries: int) -> None:
    if not device.lower().startswith("cuda"):
        return
    import torch

    if retries < 1:
        raise ValueError("cuda-init-retries must be positive")
    target = torch.device(device)
    last_error = "CUDA context was not initialized"
    for attempt in range(1, retries + 1):
        try:
            # The environment intermittently reports is_available=False before
            # its first real CUDA context initialization. Attempt the actual
            # context operation, which is the condition required by the worker.
            reported_available = bool(torch.cuda.is_available())
            torch.cuda.init()
            torch.empty(1, device=target)
            torch.cuda.synchronize(target)
            print(json.dumps({
                "cuda_context_preinitialized": str(target),
                "cuda_init_attempt": attempt,
                "cuda_reported_available_before_init": reported_available,
            }), flush=True)
            return
        except Exception as error:  # The next fresh worker attempt remains the recovery boundary.
            last_error = f"{type(error).__name__}: {error}"
        if attempt < retries:
            time.sleep(1.0)
    raise RuntimeError(f"CUDA device initialization failed after {retries} attempts for {device}: {last_error}")


def _worker_main(args: argparse.Namespace) -> None:
    _preinitialize_cuda(args.device, args.cuda_init_retries)
    worker_args = [
        str(WORKER_SCRIPT),
        "--full-preregistered-screen",
        "--recipe-start",
        str(args.recipe_start),
        "--recipe-stop",
        str(args.recipe_stop),
        "--device",
        args.device,
        "--output-dir",
        args.output_dir,
        "--run-id",
        args.run_id,
        "--attempt",
        str(args.attempt),
        "--config-name",
        args.config_name,
        "--input-embedding-hash",
        args.input_embedding_hash,
    ]
    previous_argv = sys.argv
    try:
        sys.argv = worker_args
        runpy.run_path(str(WORKER_SCRIPT), run_name="__main__")
    finally:
        sys.argv = previous_argv


def _build_run_manifest(args: argparse.Namespace, config: dict, data: dict, embedding_hash: str) -> dict:
    recipes = build_preregistered_recipes(config)
    return {
        "schema_version": "v21-uot-recipe-run-2",
        "run_id": args.run_id,
        "created_at_utc": _utc_now(),
        "launcher": "scripts/84_launch_v21_uot_recipe_shards.py",
        "worker": "scripts/82_run_v21_uot_numerical_audit.py",
        "config_name": args.config_name,
        "config_hash": sha256_file(VERSION_ROOT / "configs" / args.config_name),
        "recipe_grid_hash": canonical_json_sha256(recipes),
        "recipes": recipes,
        "input_embedding_path": str(resolve(data["memory"]["embedding_vectors"])),
        "input_embedding_hash": embedding_hash,
        "input_embedding_ids_path": str(resolve(data["memory"]["embedding_ids"])),
        "input_embedding_ids_hash": sha256_file(resolve(data["memory"]["embedding_ids"])),
        "train_plan_path": str(resolve(data["split"]["train_query_plan"])),
        "train_plan_hash": sha256_file(resolve(data["split"]["train_query_plan"])),
        "primary_convergence_rule": "potential_delta <= 1e-6 within 300 iterations",
        "selection_prohibition": "the numerical screen is a stability exclusion screen only; no model selection occurs here",
        "calibration_read": False,
        "formal_test_read": False,
        "legacy_excluded_from_merge": [
            {
                "path": "aggregate/v21_uot_full_preregistered_numerical_screen.csv",
                "reason": "legacy monolithic aggregate has no independent recipe attempts, provenance hashes, or required secondary diagnostics",
            },
            {
                "path": "aggregate/v21_uot_full_preregistered_numerical_screen_part_01_03.csv",
                "reason": "legacy overlapping partial range",
            },
            {
                "path": "aggregate/v21_uot_full_preregistered_numerical_screen_part_01_04.csv",
                "reason": "legacy overlapping partial range",
            },
            {
                "path": "aggregate/v21_uot_full_preregistered_numerical_screen_part_04_04.csv",
                "reason": "preserved legacy recipe-4 numerical reference; not an r2 compliant shard",
            },
        ],
    }


def _write_failure(
    attempt_directory: Path,
    recipe: dict,
    args: argparse.Namespace,
    run_manifest: dict,
    log_name: str,
    *,
    execution_status: str,
    exit_code: int | None,
    detail: str,
) -> Path:
    failure_path = attempt_directory / f"v21_uot_recipe_{int(recipe['recipe_id']):02d}_attempt_{args.attempt:03d}_failure.json"
    record = {
        "schema_version": "v21-uot-recipe-failure-2",
        "execution_status": execution_status,
        "run_id": args.run_id,
        "attempt": args.attempt,
        "recipe": recipe,
        "config_hash": run_manifest["config_hash"],
        "input_embedding_hash": run_manifest["input_embedding_hash"],
        "exit_code": exit_code,
        "detail": detail,
        "log": log_name,
        "created_at_utc": _utc_now(),
        "calibration_read": False,
        "formal_test_read": False,
    }
    write_json_new(str(failure_path), record)
    return failure_path


def _launch_recipe(recipe: dict, args: argparse.Namespace, run_directory: Path, run_manifest: dict) -> dict:
    recipe_id = int(recipe["recipe_id"])
    recipe_directory = run_directory / "shards" / f"recipe_{recipe_id:02d}_{recipe['recipe_key']}"
    completed = _load_completed_attempts(recipe_directory, recipe_id)
    if len(completed) > 1:
        return {
            "recipe_id": recipe_id,
            "status": "duplicate_completed_attempts",
            "paths": [str(path.relative_to(run_directory)) for _, path, _ in completed],
        }
    if completed:
        if args.resume:
            return {
                "recipe_id": recipe_id,
                "status": "skipped_completed",
                "path": str(completed[0][1].relative_to(run_directory)),
            }
        return {"recipe_id": recipe_id, "status": "existing_completed_requires_resume"}
    attempt = _next_attempt(recipe_directory)
    attempt_directory = recipe_directory / "attempts" / f"attempt_{attempt:03d}"
    attempt_directory.mkdir(parents=True, exist_ok=False)
    worker_args = argparse.Namespace(
        device=args.device,
        recipe_start=recipe_id,
        recipe_stop=recipe_id,
        output_dir=str(attempt_directory),
        run_id=args.run_id,
        attempt=attempt,
        config_name=args.config_name,
        input_embedding_hash=run_manifest["input_embedding_hash"],
        cuda_init_retries=args.cuda_init_retries,
    )
    worker_command = [
        str(Path(__file__).resolve()),
        "--worker",
        "--device",
        worker_args.device,
        "--recipe-start",
        str(recipe_id),
        "--recipe-stop",
        str(recipe_id),
        "--output-dir",
        worker_args.output_dir,
        "--run-id",
        worker_args.run_id,
        "--attempt",
        str(worker_args.attempt),
        "--config-name",
        worker_args.config_name,
        "--input-embedding-hash",
        worker_args.input_embedding_hash,
        "--cuda-init-retries",
        str(args.cuda_init_retries),
    ]
    if worker_args.device.lower().startswith("cuda"):
        command = [
            sys.executable,
            "-c",
            CUDA_RUNPY_BOOTSTRAP,
            worker_args.device,
            str(args.cuda_init_retries),
            *worker_command,
        ]
    else:
        command = [sys.executable, *worker_command]
    log_name = f"v21_uot_recipe_{recipe_id:02d}_attempt_{attempt:03d}.log"
    log_path = attempt_directory / log_name
    if args.in_process:
        with log_path.open("x", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                try:
                    _worker_main(worker_args)
                    execution_status = "completed"
                    exit_code = 0
                    detail = "in-process worker completed"
                except Exception:
                    traceback.print_exc(file=handle)
                    execution_status = "failed"
                    exit_code = 1
                    detail = "in-process worker raised an exception"
    else:
        try:
            completed_process = subprocess.run(
                command,
                cwd=VERSION_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            output = completed_process.stdout or ""
            exit_code = completed_process.returncode
            execution_status = "completed" if exit_code == 0 else "failed"
            detail = "worker completed" if exit_code == 0 else f"worker exited with code {exit_code}"
        except subprocess.TimeoutExpired as error:
            raw_output = error.stdout or ""
            output = raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else raw_output
            exit_code = None
            execution_status = "timed_out"
            detail = f"worker exceeded timeout_seconds={args.timeout_seconds}"
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write(output)
    result_path = attempt_directory / f"v21_uot_recipe_{recipe_id:02d}_attempt_{attempt:03d}.json"
    if execution_status == "completed" and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("execution_status") == "completed":
            return {
                "recipe_id": recipe_id,
                "status": "completed",
                "attempt": attempt,
                "path": str(result_path.relative_to(run_directory)),
                "numerical_status": result.get("numerical_status"),
            }
        execution_status = "invalid_result"
        detail = "worker returned zero but shard JSON is not a completed record"
    elif execution_status == "completed":
        execution_status = "invalid_result"
        detail = "worker returned zero without a completed shard JSON"
    failure_path = _write_failure(
        attempt_directory,
        recipe,
        argparse.Namespace(run_id=args.run_id, attempt=attempt),
        run_manifest,
        log_name,
        execution_status=execution_status,
        exit_code=exit_code,
        detail=detail,
    )
    return {
        "recipe_id": recipe_id,
        "status": execution_status,
        "attempt": attempt,
        "failure_path": str(failure_path.relative_to(run_directory)),
    }


def _prepare_run(args: argparse.Namespace) -> tuple[Path, dict]:
    data = load_config("v21_data.json")
    config = load_config(args.config_name)
    run_directory = _run_directory(args.run_id)
    if args.overwrite:
        raise RuntimeError("--overwrite true is rejected by the v21 no-overwrite policy; create a new --run-id")
    manifest_path = run_directory / "v21_uot_run_manifest.json"
    if run_directory.exists():
        if not args.resume:
            raise FileExistsError(f"run directory exists; use --resume or a new --run-id: {run_directory}")
        if not manifest_path.is_file():
            raise RuntimeError(f"cannot resume without a v21 run manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        current_hash = sha256_file(VERSION_ROOT / "configs" / args.config_name)
        if manifest.get("config_hash") != current_hash:
            raise RuntimeError("resume config hash differs from the original local run manifest")
        return run_directory, manifest
    if args.resume:
        raise FileNotFoundError(f"cannot resume a missing run directory: {run_directory}")
    embedding_hash = sha256_file(resolve(data["memory"]["embedding_vectors"]))
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest = _build_run_manifest(args, config, data, embedding_hash)
    write_json_new(str(manifest_path), manifest)
    return run_directory, manifest


def _launcher_main(args: argparse.Namespace) -> int:
    run_directory, run_manifest = _prepare_run(args)
    recipes = build_preregistered_recipes(load_config(args.config_name))
    recipe_stop = len(recipes) if args.recipe_stop == 0 else args.recipe_stop
    if args.recipe_start < 1 or recipe_stop < args.recipe_start or recipe_stop > len(recipes):
        raise RuntimeError(f"invalid recipe interval [{args.recipe_start}, {recipe_stop}] for {len(recipes)} recipes")
    selected = recipes[args.recipe_start - 1 : recipe_stop]
    if args.in_process and len(selected) != 1:
        raise RuntimeError("--in-process requires exactly one recipe; launch each recipe independently")
    rows = [_launch_recipe(recipe, args, run_directory, run_manifest) for recipe in selected]
    summary = {
        "schema_version": "v21-uot-launcher-summary-2",
        "run_id": args.run_id,
        "created_at_utc": _utc_now(),
        "recipe_interval": [args.recipe_start, recipe_stop],
        "device": args.device,
        "timeout_seconds": args.timeout_seconds,
        "resume": args.resume,
        "in_process": args.in_process,
        "results": rows,
        "calibration_read": False,
        "formal_test_read": False,
    }
    device_key = re.sub(r"[^A-Za-z0-9_.-]", "_", args.device)
    summary_path = run_directory / (
        f"v21_uot_launcher_summary_{_utc_now()}_recipes_"
        f"{args.recipe_start:02d}_{recipe_stop:02d}_{device_key}.json"
    )
    write_json_new(str(summary_path), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    unsuccessful = [row for row in rows if row["status"] not in {"completed", "skipped_completed"}]
    return 1 if unsuccessful else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch v21 UOT numerical recipe shards with local run provenance.")
    parser.add_argument("--recipe-start", type=int, default=1)
    parser.add_argument("--recipe-stop", type=int, default=27)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", default=f"v21_uot_screen_r2_{_utc_now()}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", type=_bool_arg, default=False, metavar="{true,false}")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--cuda-init-retries", type=int, default=5)
    parser.add_argument("--config-name", default="v21_uot_semantic.json")
    parser.add_argument("--in-process", action="store_true", help="Run one already-bootstrapped recipe in this process.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--input-embedding-hash", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    if args.worker:
        _worker_main(args)
        return
    raise SystemExit(_launcher_main(args))


if __name__ == "__main__":
    main()
