#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.uot_recipes import REQUIRED_NUMERICAL_DIAGNOSTICS, array_sha256, build_preregistered_recipes, recipe_matches  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_directory(run_id: str) -> Path:
    return VERSION_ROOT / "artifacts" / "v21_uot" / run_id


def _completed_records(run_directory: Path, recipe: dict) -> list[tuple[Path, dict]]:
    recipe_id = int(recipe["recipe_id"])
    recipe_directory = run_directory / "shards" / f"recipe_{recipe_id:02d}_{recipe['recipe_key']}" / "attempts"
    records: list[tuple[Path, dict]] = []
    if not recipe_directory.is_dir():
        return records
    for attempt_directory in sorted(recipe_directory.iterdir()):
        if not attempt_directory.is_dir():
            continue
        candidates = sorted(attempt_directory.glob(f"v21_uot_recipe_{recipe_id:02d}_attempt_*.json"))
        for path in candidates:
            if path.name.endswith("_failure.json"):
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("execution_status") == "completed":
                records.append((path, result))
    return records


def _validate_artifact_array(path: Path, expected_hash: str, expected_shape: tuple[int, ...], *, incidence: bool) -> list[str]:
    errors = []
    if not path.is_file():
        return [f"missing artifact {path}"]
    value = np.load(path, allow_pickle=False)
    if tuple(value.shape) != expected_shape:
        errors.append(f"artifact shape {value.shape} != {expected_shape}: {path}")
    if array_sha256(value) != expected_hash:
        errors.append(f"artifact hash mismatch: {path}")
    if not np.isfinite(value).all():
        errors.append(f"artifact contains non-finite values: {path}")
    if incidence and (value < 0).any():
        errors.append(f"artifact contains negative values: {path}")
    if incidence and len(value):
        row_error = np.abs(value.sum(axis=1) - 1.0).max()
        if not np.isfinite(row_error) or row_error > 1e-5:
            errors.append(f"H_sem incidence is not row normalized (max error {row_error}): {path}")
    return errors


def _validate_csv(path: Path, recipe_id: int, run_id: str) -> list[str]:
    if not path.is_file():
        return [f"missing CSV {path}"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        return [f"CSV must contain exactly one recipe row: {path}"]
    row = rows[0]
    errors = []
    if int(row.get("recipe_id", -1)) != recipe_id:
        errors.append(f"CSV recipe mismatch: {path}")
    if row.get("run_id") != run_id:
        errors.append(f"CSV run_id mismatch: {path}")
    for field in REQUIRED_NUMERICAL_DIAGNOSTICS:
        if field not in row:
            errors.append(f"CSV missing required diagnostic {field}: {path}")
    return errors


def _validate_record(run_manifest: dict, recipe: dict, path: Path, result: dict) -> tuple[list[str], dict]:
    errors: list[str] = []
    diagnostics = result.get("diagnostics", {})
    recipe_id = int(recipe["recipe_id"])
    if result.get("run_id") != run_manifest["run_id"]:
        errors.append(f"run_id mismatch in {path}")
    if result.get("screen_kind") != "full_preregistered":
        errors.append(f"smoke or unknown screen kind in {path}")
    if not recipe_matches(recipe, result.get("recipe", {})):
        errors.append(f"preregistered recipe parameters mismatch in {path}")
    if result.get("config_hash") != run_manifest["config_hash"]:
        errors.append(f"config hash mismatch in {path}")
    if result.get("input_embedding_hash") != run_manifest["input_embedding_hash"]:
        errors.append(f"embedding hash mismatch in {path}")
    if result.get("input_embedding_ids_hash") != run_manifest["input_embedding_ids_hash"]:
        errors.append(f"embedding-ID hash mismatch in {path}")
    if result.get("calibration_read") is not False or result.get("formal_test_read") is not False:
        errors.append(f"locked split access flags invalid in {path}")
    for field in REQUIRED_NUMERICAL_DIAGNOSTICS:
        if field not in diagnostics:
            errors.append(f"missing numerical diagnostic {field} in {path}")
    if diagnostics.get("finite_ratio") != 1.0:
        errors.append(f"transport is not fully finite in {path}")
    if float(diagnostics.get("minimum_plan_value", -1.0)) < 0.0:
        errors.append(f"transport plan is negative in {path}")
    if diagnostics.get("uot_plan_row_marginals_forced_equal") is not False:
        errors.append(f"balanced-marginal assertion is forbidden in {path}")
    artifact_names = result.get("artifacts", {})
    prototype_path = path.parent / str(artifact_names.get("prototypes", ""))
    incidence_path = path.parent / str(artifact_names.get("semantic_incidence", ""))
    csv_path = path.parent / str(artifact_names.get("csv", ""))
    vector_shape = tuple(int(value) for value in result.get("input_vector_shape", []))
    if len(vector_shape) != 2:
        errors.append(f"input vector shape missing in {path}")
    else:
        errors.extend(_validate_artifact_array(
            prototype_path,
            str(result.get("prototype_hash", "")),
            (int(diagnostics.get("prototype_count", 0)), vector_shape[1]),
            incidence=False,
        ))
        errors.extend(_validate_artifact_array(
            incidence_path,
            str(result.get("semantic_incidence_hash", "")),
            (vector_shape[0], int(diagnostics.get("prototype_count", 0))),
            incidence=True,
        ))
    errors.extend(_validate_csv(csv_path, recipe_id, run_manifest["run_id"]))
    flattened = {
        "run_id": run_manifest["run_id"],
        "attempt": int(result["attempt"]),
        **result["recipe"],
        "execution_status": result["execution_status"],
        "numerical_status": result["numerical_status"],
        "config_hash": result["config_hash"],
        "input_embedding_hash": result["input_embedding_hash"],
        "input_initial_prototype_hash": result["input_initial_prototype_hash"],
        "prototype_hash": result["prototype_hash"],
        "semantic_incidence_hash": result["semantic_incidence_hash"],
        **diagnostics,
    }
    return errors, flattened


def _write_report(path: Path, run_id: str, rows: list[dict], legacy: list[dict]) -> None:
    converged = sum(bool(row["transport_converged"]) for row in rows)
    nonconverged = len(rows) - converged
    report = "# Method 1 v21 UOT Full Numerical Screen\n\n"
    report += f"Run `{run_id}` mechanically merged all 27 preregistered train-only numerical recipes. "
    report += "The merge validates recipe coverage, configuration and input hashes, per-recipe artifacts, finite/nonnegative transport diagnostics, and H_sem row normalization. "
    report += "It performs no UOT recipe or model selection.\n\n"
    report += "| Measure | Value |\n| --- | ---: |\n"
    report += f"| recipes | {len(rows)} |\n"
    report += f"| primary converged (`potential_delta <= 1e-6` within 300 iterations) | {converged} |\n"
    report += f"| primary nonconverged but retained as numerical results | {nonconverged} |\n"
    report += "| calibration reads | 0 |\n| formal-test reads | 0 |\n\n"
    report += "The raw unbalanced transport plan is not required to have balanced row or column marginals. "
    report += "Normalization is checked only after conversion to the H_sem semantic incidence.\n\n"
    report += "## Legacy Exclusion\n\n"
    for item in legacy:
        report += f"- `{item['path']}`: {item['reason']}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanically merge validated v21 UOT recipe shards.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_directory = _run_directory(args.run_id)
    run_manifest_path = run_directory / "v21_uot_run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"missing v21 UOT run manifest: {run_manifest_path}")
    run_manifest = read_json(run_manifest_path)
    config = load_config(run_manifest["config_name"])
    current_config_hash = sha256_file(VERSION_ROOT / "configs" / run_manifest["config_name"])
    if current_config_hash != run_manifest["config_hash"]:
        raise RuntimeError("current UOT config hash differs from the run manifest")
    expected_recipes = build_preregistered_recipes(config)
    if run_manifest.get("recipes") != expected_recipes:
        raise RuntimeError("run manifest recipe grid differs from the fixed preregistered grid")
    errors: list[str] = []
    rows: list[dict] = []
    initial_prototype_hashes: set[str] = set()
    for recipe in expected_recipes:
        records = _completed_records(run_directory, recipe)
        if len(records) == 0:
            errors.append(f"missing completed recipe shard {recipe['recipe_id']}")
            continue
        if len(records) > 1:
            errors.append(f"duplicate completed recipe shards for {recipe['recipe_id']}: {[str(path) for path, _ in records]}")
            continue
        path, result = records[0]
        record_errors, row = _validate_record(run_manifest, recipe, path, result)
        errors.extend(record_errors)
        rows.append(row)
        initial_prototype_hashes.add(str(result.get("input_initial_prototype_hash", "")))
    observed_ids = sorted(int(row["recipe_id"]) for row in rows)
    if observed_ids != list(range(1, 28)):
        errors.append(f"merged recipe IDs are not exactly 1..27: {observed_ids}")
    if len(initial_prototype_hashes) != 1:
        errors.append("initial prototype hash differs across recipe shards")
    if errors:
        raise RuntimeError("v21 UOT shard merge validation failed:\n- " + "\n- ".join(errors))
    rows.sort(key=lambda row: int(row["recipe_id"]))
    aggregate_directory = VERSION_ROOT / "aggregate"
    csv_path = aggregate_directory / f"v21_uot_full_numerical_screen_{args.run_id}.csv"
    json_path = aggregate_directory / f"v21_uot_full_numerical_screen_{args.run_id}.json"
    shard_manifest_path = aggregate_directory / f"v21_uot_recipe_shard_manifest_{args.run_id}.json"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_UOT_FULL_NUMERICAL_SCREEN.md"
    if report_path.exists():
        report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_UOT_FULL_NUMERICAL_SCREEN_{args.run_id}.md"
    summary = {
        "schema_version": "v21-uot-full-numerical-screen-2",
        "run_id": args.run_id,
        "merged_at_utc": _utc_now(),
        "recipe_ids": list(range(1, 28)),
        "recipes": len(rows),
        "primary_convergence_rule": run_manifest["primary_convergence_rule"],
        "primary_converged": sum(bool(row["transport_converged"]) for row in rows),
        "primary_nonconverged": sum(not bool(row["transport_converged"]) for row in rows),
        "config_hash": run_manifest["config_hash"],
        "input_embedding_hash": run_manifest["input_embedding_hash"],
        "input_embedding_ids_hash": run_manifest["input_embedding_ids_hash"],
        "input_initial_prototype_hash": next(iter(initial_prototype_hashes)),
        "legacy_excluded_from_merge": run_manifest["legacy_excluded_from_merge"],
        "selection_prohibition": run_manifest["selection_prohibition"],
        "calibration_read": False,
        "formal_test_read": False,
    }
    shard_manifest = {
        "schema_version": "v21-uot-recipe-shard-manifest-2",
        "run_id": args.run_id,
        "run_manifest": str(run_manifest_path.relative_to(VERSION_ROOT)),
        "merged_csv": str(csv_path.relative_to(VERSION_ROOT)),
        "merged_json": str(json_path.relative_to(VERSION_ROOT)),
        "records": [
            {
                "recipe_id": int(row["recipe_id"]),
                "attempt": int(row["attempt"]),
                "config_hash": row["config_hash"],
                "input_embedding_hash": row["input_embedding_hash"],
                "prototype_hash": row["prototype_hash"],
                "semantic_incidence_hash": row["semantic_incidence_hash"],
            }
            for row in rows
        ],
        "validation": {
            "exact_recipe_coverage_1_to_27": True,
            "duplicate_completed_shards": 0,
            "smoke_rows": 0,
            "h_sem_normalization_checked": True,
            "unbalanced_marginals_not_forced": True,
        },
    }
    write_csv_new(str(csv_path), rows)
    write_json_new(str(json_path), summary)
    write_json_new(str(shard_manifest_path), shard_manifest)
    _write_report(report_path, args.run_id, rows, run_manifest["legacy_excluded_from_merge"])
    print(json.dumps({
        "run_id": args.run_id,
        "merged_recipes": len(rows),
        "csv": str(csv_path.relative_to(VERSION_ROOT)),
        "report": str(report_path.relative_to(VERSION_ROOT)),
        "primary_converged": summary["primary_converged"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
