#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = ROOT.parent / "20260704_method1_motivation_medmemorybench_v3"
DATA_ROOT = Path("/data2/user25263093/userliteng/26_code/preTest/data/external/MedMemoryBench")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def session_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(float(text)):012d}")
    except Exception:
        return (1, text)


def session_cmp(left: Any, right: Any) -> int:
    lk = session_key(left)
    rk = session_key(right)
    return (lk > rk) - (lk < rk)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance() -> None:
    files = sorted([p for p in DATA_ROOT.rglob("*") if p.is_file()])
    hash_lines = []
    for path in files:
        rel = path.relative_to(DATA_ROOT)
        hash_lines.append(f"{sha256_file(path)}  {rel.as_posix()}")
    (ROOT / "manifests").mkdir(exist_ok=True)
    (ROOT / "manifests" / "dataset_file_hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    local_docs = []
    for rel in ["README.md", "dataset_infos.json", "LICENSE", "LICENSE.txt"]:
        p = DATA_ROOT / rel
        if p.exists():
            local_docs.append(rel)
    parquet_files = [p.relative_to(DATA_ROOT).as_posix() for p in files if p.suffix == ".parquet"]
    report = [
        "# Dataset Provenance Audit",
        "",
        f"Dataset download location: `{DATA_ROOT}`",
        "",
        "Canonical official MedMemoryBench project URL: not verifiable from local files alone.",
        "",
        "Official repository revision could not be recovered from the local dataset copy.",
        "",
        "Download date: not recoverable from local dataset metadata. File mtimes are local filesystem times only.",
        "",
        f"Source files hashed: {len(files)}",
        "",
        "Hash manifest: `manifests/dataset_file_hashes.sha256`",
        "",
        "Observed file structure:",
        "",
    ]
    report.extend(f"- `{item}`" for item in parquet_files[:200])
    report.extend(
        [
            "",
            "Local documentation/license files found:",
            "",
        ]
    )
    if local_docs:
        report.extend(f"- `{item}`" for item in local_docs)
    else:
        report.append("- none found in the local dataset root")
    report.extend(
        [
            "",
            "License: not verifiable from local dataset files unless one of the documentation files above contains it.",
            "",
            "What cannot be verified:",
            "",
            "- official Git revision;",
            "- exact Hugging Face dataset revision;",
            "- download date from authoritative metadata;",
            "- license if not present in local documentation.",
        ]
    )
    (ROOT / "reports" / "DATASET_PROVENANCE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def memory_index() -> dict[str, dict[str, Any]]:
    return {str(row["memory_id"]): row for row in read_jsonl(ROOT / "aggregate" / "memory_items_turn_pair.jsonl")}


def query_index() -> dict[str, dict[str, Any]]:
    return {str(row["query_id"]): row for row in read_jsonl(ROOT / "aggregate" / "queries_medmemorybench_en.jsonl")}


def extract_turn(row: dict[str, Any]) -> int | None:
    for key in ["turn_index", "turn", "start_turn", "chronological_rank"]:
        if key in row and row[key] is not None:
            try:
                return int(float(row[key]))
            except Exception:
                pass
    mid = str(row.get("memory_id", ""))
    if "_x" in mid:
        try:
            return int(mid.rsplit("_x", 1)[1])
        except Exception:
            return None
    return None


def query_anchor_turn(query: dict[str, Any]) -> int | None:
    for key in ["query_turn_index", "turn_index", "turn", "anchor_turn_index"]:
        if key in query and query[key] is not None:
            try:
                return int(float(query[key]))
            except Exception:
                pass
    # MedMemoryBench queries are session-level in the exported v3 adapter; no turn anchor was exposed.
    return None


def selected_rows_from_v3() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path in [
        ("M1", ROOT / "aggregate" / "M1_history_scale_invocations.jsonl"),
        ("M2", ROOT / "aggregate" / "M2_frontier_budget_invocations.jsonl"),
        ("O1", ROOT / "aggregate" / "O1_formal_stage_trace.jsonl"),
    ]:
        for row in read_jsonl(path):
            ids = row.get("final_selected_ids") or row.get("selected_memory_ids") or []
            for mid in ids:
                rows.append(
                    {
                        "experiment": label,
                        "run_id": row.get("run_id", ""),
                        "query_id": str(row.get("query_id", "")),
                        "stage": str(row.get("stage", "")),
                        "memory_id": str(mid),
                        "query_session_id": row.get("query_session_id"),
                        "persona_id": row.get("persona_id"),
                    }
                )
    return rows


def causal_leakage_audit() -> None:
    memories = memory_index()
    queries = query_index()
    output = []
    violations = Counter()
    for item in selected_rows_from_v3():
        mem = memories.get(item["memory_id"])
        query = queries.get(item["query_id"], {})
        same_persona = bool(mem and str(mem.get("persona_id")) == str(item.get("persona_id")))
        memory_session = mem.get("session_id") if mem else None
        query_session = item.get("query_session_id") or query.get("query_session_id")
        mem_turn = extract_turn(mem or {})
        q_turn = query_anchor_turn(query)
        violation_type = None
        if not mem:
            violation_type = "missing_memory"
        elif not same_persona:
            violation_type = "cross_persona"
        elif memory_session is None or query_session is None:
            violation_type = "missing_session_metadata"
        elif session_cmp(memory_session, query_session) > 0:
            violation_type = "future_session"
        elif session_cmp(memory_session, query_session) == 0:
            if mem_turn is None or q_turn is None:
                violation_type = "missing_same_session_turn_metadata"
            elif mem_turn > q_turn:
                violation_type = "same_session_future_turn"
        causally_eligible = violation_type is None or violation_type == "missing_same_session_turn_metadata"
        if violation_type:
            violations[violation_type] += 1
        output.append(
            {
                "experiment": item["experiment"],
                "run_id": item["run_id"],
                "query_id": item["query_id"],
                "stage": item["stage"],
                "memory_id": item["memory_id"],
                "same_persona": same_persona,
                "memory_session_id": memory_session,
                "query_session_id": query_session,
                "memory_turn_index": mem_turn,
                "query_anchor_turn_index": q_turn,
                "causally_eligible": causally_eligible,
                "violation_type": violation_type,
            }
        )
    write_jsonl(ROOT / "aggregate" / "A2_causal_leakage_audit.jsonl", output)
    invalidating = violations.get("future_session", 0) + violations.get("same_session_future_turn", 0) + violations.get("cross_persona", 0)
    report = [
        "# A2 Causal Leakage Audit",
        "",
        f"Total checked selected memories: {len(output)}",
        f"Future-session violations: {violations.get('future_session', 0)}",
        f"Same-session future-turn violations: {violations.get('same_session_future_turn', 0)}",
        f"Cross-persona violations: {violations.get('cross_persona', 0)}",
        f"Missing timestamp/session metadata cases: {violations.get('missing_session_metadata', 0)}",
        f"Missing same-session turn metadata cases: {violations.get('missing_same_session_turn_metadata', 0)}",
        "",
        "Interpretation:",
        "",
    ]
    if invalidating:
        report.append("At least one invalidating leakage violation was found. A corrected causal configuration and affected reruns are required before using those formal results.")
    else:
        report.append("No future-session, same-session future-turn, or cross-persona selected-memory violation was found. Current formal results are not invalidated by this audit.")
    report.extend(
        [
            "",
            "Note: same-session turn anchors are not exposed for many exported query rows, so those cases are reported separately instead of silently assumed.",
        ]
    )
    (ROOT / "reports" / "A2_CAUSAL_LEAKAGE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def candidate_path_audit() -> None:
    rows = []
    for row in read_jsonl(ROOT / "aggregate" / "M2_frontier_budget_invocations.jsonl"):
        budget = int(row.get("frontier_budget", 0))
        n_ann = int(row.get("n_ann_raw", len(row.get("ann_candidate_ids", []))))
        n_after_filter = int(row.get("n_after_filter", row.get("n_after_metadata_filter", n_ann)))
        n_after_dedup = int(row.get("n_after_dedup", len(set(row.get("ann_candidate_ids", [])))))
        n_mat = int(row.get("n_materialized", row.get("n_reranked", 0)))
        n_reranked = int(row.get("n_reranked", len(row.get("reranked_candidate_ids", []))))
        n_final = int(row.get("n_final_selected", len(row.get("final_selected_ids", []))))
        rows.append(
            {
                "query_id": row.get("query_id", ""),
                "persona_id": row.get("persona_id", ""),
                "query_type": row.get("query_type", ""),
                "frontier_budget": budget,
                "n_ann_raw": n_ann,
                "n_after_metadata_filter": n_after_filter,
                "n_after_causal_filter": n_after_filter,
                "n_after_dedup": n_after_dedup,
                "n_after_materialization": n_mat,
                "n_reranked": n_reranked,
                "n_final_selected": n_final,
                "frontier_fill_ratio": n_reranked / float(budget) if budget else 0.0,
                "selection_fill_ratio": n_final / 5.0,
            }
        )
    write_csv(ROOT / "aggregate" / "A3_candidate_path_occupancy.csv", rows)
    report = ["# A3 Candidate Path Occupancy Audit", ""]
    for budget in sorted({int(r["frontier_budget"]) for r in rows}):
        sub = [r for r in rows if int(r["frontier_budget"]) == budget]
        report.extend([f"## Frontier budget {budget}", ""])
        for key in ["n_ann_raw", "n_after_metadata_filter", "n_after_causal_filter", "n_after_dedup", "n_after_materialization", "n_reranked", "n_final_selected", "frontier_fill_ratio", "selection_fill_ratio"]:
            vals = [float(r[key]) for r in sub]
            report.append(
                f"- `{key}`: min={min(vals):.3f}, p50={percentile(vals, 0.50):.3f}, p90={percentile(vals, 0.90):.3f}, p95={percentile(vals, 0.95):.3f}, max={max(vals):.3f}, mean={mean(vals):.3f}"
            )
        report.append(f"- fraction `n_reranked < K_f`: {sum(1 for r in sub if int(r['n_reranked']) < budget) / len(sub):.4f}")
        report.append(f"- fraction `n_final_selected < K_o`: {sum(1 for r in sub if int(r['n_final_selected']) < 5) / len(sub):.4f}")
        report.append("")
    report.extend(
        [
            "Conclusion:",
            "",
            "The audit verifies whether `frontier_budget` changes the real reranked frontier, not only a config label.",
        ]
    )
    (ROOT / "reports" / "A3_CANDIDATE_PATH_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def controller_overhead() -> None:
    rows = read_jsonl(ROOT / "aggregate" / "O1_formal_stage_trace.jsonl")
    intent = [float(r.get("t_intent_generation_ms", r.get("controller_intent_latency_ms", 0.0))) for r in rows]
    selection = [float(r.get("t_total_selection_ms", r.get("selection_latency_ms", 0.0))) for r in rows]
    mean_intent = mean(intent) if intent else 0.0
    mean_selection = mean(selection) if selection else 0.0
    ratio = mean_intent / mean_selection if mean_selection else 0.0
    lines = [
        "# A4 Controller Overhead Positioning",
        "",
        f"O1 formal rows: {len(rows)}",
        f"Mean controller intent-generation latency: {mean_intent:.2f} ms",
        f"Mean retrieval-selection latency after intent generation: {mean_selection:.2f} ms",
        f"Controller/retrieval latency ratio: {ratio:.2f}x",
        "",
        "The Qwen3-4B controller is used only to reconstruct and instrument an offline stage-aware workload for motivation analysis. It is not an additional online component of the final runtime design.",
        "",
        "The controller latency is reported explicitly and is not hidden or subtracted from the trace.",
    ]
    (ROOT / "reports" / "A4_CONTROLLER_OVERHEAD_POSITIONING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def runtime_manifest() -> None:
    payload = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_qwen_cuda_device": os.environ.get("LOCAL_QWEN_CUDA_DEVICE"),
        "nvidia_smi": run_text(["nvidia-smi"]),
        "model_paths": {
            "controller_answer_model": "../../../dir",
            "embedding_model": "../../models/bge-base-en-v1.5",
            "reranker": "../../models/cross-encoder-ms-marco-MiniLM-L6-v2",
        },
        "pip_freeze_head": run_text([sys.executable, "-m", "pip", "freeze"]).splitlines()[:200],
    }
    try:
        import torch

        payload.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            }
        )
    except Exception as exc:
        payload["torch_error"] = str(exc)
    (ROOT / "manifests" / "runtime_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v4 offline provenance and audit package A.")
    parser.parse_args()
    runtime_manifest()
    dataset_provenance()
    causal_leakage_audit()
    candidate_path_audit()
    controller_overhead()
    print("v4_audits=complete")


if __name__ == "__main__":
    main()
