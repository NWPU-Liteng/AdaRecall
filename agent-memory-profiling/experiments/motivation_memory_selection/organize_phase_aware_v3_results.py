#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from metrics import build_summary
from plot_results import make_plots


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CONFIG_ROOT = PROJECT_ROOT / "experiments" / "motivation_memory_selection" / "configs"
VERSIONS_ROOT = PROJECT_ROOT / "versions"
VERSION_TAG = "20260616_phase_aware_real_agent_v3_formal"


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    rerank: str
    output_dir: str
    config_name: str


FORMAL_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        dataset="locomo",
        rerank="heuristic",
        output_dir="motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615",
        config_name="phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615.yaml",
    ),
    RunSpec(
        dataset="locomo",
        rerank="cross_encoder",
        output_dir="motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615",
        config_name="phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615.yaml",
    ),
    RunSpec(
        dataset="longmemeval",
        rerank="heuristic",
        output_dir="motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_heuristic_rerank_20260616",
        config_name="phase_aware_real_agent_v3_longmemeval_full_heuristic_rerank_20260616.yaml",
    ),
    RunSpec(
        dataset="longmemeval",
        rerank="cross_encoder",
        output_dir="motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616",
        config_name="phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616.yaml",
    ),
    RunSpec(
        dataset="msc",
        rerank="heuristic",
        output_dir="motivation_memory_selection_phase_aware_real_agent_v3_msc_full_heuristic_rerank_20260616",
        config_name="phase_aware_real_agent_v3_msc_full_heuristic_rerank_20260616.yaml",
    ),
    RunSpec(
        dataset="msc",
        rerank="cross_encoder",
        output_dir="motivation_memory_selection_phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616",
        config_name="phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616.yaml",
    ),
)


ARCHIVE_DIRS: tuple[str, ...] = (
    "motivation_memory_selection",
    "motivation_memory_selection_locomo_cross_encoder_agent_medium_20260615",
    "motivation_memory_selection_locomo_cross_encoder_agent_quick_20260615",
    "motivation_memory_selection_locomo_cross_encoder_medium_20260615",
    "motivation_memory_selection_locomo_cross_encoder_quick_20260615",
    "motivation_memory_selection_locomo_medium_20260614",
    "motivation_memory_selection_locomo_quick_20260614",
    "motivation_memory_selection_longmemeval_medium_20260614",
    "motivation_memory_selection_medium_20260614",
    "motivation_memory_selection_msc_medium_20260614",
    "motivation_memory_selection_msc_quick_20260614",
    "motivation_memory_selection_phase_aware_method_validation_locomo_20260615",
    "motivation_memory_selection_phase_aware_method_validation_locomo_20260615_replay",
    "motivation_memory_selection_phase_aware_real_agent_v2_locomo_validation_20260615",
    "motivation_memory_selection_phase_aware_real_agent_v3_locomo_validation_20260615",
    "agent_latency_suite_20260615",
)


def _dedupe_jsonl(path: Path, key_fields: Iterable[str]) -> int:
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        seen[key] = row
    deduped = list(seen.values())
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in deduped), encoding="utf-8")
    return len(rows) - len(deduped)


def _ensure_summary_and_plots(run_dir: Path) -> pd.DataFrame:
    raw_path = run_dir / "raw_profile.jsonl"
    summary_csv = run_dir / "summary.csv"
    summary_md = run_dir / "summary.md"
    figures_dir = run_dir / "figures"

    df = pd.read_json(raw_path, lines=True)
    summary = build_summary(df)
    summary.to_csv(summary_csv, index=False)

    lines = ["# Motivation Memory Selection Summary", ""]
    if summary.empty:
        lines.append("No clean rows were available.")
    else:
        try:
            lines.append(summary.to_markdown(index=False))
        except ImportError:
            lines.append("`tabulate` is unavailable; falling back to plain-text table output.")
            lines.append("")
            lines.append("```text")
            lines.append(summary.to_string(index=False))
            lines.append("```")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_plots(raw_path, figures_dir)
    return summary


def _resolve_run_dir(spec: RunSpec, version_dir: Path) -> Path:
    root_run_dir = OUTPUTS_ROOT / spec.output_dir
    versioned_run_dir = version_dir / "runs" / spec.output_dir
    if root_run_dir.exists():
        return root_run_dir
    if versioned_run_dir.exists():
        return versioned_run_dir
    raise FileNotFoundError(f"Could not locate run directory for {spec.output_dir}")


def _fmt_ms(value: float) -> str:
    return f"{value:.1f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _build_report(version_dir: Path) -> None:
    rows = []
    baseline_rows = []
    for spec in FORMAL_RUNS:
        run_dir = _resolve_run_dir(spec, version_dir)
        raw_path = run_dir / "raw_profile.jsonl"
        phase_path = run_dir / "phase_trace.jsonl"
        config_path = CONFIG_ROOT / spec.config_name
        config_snapshot_path = version_dir / "configs_snapshot" / spec.config_name
        raw_df = pd.read_json(raw_path, lines=True)
        clean = raw_df[raw_df["error"].fillna("") == ""].copy()
        summary = build_summary(raw_df)
        rows.append(
            {
                "dataset": spec.dataset,
                "rerank": spec.rerank,
                "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                "config_snapshot_path": str(config_snapshot_path.relative_to(PROJECT_ROOT)),
                "output_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "raw_rows": len(raw_df),
                "phase_rows": sum(1 for _ in phase_path.open("r", encoding="utf-8")) if phase_path.exists() else 0,
                "query_types": ", ".join(sorted(clean["query_type"].astype(str).unique())),
                "phase_selection_ms": clean["phase_selection_total_latency_ms"].mean(),
                "phase_retrieval_ms": clean["phase_retrieval_total_latency_ms"].mean(),
                "phase_rerank_ms": clean["phase_rerank_total_latency_ms"].mean(),
                "agent_ms": clean["agent_execution_total_latency_ms"].mean(),
                "total_ms": clean["total_response_latency_ms"].mean(),
                "selection_share": clean["phase_selection_to_total_response_ratio"].mean(),
                "retrieval_share": clean["phase_retrieval_to_total_response_ratio"].mean(),
                "rerank_share": clean["phase_rerank_to_total_response_ratio"].mean(),
                "selection_invocations": clean["selection_invocation_count"].mean(),
                "before_rerank": clean["num_before_rerank"].mean(),
            }
        )
        for baseline in ["full_selection", "oracle_pruned_rerank"]:
            subset = clean[clean["baseline_mode"] == baseline].copy()
            baseline_rows.append(
                {
                    "dataset": spec.dataset,
                    "rerank": spec.rerank,
                    "baseline": baseline,
                    "total_ms": subset["total_response_latency_ms"].mean(),
                    "phase_selection_ms": subset["phase_selection_total_latency_ms"].mean(),
                    "phase_rerank_ms": subset["phase_rerank_total_latency_ms"].mean(),
                    "agent_ms": subset["agent_execution_total_latency_ms"].mean(),
                    "selection_share": subset["phase_selection_to_total_response_ratio"].mean(),
                    "rerank_share": subset["phase_rerank_to_total_response_ratio"].mean(),
                    "before_rerank": subset["num_before_rerank"].mean(),
                }
            )

    report_path = version_dir / "phase_aware_real_agent_v3_experiment_report_20260616.md"
    lines = [
        "# Phase-Aware Real-Agent V3 Experiment Report",
        "",
        "## Scope",
        "",
        "- Agent memory mode: `phase_aware`",
        "- Agent execution backend: local Qwen (`generation_backend=local_qwen`, `agent_execution_backend=llm`)",
        "- Embedding backend: BGE transformer (`models/bge-base-en-v1.5`)",
        "- Compared rerank backends: `heuristic` and `cross_encoder`",
        "- Datasets: `LoCoMo`, `LongMemEval`, `MSC`",
        "- Compared baselines inside each run: `full_selection` and `oracle_pruned_rerank`",
        "",
        "## Formal Run Map",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['dataset']}/{row['rerank']}`: active config `{row['config_path']}`, config snapshot `{row['config_snapshot_path']}`, outputs `{row['output_dir']}`"
        )
    lines.extend(
        [
            "",
            "## Overall Summary",
            "",
            "| Dataset | Rerank | Raw Rows | Phase Rows | Query Types | Phase Selection ms | Retrieval-like ms | Rerank ms | Agent ms | Total Response ms | Selection Share | Retrieval Share | Rerank Share | Selection Calls | Avg Before Rerank |",
            "|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|:--|:--|--:|--:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['rerank']} | {row['raw_rows']} | {row['phase_rows']} | {row['query_types']} | "
            f"{_fmt_ms(row['phase_selection_ms'])} | {_fmt_ms(row['phase_retrieval_ms'])} | {_fmt_ms(row['phase_rerank_ms'])} | "
            f"{_fmt_ms(row['agent_ms'])} | {_fmt_ms(row['total_ms'])} | {_fmt_pct(row['selection_share'])} | "
            f"{_fmt_pct(row['retrieval_share'])} | {_fmt_pct(row['rerank_share'])} | {row['selection_invocations']:.1f} | {row['before_rerank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Breakdown",
            "",
            "| Dataset | Rerank | Baseline | Total ms | Phase Selection ms | Phase Rerank ms | Agent ms | Selection Share | Rerank Share | Avg Before Rerank |",
            "|:--|:--|:--|--:|--:|--:|--:|:--|:--|--:|",
        ]
    )
    for row in baseline_rows:
        lines.append(
            f"| {row['dataset']} | {row['rerank']} | {row['baseline']} | {_fmt_ms(row['total_ms'])} | "
            f"{_fmt_ms(row['phase_selection_ms'])} | {_fmt_ms(row['phase_rerank_ms'])} | {_fmt_ms(row['agent_ms'])} | "
            f"{_fmt_pct(row['selection_share'])} | {_fmt_pct(row['rerank_share'])} | {row['before_rerank']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            "- Under `heuristic` rerank, phase-aware memory selection remains light on `LoCoMo` and `LongMemEval`, and becomes moderately visible on `MSC` because retrieval/filter/aggregation scales with the larger memory pool.",
            "- Under `cross_encoder` rerank, phase-aware memory selection becomes a first-order cost on all three datasets, with rerank dominating the selection path.",
            "- `oracle_pruned_rerank` consistently reduces `num_before_rerank`, and this reduction materially lowers total response time only when the rerank backend is heavy enough for rerank to matter.",
            "- The cleanest paper-facing evidence for `selection/rerank` as the primary optimization target comes from the `cross_encoder` runs, not the `heuristic` runs.",
            "",
            "## Versioning",
            "",
            f"- This report is stored under `{version_dir.relative_to(PROJECT_ROOT)}`.",
            f"- Formal run directories are versioned under `{(version_dir / 'runs').relative_to(PROJECT_ROOT)}`.",
            f"- Formal config snapshots are stored under `{(version_dir / 'configs_snapshot').relative_to(PROJECT_ROOT)}`.",
            "- Older, non-formal, mock-based, quick, medium, replay, and pre-v3 validation outputs were moved to the archive subtree to reduce root-level clutter.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    archive_root = OUTPUTS_ROOT / "archive" / "20260616_phase_aware_cleanup"
    historical_runs_dir = archive_root / "historical_runs"
    historical_configs_dir = archive_root / "historical_configs"
    version_dir = VERSIONS_ROOT / VERSION_TAG
    version_runs_dir = version_dir / "runs"
    version_configs_dir = version_dir / "configs_snapshot"

    historical_runs_dir.mkdir(parents=True, exist_ok=True)
    historical_configs_dir.mkdir(parents=True, exist_ok=True)
    version_dir.mkdir(parents=True, exist_ok=True)
    version_runs_dir.mkdir(parents=True, exist_ok=True)
    version_configs_dir.mkdir(parents=True, exist_ok=True)

    raw_key_fields = (
        "query_id",
        "baseline_mode",
        "retrieval_topk",
        "relation_expansion_depth",
        "rerank_topk_input",
        "memory_size",
    )
    phase_key_fields = (
        "query_id",
        "baseline_mode",
        "retrieval_topk",
        "relation_expansion_depth",
        "rerank_topk_input",
        "memory_size",
        "phase_name",
    )

    for spec in FORMAL_RUNS:
        run_dir = _resolve_run_dir(spec, version_dir)
        _dedupe_jsonl(run_dir / "raw_profile.jsonl", raw_key_fields)
        _dedupe_jsonl(run_dir / "phase_trace.jsonl", phase_key_fields)
        _ensure_summary_and_plots(run_dir)

    for spec in FORMAL_RUNS:
        src = OUTPUTS_ROOT / spec.output_dir
        dst = version_runs_dir / spec.output_dir
        if src.exists():
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))
        shutil.copy2(CONFIG_ROOT / spec.config_name, version_configs_dir / spec.config_name)

    for name in ARCHIVE_DIRS:
        src = OUTPUTS_ROOT / name
        if src.exists():
            dst = historical_runs_dir / name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))

    active_configs = {spec.config_name for spec in FORMAL_RUNS}
    for path in CONFIG_ROOT.iterdir():
        if path.name in active_configs:
            continue
        dst = historical_configs_dir / path.name
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(path), str(dst))

    _build_report(version_dir)


if __name__ == "__main__":
    main()
