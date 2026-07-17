#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-agent-memory-motivation")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/agent-memory-motivation-cache")


VERSION_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_ROOT = VERSION_ROOT.parent
PROJECT_ROOT = VERSIONS_ROOT.parent

RAW_DIR = VERSION_ROOT / "raw"
AGGREGATE_DIR = VERSION_ROOT / "aggregate"
FIGURES_DIR = VERSION_ROOT / "figures"
MANIFESTS_DIR = VERSION_ROOT / "manifests"
CONFIG_DIR = VERSION_ROOT / "config"
SCRIPTS_DIR = VERSION_ROOT / "scripts"

V3_ROOT = PROJECT_ROOT / "versions" / "20260616_phase_aware_real_agent_v3_formal"
V4_ROOT = PROJECT_ROOT / "versions" / "20260621_real_trace_adaptive_phase_v4_formal"

V3_CROSS_ENCODER_RUNS = {
    "locomo": "motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615",
    "longmemeval": "motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616",
    "msc": "motivation_memory_selection_phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616",
}

V4_ADAPTIVE_RUNS = {
    "locomo": "locomo_official_qa_full_adaptive_phase_controller_lite_selection_only",
    "longmemeval": "longmemeval_s_official_qa_full_adaptive_phase_controller_lite_selection_only",
    "msc": "msc_replay_turn_full_adaptive_phase_controller_lite_selection_only",
}

DATASET_SPECS = {
    "locomo": {
        "memory_path": PROJECT_ROOT / "data" / "motivation_memory_selection" / "locomo_memories_1000.jsonl",
        "runtime_manifest": V4_ROOT / "manifests" / "locomo" / "runtime_manifest.jsonl",
        "runtime_memory_bank": V4_ROOT / "manifests" / "locomo" / "runtime_memory_bank.jsonl",
        "memory_size": 5882,
    },
    "longmemeval": {
        "memory_path": PROJECT_ROOT / "data" / "motivation_memory_selection" / "longmemeval_memories_1000.jsonl",
        "runtime_manifest": V4_ROOT / "manifests" / "longmemeval_s" / "runtime_manifest.jsonl",
        "runtime_memory_bank": V4_ROOT / "manifests" / "longmemeval_s" / "runtime_memory_bank.jsonl",
        "memory_size": 23867,
    },
    "msc": {
        "memory_path": PROJECT_ROOT / "data" / "motivation_memory_selection" / "msc_memories_5000.jsonl",
        "runtime_manifest": V4_ROOT / "manifests" / "msc" / "runtime_manifest.jsonl",
        "runtime_memory_bank": V4_ROOT / "manifests" / "msc" / "runtime_memory_bank.jsonl",
        "memory_size": 395934,
    },
}

QUERY_STRATUM_MAP = {
    "fact": "single_evidence",
    "temporal": "long_range",
    "relation": "multi_evidence",
    "plan_continuity": "multi_evidence",
    "constraint": "multi_evidence",
}


def _ensure_dirs() -> None:
    for path in [RAW_DIR, AGGREGATE_DIR, FIGURES_DIR, MANIFESTS_DIR, CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def _trace_key(row: dict) -> tuple:
    return (
        row["query_id"],
        row["baseline_mode"],
        int(row["memory_size"]),
        int(row["retrieval_topk"]),
        int(row["relation_expansion_depth"]),
        str(row["rerank_topk_input"]),
    )


def _copy_and_enrich_sources() -> tuple[Path, Path, Path]:
    m1_rows: list[dict] = []
    o1_rows: list[dict] = []
    v4_raw_rows: list[dict] = []
    source_runs: list[dict] = []

    for dataset, run_name in V3_CROSS_ENCODER_RUNS.items():
        run_dir = V3_ROOT / "runs" / run_name
        raw_path = run_dir / "raw_profile.jsonl"
        phase_path = run_dir / "phase_trace.jsonl"
        config_path = V3_ROOT / "configs_snapshot" / f"{run_name.replace('motivation_memory_selection_', '')}.yaml"
        raw_rows = _read_jsonl(raw_path)
        phase_rows = _read_jsonl(phase_path)
        gold_groups_by_key = {_trace_key(row): list(row.get("gold_group_ids", [])) for row in phase_rows}
        for row in raw_rows:
            enriched = dict(row)
            enriched["dataset"] = dataset
            enriched["source_run"] = run_name
            enriched["source_version"] = V3_ROOT.name
            enriched["source_path"] = str(raw_path)
            enriched["frontier_budget"] = int(row["retrieval_topk"])
            enriched["final_evidence_k"] = int(len(row.get("selected_memory_ids", [])))
            enriched["query_stratum"] = QUERY_STRATUM_MAP.get(str(row.get("query_type", "")), "other")
            enriched["gold_group_ids"] = gold_groups_by_key.get(_trace_key(row), [])
            enriched["t_ann_ms"] = float(row.get("coarse_retrieval_latency_ms", 0.0))
            enriched["t_candidate_materialize_ms"] = float(
                row.get("metadata_filter_latency_ms", 0.0)
                + row.get("relation_expansion_latency_ms", 0.0)
                + row.get("candidate_aggregation_latency_ms", 0.0)
                + row.get("context_construction_latency_ms", 0.0)
            )
            enriched["t_rerank_ms"] = float(row.get("rerank_latency_ms", 0.0))
            enriched["t_answer_ms"] = float(row.get("generation_latency_ms", 0.0))
            enriched["t_total_ms"] = float(row.get("total_response_latency_ms", 0.0))
            enriched["host_memory_bytes"] = int(row.get("estimated_host_memory_traffic_bytes", 0))
            enriched["candidate_payload_bytes"] = int(row.get("pim_optimization_potential_bytes", 0))
            enriched["candidate_metadata_bytes"] = int(
                max(
                    int(row.get("estimated_host_memory_traffic_bytes", 0))
                    - int(row.get("pim_optimization_potential_bytes", 0)),
                    0,
                )
            )
            enriched["gpu_name"] = "unknown_historical_v3"
            enriched["seed"] = 42
            enriched["git_commit"] = "unavailable"
            m1_rows.append(enriched)
        source_runs.append(
            {
                "dataset": dataset,
                "source_version": V3_ROOT.name,
                "run_name": run_name,
                "config_path": str(config_path),
                "raw_profile_path": str(raw_path),
                "phase_trace_path": str(phase_path),
                "notes": "historical real-GPU cross-encoder run reused for M1/M2/M3",
            }
        )

    for dataset, run_name in V4_ADAPTIVE_RUNS.items():
        run_dir = V4_ROOT / "runs" / run_name
        phase_path = run_dir / "phase_trace.jsonl"
        raw_path = run_dir / "raw_profile.jsonl"
        manifest_path = run_dir / "run_manifest.json"
        phase_rows = _read_jsonl(phase_path)
        for sequence_index, row in enumerate(phase_rows):
            enriched = dict(row)
            enriched["dataset"] = dataset
            enriched["source_run"] = run_name
            enriched["source_version"] = V4_ROOT.name
            enriched["source_path"] = str(phase_path)
            enriched["sequence_index"] = sequence_index
            o1_rows.append(enriched)
        raw_rows = _read_jsonl(raw_path)
        for row in raw_rows:
            enriched = dict(row)
            enriched["dataset"] = dataset
            enriched["source_run"] = run_name
            enriched["source_version"] = V4_ROOT.name
            enriched["source_path"] = str(raw_path)
            v4_raw_rows.append(enriched)
        source_runs.append(
            {
                "dataset": dataset,
                "source_version": V4_ROOT.name,
                "run_name": run_name,
                "run_manifest_path": str(manifest_path),
                "raw_profile_path": str(raw_path),
                "phase_trace_path": str(phase_path),
                "runtime_manifest_path": str(DATASET_SPECS[dataset]["runtime_manifest"]),
                "runtime_memory_bank_path": str(DATASET_SPECS[dataset]["runtime_memory_bank"]),
                "notes": "historical real-trace adaptive-phase run reused for O1",
            }
        )

    m1_path = RAW_DIR / "M1_profile_cross_encoder_v3.jsonl"
    o1_path = RAW_DIR / "O1_phase_trace_v4_adaptive.jsonl"
    v4_raw_path = RAW_DIR / "O1_raw_profile_v4_adaptive.jsonl"
    _write_jsonl(m1_path, m1_rows)
    _write_jsonl(o1_path, o1_rows)
    _write_jsonl(v4_raw_path, v4_raw_rows)
    _write_json(MANIFESTS_DIR / "source_runs.json", source_runs)
    return m1_path, o1_path, v4_raw_path


def _load_memory_group_maps() -> dict[str, dict[str, str]]:
    group_maps: dict[str, dict[str, str]] = {}
    for dataset, spec in DATASET_SPECS.items():
        mapping: dict[str, str] = {}
        for row in _read_jsonl(spec["memory_path"]):
            mapping[str(row["memory_id"])] = str(row.get("gold_group_id", "group_default"))
        group_maps[dataset] = mapping
    return group_maps


def _build_manifests(m1_path: Path, o1_path: Path) -> None:
    dataset_rows = []
    for dataset, spec in DATASET_SPECS.items():
        runtime_rows = _read_jsonl(spec["runtime_manifest"])
        dataset_rows.append(
            {
                "dataset": dataset,
                "split": "historical_formal",
                "num_queries": len(runtime_rows),
                "num_agents": len({row.get("tenant_id", "") for row in runtime_rows}),
                "num_episodes": len({row.get("timeline_id", row.get("tenant_id", "")) for row in runtime_rows}),
                "memory_item_definition": "JSONL row with stable memory_id and per-item text payload",
                "max_memory_scale": spec["memory_size"],
                "has_original_task_ground_truth": dataset != "msc",
                "has_trace_or_phase_source": True,
                "memory_path": str(spec["memory_path"]),
                "runtime_manifest_path": str(spec["runtime_manifest"]),
                "runtime_memory_bank_path": str(spec["runtime_memory_bank"]),
            }
        )

    run_manifest = {
        "version_root": str(VERSION_ROOT),
        "historical_inputs": {
            "m1_profile": str(m1_path),
            "o1_phase_trace": str(o1_path),
        },
        "git_commit": "unavailable_empty_dot_git_directory",
        "gpu_check_reconfirmed_on_2026_07_03": True,
        "notes": [
            "This package consolidates historical real-GPU runs instead of re-running the full matrix.",
            "The workspace contains an empty .git directory, so historical commit metadata cannot be reconstructed.",
        ],
    }

    _write_json(MANIFESTS_DIR / "dataset_manifest.json", dataset_rows)
    _write_json(MANIFESTS_DIR / "run_manifest.json", run_manifest)


def _compute_m1_m2(m1_path: Path) -> tuple[Path, Path]:
    df = pd.read_json(m1_path, lines=True)
    clean = df[df["error"].fillna("") == ""].copy()
    if clean.empty:
        raise ValueError("No clean M1 rows available.")

    group_maps = _load_memory_group_maps()

    def task_proxy(row: pd.Series) -> float:
        gold_groups = set(row.get("gold_group_ids") or [])
        if not gold_groups:
            return float("nan")
        selected = row.get("selected_memory_ids") or []
        selected_groups = {group_maps[row["dataset"]].get(memory_id, "") for memory_id in selected}
        return 1.0 if selected_groups & gold_groups else 0.0

    clean["task_proxy_group_hit"] = clean.apply(task_proxy, axis=1)
    clean["selection_downstream_ms"] = (
        clean["t_candidate_materialize_ms"].astype(float) + clean["t_rerank_ms"].astype(float)
    )
    clean["query_stratum"] = clean["query_stratum"].fillna("other")

    high_budget = (
        clean[clean["frontier_budget"] == clean.groupby(["dataset", "query_id"])["frontier_budget"].transform("max")]
        .copy()
        .rename(columns={"selected_memory_ids": "selected_memory_ids_high_budget"})
    )
    stability = clean.merge(
        high_budget[
            [
                "dataset",
                "query_id",
                "query_type",
                "baseline_mode",
                "relation_expansion_depth",
                "selected_memory_ids_high_budget",
            ]
        ],
        on=["dataset", "query_id", "query_type", "baseline_mode", "relation_expansion_depth"],
        how="left",
    )

    def stability_score(row: pd.Series) -> float:
        selected = list(row.get("selected_memory_ids") or [])
        reference = list(row.get("selected_memory_ids_high_budget") or [])
        k = max(len(selected), len(reference), 1)
        return len(set(selected) & set(reference)) / float(k)

    stability["high_budget_reranker_reference_stability"] = stability.apply(stability_score, axis=1)

    m1_summary = (
        clean.groupby(["dataset", "memory_size", "frontier_budget"], dropna=False)
        .agg(
            n_queries=("query_id", "count"),
            n_failures=("error", lambda s: int((s.fillna("") != "").sum())),
            t_ann_mean_ms=("t_ann_ms", "mean"),
            t_ann_median_ms=("t_ann_ms", "median"),
            t_ann_p95_ms=("t_ann_ms", lambda s: _safe_quantile(s, 0.95)),
            t_ann_p99_ms=("t_ann_ms", lambda s: _safe_quantile(s, 0.99)),
            t_candidate_materialize_mean_ms=("t_candidate_materialize_ms", "mean"),
            t_candidate_materialize_p95_ms=("t_candidate_materialize_ms", lambda s: _safe_quantile(s, 0.95)),
            t_rerank_mean_ms=("t_rerank_ms", "mean"),
            t_rerank_p95_ms=("t_rerank_ms", lambda s: _safe_quantile(s, 0.95)),
            t_answer_mean_ms=("t_answer_ms", "mean"),
            t_total_mean_ms=("t_total_ms", "mean"),
            t_total_median_ms=("t_total_ms", "median"),
            t_total_p95_ms=("t_total_ms", lambda s: _safe_quantile(s, 0.95)),
            t_total_p99_ms=("t_total_ms", lambda s: _safe_quantile(s, 0.99)),
            mean_candidate_payload_bytes=("candidate_payload_bytes", "mean"),
            mean_candidate_metadata_bytes=("candidate_metadata_bytes", "mean"),
            mean_host_memory_bytes=("host_memory_bytes", "mean"),
            mean_num_before_rerank=("num_before_rerank", "mean"),
        )
        .reset_index()
        .sort_values(["memory_size", "frontier_budget", "dataset"])
    )
    m1_csv = AGGREGATE_DIR / "M1_stage_latency.csv"
    m1_summary.to_csv(m1_csv, index=False)

    m2_summary = (
        stability[stability["baseline_mode"] == "full_selection"]
        .groupby(["dataset", "query_stratum", "frontier_budget"], dropna=False)
        .agg(
            n_queries=("query_id", "count"),
            mean_downstream_selection_ms=("selection_downstream_ms", "mean"),
            mean_task_proxy_group_hit=("task_proxy_group_hit", "mean"),
            mean_high_budget_reference_stability=("high_budget_reranker_reference_stability", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "query_stratum", "frontier_budget"])
    )
    m2_csv = AGGREGATE_DIR / "M2_pareto.csv"
    m2_summary.to_csv(m2_csv, index=False)
    return m1_csv, m2_csv


def _plot_m1_m2(m1_csv: Path, m2_csv: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    m1_df = pd.read_csv(m1_csv)
    m2_df = pd.read_csv(m2_csv)

    budget_max = int(m1_df["frontier_budget"].max())
    focus = m1_df[m1_df["frontier_budget"] == budget_max].sort_values("memory_size")
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    bottoms = [0.0] * len(focus)
    for column, label in [
        ("t_ann_mean_ms", "ANN"),
        ("t_candidate_materialize_mean_ms", "Candidate materialize"),
        ("t_rerank_mean_ms", "Rerank"),
        ("t_answer_mean_ms", "Answer"),
    ]:
        values = list(focus[column])
        ax1.bar(focus["dataset"], values, bottom=bottoms, label=label)
        bottoms = [left + right for left, right in zip(bottoms, values)]
    ax1.set_title(f"M1 Latency Breakdown at Frontier Budget {budget_max}")
    ax1.set_ylabel("Mean latency (ms)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "fig_m1_latency_breakdown.png", dpi=180)
    fig1.savefig(FIGURES_DIR / "fig_m1_latency_breakdown.pdf")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for dataset in sorted(m1_df["dataset"].unique()):
        subset = m1_df[m1_df["dataset"] == dataset].sort_values("frontier_budget")
        ax2.plot(
            subset["frontier_budget"],
            subset["t_candidate_materialize_mean_ms"] + subset["t_rerank_mean_ms"],
            marker="o",
            label=dataset,
        )
    ax2.set_title("M1 Budget vs Downstream Cost")
    ax2.set_xlabel("Frontier budget")
    ax2.set_ylabel("Candidate materialize + rerank (ms)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "fig_m1_budget_vs_downstream_cost.png", dpi=180)
    fig2.savefig(FIGURES_DIR / "fig_m1_budget_vs_downstream_cost.pdf")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for stratum in sorted(m2_df["query_stratum"].unique()):
        subset = (
            m2_df[m2_df["query_stratum"] == stratum]
            .groupby("frontier_budget", as_index=False)[["mean_downstream_selection_ms", "mean_task_proxy_group_hit"]]
            .mean()
            .sort_values("frontier_budget")
        )
        ax3.plot(
            subset["mean_downstream_selection_ms"],
            subset["mean_task_proxy_group_hit"],
            marker="o",
            label=stratum,
        )
        for _, row in subset.iterrows():
            ax3.annotate(str(int(row["frontier_budget"])), (row["mean_downstream_selection_ms"], row["mean_task_proxy_group_hit"]))
    ax3.set_title("M2 Quality-Cost Pareto")
    ax3.set_xlabel("Downstream selection cost (ms)")
    ax3.set_ylabel("Task proxy group-hit")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / "fig_m2_quality_cost_pareto.png", dpi=180)
    fig3.savefig(FIGURES_DIR / "fig_m2_quality_cost_pareto.pdf")
    plt.close(fig3)


def _run_subprocess(script_name: str, *args: str) -> None:
    cmd = ["python", str(SCRIPTS_DIR / script_name), *args]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _write_results_and_limitations() -> None:
    m1_df = pd.read_csv(AGGREGATE_DIR / "M1_stage_latency.csv")
    m2_df = pd.read_csv(AGGREGATE_DIR / "M2_pareto.csv")
    m3_df = pd.read_csv(AGGREGATE_DIR / "M3_tile_scattering.csv")
    o1_df = pd.read_csv(AGGREGATE_DIR / "O1_phase_locality.csv")

    largest_scale_row = m1_df.sort_values(["memory_size", "frontier_budget"]).iloc[-1]
    m2_full = m2_df[m2_df["query_stratum"] == "multi_evidence"].sort_values("frontier_budget")
    m2_observation = "insufficient multi_evidence rows"
    if len(m2_full) >= 2:
        lo = m2_full.iloc[0]
        hi = m2_full.iloc[-1]
        m2_observation = (
            f"multi-evidence downstream selection cost rises from {lo['mean_downstream_selection_ms']:.1f} ms "
            f"to {hi['mean_downstream_selection_ms']:.1f} ms as frontier budget increases "
            f"from {int(lo['frontier_budget'])} to {int(hi['frontier_budget'])}."
        )

    chrono = m3_df[(m3_df["layout"] == "chronological") & (m3_df["tile_capacity"] == 16)]
    ideal = m3_df[(m3_df["layout"] == "ideal") & (m3_df["tile_capacity"] == 16)]
    m3_observation = "insufficient M3 rows"
    if not chrono.empty and not ideal.empty:
        m3_observation = (
            f"chronological mean tile utilization at capacity 16 is {chrono['mean_tile_utilization'].mean():.3f}, "
            f"versus the ideal lower-bound utilization {ideal['mean_tile_utilization'].mean():.3f}."
        )

    o1_real = o1_df[(o1_df["model"] == "real") & (o1_df["metric"] == "adjacent_overlap") & (o1_df["window_size"] == 1)]
    o1_phase = o1_df[(o1_df["model"] == "phase_shuffle") & (o1_df["metric"] == "adjacent_overlap") & (o1_df["window_size"] == 1)]
    o1_memory = o1_df[(o1_df["model"] == "memory_shuffle") & (o1_df["metric"] == "adjacent_overlap") & (o1_df["window_size"] == 1)]
    o1_observation = "insufficient O1 rows"
    if not o1_real.empty and not o1_phase.empty and not o1_memory.empty:
        o1_observation = (
            f"real-trace adjacent overlap at window=1 is {o1_real['value'].mean():.4f}; "
            f"memory-shuffle drops it to {o1_memory['value'].mean():.4f}, while phase-shuffle stays at {o1_phase['value'].mean():.4f} because it preserves the access sets and only perturbs labels."
        )

    results_lines = [
        "# RESULTS",
        "",
        "## Environment",
        "- Server / GPU: validated on July 3, 2026 against the `agent-memory-profiling` Conda environment; current server exposes NVIDIA GeForce RTX 3090 GPUs.",
        "- Conda environment: `agent-memory-profiling`.",
        "- Git commit: unavailable because the workspace contains an empty `.git/` directory rather than a valid git metadata tree.",
        "- Historical source runs: `20260616_phase_aware_real_agent_v3_formal` for M1/M2/M3 and `20260621_real_trace_adaptive_phase_v4_formal` for O1.",
        "",
        "## M1: Memory-scale profiling",
        f"- Main observation: at scale {int(largest_scale_row['memory_size'])}, budget {int(largest_scale_row['frontier_budget'])}, mean rerank latency is {largest_scale_row['t_rerank_mean_ms']:.1f} ms and mean total latency is {largest_scale_row['t_total_mean_ms']:.1f} ms.",
        "- Figure: `fig_m1_latency_breakdown` and `fig_m1_budget_vs_downstream_cost`.",
        "- Caveat: this consolidated package reuses historical budgets `{50,100}` instead of a freshly rerun `{8,16,32,64}` grid.",
        "",
        "## M2: Budget-quality trade-off",
        f"- Main observation: {m2_observation}",
        "- Figure: `fig_m2_quality_cost_pareto`.",
        "- Caveat: quality is proxied by gold-group hit and high-budget stability on the historical V3 synthetic-query workload, not by a benchmark-native answer metric.",
        "",
        "## M3: Candidate scattering",
        f"- Main observation: {m3_observation}",
        "- Figure: `fig_m3_tile_scattering_utilization` and `fig_m3_tile_activation_vs_ideal`.",
        "- Caveat: final evidence count follows the historical repo default `K_o=8`, not the guide-preferred `K_o=5`.",
        "",
        "## O1: Phase-conditioned locality",
        f"- Main observation: {o1_observation}",
        "- Figure: `fig_o1_phase_locality_matrix` and `fig_o1_null_model_gap`.",
        "- Caveat: the available formal V4 traces cover `planning` and `reasoning` under `selection_only`, so phase coverage is narrower than the five-phase idealization in the guide.",
        "",
        "## Readiness for Method Implementation",
        "- What is now empirically supported: rerank-heavy selection cost exists in the historical real-GPU runs; selected memories are physically scattered under chronological placement; real traces show strong non-random adjacent overlap relative to memory-shuffled null traces.",
        "- What remains unproven: a full rerun on the exact guide grid, official task-score-vs-budget curves on real QA, and any claim about hypergraph or PIM speedup.",
    ]
    (VERSION_ROOT / "RESULTS.md").write_text("\n".join(results_lines) + "\n", encoding="utf-8")

    limitations_lines = [
        "# LIMITATIONS",
        "",
        "- This package consolidates historical formal runs instead of re-running a new guide-perfect M1 matrix.",
        "- Historical V3 cross-encoder raws do not preserve `gpu_name` and `git_commit` per record; the suite fills `gpu_name=unknown_historical_v3` and `git_commit=unavailable` rather than inventing metadata.",
        "- The available historical frontier budgets are `{50, 100}` rather than the guide-preferred `{8, 16, 32, 64}`.",
        "- M2 uses a task-proxy based on synthetic query gold-group hit and high-budget stability because the historical V3 path is not an official QA benchmark run.",
        "- O1 relies on the real-trace V4 `selection_only` runs, which preserve real retrieval decisions but omit answer generation and narrow the available phases to `planning` and `reasoning`.",
        "- The workspace `.git/` directory is empty, so historical commit reconstruction is not possible from local metadata.",
    ]
    (VERSION_ROOT / "LIMITATIONS.md").write_text("\n".join(limitations_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# Method-1 Motivation Package",
        "",
        "This directory consolidates the historical real-GPU motivation assets into a single guide-shaped package.",
        "",
        "## Inputs",
        "- V3 cross-encoder formal runs: `versions/20260616_phase_aware_real_agent_v3_formal/`",
        "- V4 real-trace adaptive formal runs: `versions/20260621_real_trace_adaptive_phase_v4_formal/`",
        "",
        "## Generated Outputs",
        "- `raw/`: merged M1 and O1 JSONL files plus M3 analytical rows",
        "- `aggregate/`: M1/M2/M3/O1 CSV summaries",
        "- `figures/`: PNG and PDF figures",
        "- `manifests/`: dataset and source-run manifests",
        "- `RESULTS.md` and `LIMITATIONS.md`",
        "",
        "## Regeneration",
        "```bash",
        "source \"$(conda info --base)/etc/profile.d/conda.sh\"",
        "conda activate agent-memory-profiling",
        "python agent-memory-profiling/versions/20260703_method1_motivation/scripts/run_motivation_suite.py",
        "```",
        "",
        "After regeneration, validate with:",
        "",
        "```bash",
        "python agent-memory-profiling/versions/20260703_method1_motivation/scripts/validate_motivation_outputs.py \\",
        "  --root agent-memory-profiling/versions/20260703_method1_motivation",
        "```",
    ]
    (VERSION_ROOT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate historical motivation runs into a guide-shaped package.")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    _ensure_dirs()
    m1_path, o1_path, _ = _copy_and_enrich_sources()
    _build_manifests(m1_path, o1_path)
    m1_csv, m2_csv = _compute_m1_m2(m1_path)
    _plot_m1_m2(m1_csv, m2_csv)

    if not args.skip_analysis:
        _run_subprocess(
            "analyze_tile_scattering.py",
            "--input",
            str(m1_path),
            "--output-dir",
            str(VERSION_ROOT),
            "--tile-capacity",
            "8",
            "16",
            "32",
            "--seed",
            "2026",
        )
        _run_subprocess(
            "analyze_phase_locality.py",
            "--trace",
            str(o1_path),
            "--phase-map",
            str(CONFIG_DIR / "phase_mapping.yaml"),
            "--output-dir",
            str(VERSION_ROOT),
            "--window-size",
            "1",
            "3",
            "5",
            "--repetitions",
            "20",
            "--seed",
            "2026",
        )
        _write_results_and_limitations()
        _run_subprocess("validate_motivation_outputs.py", "--root", str(VERSION_ROOT))


if __name__ == "__main__":
    main()
