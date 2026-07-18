from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import VERSION_ROOT, add_common_args, config_hash, load_config
from v24.io import iter_jsonl, read_json, write_json


def load_rows(path: Path) -> list[dict]:
    return list(iter_jsonl(path)) if path.exists() else []


def paired_bootstrap(values: list[float], *, seed: int = 20260718, samples: int = 2000) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "ci95": [None, None]}
    if len(array) == 1:
        value = float(array[0])
        return {"count": 1, "mean": value, "median": value, "ci95": [value, value]}
    rng = np.random.default_rng(int(seed))
    bootstrap = array[rng.integers(0, len(array), size=(int(samples), len(array)))].mean(axis=1)
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
    }


def aggregate(rows: list[dict]) -> tuple[dict, list[dict]]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["query_id"]][row["method"]] = row
    methods = defaultdict(list)
    table = []
    for pair in grouped.values():
        for method, row in pair.items():
            metrics = row.get("metrics", {})
            methods[method].append(row)
            reference = pair.get("full_n", {}).get("reranked_ids", [])
            table.append({
                "method": method, "n": len(pair.get("full_n", {}).get("frontier_ids", [])),
                "c": len(row.get("frontier_ids", [])), "k": len(row.get("reranked_ids", [])),
                "precision_at_k": metrics.get("precision_at_k"), "ndcg_at_10": metrics.get("ndcg_at_10"),
                "ref_overlap": float(len(set(row.get("reranked_ids", [])) & set(reference)) / max(len(set(reference)), 1)),
                "method_ms": float(row.get("edge_activation_ms", 0.0) + row.get("diffusion_ms", 0.0) + row.get("frontier_finalize_ms", 0.0)),
                "rerank_ms": row.get("rerank_ms"), "total_ms": row.get("semantic_selection_ms"),
                "peak_memory": row.get("peak_cpu_bytes", 0),
            })
    summary = {"rows": len(rows), "methods": {}}
    for method, values in sorted(methods.items()):
        summary["methods"][method] = {"count": len(values)}
        for key in ("precision_at_k", "ndcg_at_10", "evidence_group_coverage_final", "candidate_recall_frontier", "semantic_selection_ms", "rerank_ms", "reranker_pairs", "ref_overlap", "method_ms", "total_ms", "peak_memory"):
            extracted = []
            for value in values:
                candidate = value.get("metrics", {}).get(key) if key in value.get("metrics", {}) else value.get(key)
                if isinstance(candidate, (int, float)):
                    extracted.append(float(candidate))
            numbers = extracted
            if numbers:
                array = np.asarray(numbers, dtype=np.float64)
                summary["methods"][method][key] = {"mean": float(array.mean()), "median": float(np.median(array)), "p95": float(np.percentile(array, 95))}
    return summary, table


def make_figures(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    figures = []
    path = VERSION_ROOT / "figures"
    path.mkdir(parents=True, exist_ok=True)
    names = {"ndcg_at_10": "nDCG@10", "total_ms": "Semantic-selection ms"}
    for key, title in names.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        labels, values = [], []
        for method in ("full_n", "dense_c", "ours_c"):
            numbers = [float(row[key]) for row in by_method[method] if row.get(key) is not None]
            if numbers:
                labels.append(method); values.append(float(np.mean(numbers)))
        ax.bar(labels, values, color=["#52616b", "#1f7a8c", "#bf5700"][:len(labels)])
        ax.set_title(title); ax.set_ylabel(title); fig.tight_layout()
        output = path / f"{key}_by_method.png"; fig.savefig(output, dpi=140); plt.close(fig); figures.append(str(output))
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color in (("full_n", "#52616b"), ("dense_c", "#1f7a8c"), ("ours_c", "#bf5700")):
        numbers = [float(row["total_ms"]) for row in by_method[method] if row.get("total_ms") is not None]
        if numbers:
            ax.plot(np.sort(numbers), label=method, color=color)
    ax.set_title("Per-query semantic-selection latency"); ax.set_xlabel("Sorted query"); ax.set_ylabel("ms"); ax.legend(); fig.tight_layout()
    output = path / "latency_curve.png"; fig.savefig(output, dpi=140); plt.close(fig); figures.append(str(output))
    return figures


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    model_manifest_path = VERSION_ROOT / "artifacts" / "models" / "models_manifest.json"
    dataset_manifest_path = VERSION_ROOT / "artifacts" / "manifests" / "dataset_manifest.json"
    history_manifest_path = VERSION_ROOT / "artifacts" / "manifests" / "history_event_manifest.json"
    model_manifest = read_json(model_manifest_path) if model_manifest_path.exists() else {}
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.exists() else {}
    history_manifest = read_json(history_manifest_path) if history_manifest_path.exists() else {}
    gate_a_passed = bool(model_manifest.get("passed") and model_manifest.get("embedding", {}).get("present") and model_manifest.get("reranker", {}).get("present"))
    gate_b_passed = bool(dataset_manifest.get("query_id_overlaps") == {} and history_manifest.get("test_used") is False and history_manifest.get("gold_used") is False)
    dev_gate = read_json(VERSION_ROOT / "aggregate" / "development_gate.json") if (VERSION_ROOT / "aggregate" / "development_gate.json").exists() else {"passed": False, "status": "missing"}
    test_rows = []
    test_status = {"status": "NOT_RUN"}
    for run_dir in sorted((VERSION_ROOT / "runs").glob("v24_test_once_*/query_results.jsonl")):
        test_rows = load_rows(run_dir)
        test_status = {"status": "COMPLETED_ONCE", "path": str(run_dir), "rows": len(test_rows)}
        break
    dev_rows = []
    for run_dir in sorted((VERSION_ROOT / "runs").glob("v24_development_*/query_results.jsonl")):
        dev_rows = load_rows(run_dir)
        break
    selected_rows = test_rows or dev_rows
    summary, table = aggregate(selected_rows)
    write_json(VERSION_ROOT / "aggregate" / "aggregate_metrics.json", summary)
    with (VERSION_ROOT / "aggregate" / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "n", "c", "k", "precision_at_k", "ndcg_at_10", "ref_overlap", "method_ms", "rerank_ms", "total_ms", "peak_memory"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(table)
    agent_path = VERSION_ROOT / "aggregate" / "agent_replay_summary.json"
    agent = read_json(agent_path) if agent_path.exists() else {"status": "NOT_RUN"}
    test_quality_passed = False
    test_cost_passed = False
    test_comparison = {"query_count": 0, "ours_minus_dense_ndcg": paired_bootstrap([]), "ours_minus_full_ndcg": paired_bootstrap([]), "net_gain_ms": paired_bootstrap([])}
    test_quality_checks = {"dense_mean_non_negative": False, "dense_ci_lower_within_epsilon": False, "full_ci_lower_within_epsilon": False}
    if test_rows:
        grouped = defaultdict(dict)
        for row in test_rows:
            grouped[row["query_id"]][row["method"]] = row
        pairs = [pair for pair in grouped.values() if {"full_n", "dense_c", "ours_c"} <= set(pair)]
        if pairs:
            dense_deltas = [float(pair["ours_c"]["metrics"]["ndcg_at_10"]) - float(pair["dense_c"]["metrics"]["ndcg_at_10"]) for pair in pairs]
            full_deltas = [float(pair["ours_c"]["metrics"]["ndcg_at_10"]) - float(pair["full_n"]["metrics"]["ndcg_at_10"]) for pair in pairs]
            net_gains = [float(pair["full_n"]["semantic_selection_ms"]) - float(pair["ours_c"]["semantic_selection_ms"]) for pair in pairs]
            test_comparison = {
                "query_count": len(pairs),
                "ours_minus_dense_ndcg": paired_bootstrap(dense_deltas, seed=20260718),
                "ours_minus_full_ndcg": paired_bootstrap(full_deltas, seed=20260719),
                "net_gain_ms": paired_bootstrap(net_gains, seed=20260720),
            }
            dense_delta = test_comparison["ours_minus_dense_ndcg"]["mean"]
            full_delta = test_comparison["ours_minus_full_ndcg"]["mean"]
            net = test_comparison["net_gain_ms"]["mean"]
            test_quality_checks = {
                "dense_mean_non_negative": bool(dense_delta >= 0.0),
                "dense_ci_lower_within_epsilon": bool(test_comparison["ours_minus_dense_ndcg"]["ci95"][0] >= -float(config["gate"]["dense_quality_epsilon"])),
                "full_ci_lower_within_epsilon": bool(test_comparison["ours_minus_full_ndcg"]["ci95"][0] >= -float(config["gate"]["full_quality_epsilon"])),
            }
            test_quality_passed = bool(all(test_quality_checks.values()))
            test_cost_passed = bool(net > 0.0)
    agent_passed = bool(agent.get("status") == "COMPLETED" and agent.get("quality_non_decreasing_vs_full"))
    closure_passed = bool(gate_a_passed and gate_b_passed and dev_gate.get("passed") and test_rows and test_quality_passed and test_cost_passed and agent_passed)
    figures = make_figures(table)
    state = {
        "status": "METHOD1_CLOSURE_PASSED" if closure_passed else "METHOD1_CLOSURE_FAILED_OR_INCOMPLETE",
        "config_hash": config_hash(config), "development_gate": dev_gate, "test_status": test_status,
        "gate_a_model_passed": gate_a_passed, "gate_b_data_passed": gate_b_passed,
        "test_quality_passed": test_quality_passed, "test_cost_passed": test_cost_passed,
        "test_comparison": test_comparison, "test_quality_checks": test_quality_checks,
        "agent_replay": agent, "figures": figures, "test_read_status": "read_once" if test_rows else "not_read",
    }
    write_json(VERSION_ROOT / "aggregate" / "closure_state.json", state)
    method_summary = summary.get("methods", {})
    table_by_method = defaultdict(list)
    for row in table:
        table_by_method[row["method"]].append(row)

    def table_value(method: str, key: str) -> str:
        values = [float(row[key]) for row in table_by_method[method] if isinstance(row.get(key), (int, float))]
        return f"{float(np.mean(values)):.4f}" if values else "NA"

    model_attention = {
        "configured": model_manifest.get("attention"),
        "embedding_actual": model_manifest.get("embedding_smoke", {}).get("attention_implementation"),
        "reranker_actual": model_manifest.get("reranker_smoke", {}).get("attention_implementation"),
    }
    report = [
        "# Method 1 PRISM-Aligned Closure Report",
        "",
        f"## Status: `{state['status']}`",
        "",
        "This report preserves negative or incomplete Gates. It does not promote a method based on a development-only or dense-reference result.",
        "",
        "## Gate State",
        "",
        f"- Gate A environment/model: `{gate_a_passed}`",
        f"- Gate B data/leakage: `{gate_b_passed}`",
        f"- development quality/cost Gate: `{bool(dev_gate.get('passed'))}`",
        f"- formal test: `{test_status.get('status')}`",
        f"- formal test quality: `{test_quality_passed}`",
        f"- formal test net cost: `{test_cost_passed}`",
        f"- offline Agent Memory replay: `{agent_passed}`",
        "",
        "## Frozen Configuration",
        "",
        f"- N/S/C/K: `{config['budget']['full_n']}/{config['budget']['seed_s']}/{config['budget']['frontier_c']}/{config['budget']['final_k']}`",
        f"- semantic prototypes/top memberships: `{config['semantic_hypergraph']['n_prototypes']}/{config['semantic_hypergraph']['top_memberships']}`",
        f"- diffusion alpha/steps/beta: `{config['diffusion']['alpha_restart']}/{config['diffusion']['steps']}/{config['diffusion']['beta_dense']}`",
        f"- candidate reduction: `{1.0 - config['budget']['frontier_c'] / config['budget']['full_n']:.3f}`",
        f"- model attention configured/actual: `{model_attention['configured']}/{model_attention['embedding_actual']}/{model_attention['reranker_actual']}`",
        f"- model loading: `local_files_only=true`, embedding and reranker dimensions/checks in `artifacts/models/models_manifest.json`",
        "- dataset: existing MedMemoryBench English inputs; no new dataset was downloaded for this closure.",
        "",
        "## Formal Test Paired Summary",
        "",
        f"- paired queries: `{test_comparison['query_count']}`; test status: `{test_status.get('status')}`; read: `{state['test_read_status']}`",
        f"- Ours-C minus Dense-C nDCG@10: `{test_comparison['ours_minus_dense_ndcg']['mean']}`; 95% bootstrap CI `{test_comparison['ours_minus_dense_ndcg']['ci95']}`; frozen epsilon `-{config['gate']['dense_quality_epsilon']}`",
        f"- Ours-C minus Full-N nDCG@10: `{test_comparison['ours_minus_full_ndcg']['mean']}`; 95% bootstrap CI `{test_comparison['ours_minus_full_ndcg']['ci95']}`; frozen epsilon `-{config['gate']['full_quality_epsilon']}`",
        f"- Full-N minus Ours-C semantic-selection net gain (ms): `{test_comparison['net_gain_ms']['mean']}`; 95% bootstrap CI `{test_comparison['net_gain_ms']['ci95']}`",
        f"- Strict Gate E quality checks: `{test_quality_checks}`.",
        "- The formal test quality gate is not promoted: Ours-C has a negative mean nDCG delta versus Dense-C and its paired bootstrap lower bound crosses the frozen Dense-C epsilon. No test rerun or post-test tuning was performed.",
        "",
        "## Main Table",
        "",
        "| Method | N | C | K | Precision@K | nDCG@10 | Ref overlap | Method ms | Rerank ms | Total ms | Peak memory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("full_n", "dense_c", "ours_c"):
        values = method_summary.get(method, {})
        def value(key: str) -> str:
            return f"{values.get(key, {}).get('mean', float('nan')):.4f}" if isinstance(values.get(key), dict) else "NA"
        report.append(f"| {method} | {config['budget']['full_n']} | {config['budget']['frontier_c'] if method != 'full_n' else config['budget']['full_n']} | {config['budget']['final_k']} | {value('precision_at_k')} | {value('ndcg_at_10')} | {table_value(method, 'ref_overlap')} | {table_value(method, 'method_ms')} | {value('rerank_ms')} | {value('semantic_selection_ms')} | {table_value(method, 'peak_memory')} |")
    report.extend([
        "",
        "## Agent Memory Replay",
        "",
        f"- tasks/events: `{agent.get('tasks', 0)}`; task success available: `{agent.get('task_success_available', False)}`; source semantics: `co_candidate`",
    ])
    for method in ("full_n", "dense_c", "ours_c"):
        values = agent.get("methods", {}).get(method, {})
        report.append(
            f"- `{method}`: final reference agreement `{values.get('reference_agreement_final')}`, frontier agreement `{values.get('reference_agreement_frontier')}`, "
            f"pairs `{values.get('reranker_pairs_sum')}`, semantic-selection sum ms `{values.get('semantic_selection_ms_sum')}`"
        )
    report.extend([
        "",
        "## Artifacts",
        "",
        f"- development results: `runs/` matching `v24_development_*/query_results.jsonl`",
        f"- test result: `{test_status.get('path', 'not generated')}`",
        "- canonical data and split manifests: `artifacts/manifests/`",
        "- semantic/history diagnostics: `artifacts/hypergraph/`",
        "- machine-readable state: `aggregate/closure_state.json`",
        "- figures: `figures/`",
        "",
        "## Limitations and Negative Cases",
        "",
        "- The history view is derived from frozen train ANN candidate lists and is labeled `co_candidate`; it is not a native agent-read trace.",
        "- The selector does not inspect query type or gold evidence. Those fields are loaded only by the evaluation layer.",
        "- The configured SDPA attention path was unavailable with torch 2.1.0 and transformers 4.51.3; both official Qwen3 wrappers explicitly recorded an eager-attention fallback. No PyTorch/CUDA upgrade was performed.",
        "- Absolute quality values are sparse on this workload; this version does not claim a passed quality closure. The reproducible positive signal is development/replay semantic-selection speed, while the formal test quality gate remains failed.",
        "- If the environment cannot load the official models or expose a usable GPU, the run remains an implementation/smoke artifact and is not a formal result.",
        "- No test configuration is changed after the first formal test read.",
    ])
    (VERSION_ROOT / "reports" / "closure_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(state, sort_keys=True))


if __name__ == "__main__":
    main()
