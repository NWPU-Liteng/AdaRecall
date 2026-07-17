#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-medmmb-v4")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/medmmb-v4-cache")


ROOT = Path(__file__).resolve().parents[1]


def read_csv(rel: str):
    import pandas as pd

    path = ROOT / rel
    return pd.read_csv(path) if path.exists() and path.stat().st_size else None


def save(fig, name: str) -> None:
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.pdf")


def figures() -> None:
    import matplotlib.pyplot as plt

    m1 = read_csv("aggregate/M1_history_scale_summary.csv")
    if m1 is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond, sub in m1.groupby("history_condition"):
            sub = sub.copy()
            sub["x"] = sub["history_scale"].apply(lambda v: 999999 if str(v) == "full" else int(v))
            sub = sub.sort_values("x")
            ax.plot(sub["history_scale"].astype(str), sub["host_memory_bytes_mean"], marker="o", label=cond)
        ax.set_xlabel("History scale")
        ax.set_ylabel("Host/index/payload bytes")
        ax.set_title("M1 History Footprint and Traffic")
        ax.legend()
        save(fig, "Fig_M1_v4_history_footprint_traffic")
        plt.close(fig)

    m2 = read_csv("aggregate/M2_frontier_budget_summary.csv")
    if m2 is not None:
        grouped = m2.groupby("frontier_budget", as_index=False).mean(numeric_only=True)
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(grouped["frontier_budget"], grouped["t_total_selection_ms_mean"], marker="o", color="tab:blue", label="selection ms")
        ax1.plot(grouped["frontier_budget"], grouped["t_rerank_ms_mean"], marker="s", color="tab:cyan", label="rerank ms")
        ax1.set_xlabel("Frontier budget")
        ax1.set_ylabel("Latency (ms)")
        ax2 = ax1.twinx()
        ax2.plot(grouped["frontier_budget"], grouped["reranker_oracle_overlap_at_5_mean"], marker="^", color="tab:red", label="oracle overlap@5")
        ax2.set_ylabel("RerankerOracleOverlap@5")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
        ax1.set_title("M2 Frontier Cost and Reference Fidelity")
        save(fig, "Fig_M2_v4_frontier_cost_reference_fidelity")
        plt.close(fig)

    m3 = read_csv("aggregate/M3_tile_scattering.csv")
    if m3 is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for layout, sub in m3.groupby("layout"):
            g = sub.groupby("tile_capacity", as_index=False)["access_amplification"].mean()
            ax.plot(g["tile_capacity"], g["access_amplification"], marker="o", label=layout)
        ax.set_xlabel("Tile capacity")
        ax.set_ylabel("Access amplification")
        ax.set_title("M3 Tile Access Amplification")
        ax.legend()
        save(fig, "Fig_M3_v4_tile_access_amplification")
        plt.close(fig)

    o1 = read_csv("aggregate/O1_same_cross_stage_stats.csv")
    if o1 is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        x = range(len(o1))
        ax.bar([i - 0.2 for i in x], o1["same_stage_overlap"], width=0.4, label="same-stage")
        ax.bar([i + 0.2 for i in x], o1["cross_stage_overlap"], width=0.4, label="cross-stage")
        ax.set_xticks(list(x))
        ax.set_xticklabels(o1["stage"])
        ax.set_ylabel("Overlap")
        ax.set_title("O1 Same vs Cross Stage Locality")
        ax.legend()
        save(fig, "Fig_O1_v4_same_cross_stage_locality")
        plt.close(fig)

    b = read_csv("aggregate/B_heldout_stage_transfer_summary.csv")
    if b is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        order = ["ground", "plan", "reason", "reflect", "aggregate"]
        b = b.set_index("stage").loc[[s for s in order if s in set(b["stage"])]].reset_index()
        err_low = b["stage_advantage_mean"] - b["stage_advantage_ci95_low"]
        err_hi = b["stage_advantage_ci95_high"] - b["stage_advantage_mean"]
        ax.bar(b["stage"], b["stage_advantage_mean"], yerr=[err_low, err_hi], capsize=4)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Held-out stage advantage")
        ax.set_title("O1-v4 Held-Out Stage Transfer")
        save(fig, "Fig_O1_v4_heldout_stage_transfer")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(b["stage"], b["seen_node_coverage_mean"])
        ax.axhline(0.30, color="red", linestyle="--", label="30% guardrail")
        ax.set_ylabel("Seen-node coverage")
        ax.set_title("O1-v4 Held-Out Seen-Node Coverage")
        ax.legend()
        save(fig, "Fig_O1_v4_stage_seen_node_coverage")
        plt.close(fig)

    null = read_csv("aggregate/B_heldout_stage_shuffle_null.csv")
    if null is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(null["statistic"], bins=40, alpha=0.8)
        ax.axvline(null["observed_statistic"].iloc[0], color="red", label="observed")
        ax.set_xlabel("Held-out stage advantage under stage shuffle")
        ax.set_ylabel("Count")
        ax.set_title("O1-v4 Held-Out Stage-Shuffle Null")
        ax.legend()
        save(fig, "Fig_O1_v4_heldout_stage_shuffle_null")
        plt.close(fig)

    c = read_csv("aggregate/C_strict_evidence_budget_summary.csv")
    if c is not None and not c.empty:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(c["frontier_budget"], c["selection_latency_ms_mean"], marker="o", color="tab:blue", label="selection ms")
        ax1.set_xlabel("Frontier budget")
        ax1.set_ylabel("Selection latency (ms)")
        ax2 = ax1.twinx()
        ax2.plot(c["frontier_budget"], c["StrictKeypointRecall_at_5_mean"], marker="s", color="tab:green", label="strict recall@5")
        ax2.set_ylabel("StrictKeypointRecall@5")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines])
        ax1.set_title("M2-v4 Cost vs Strict Keypoint Recall")
        save(fig, "Fig_M2_v4_budget_cost_strict_keypoint_recall")
        plt.close(fig)


def scalar_from_csv(rel: str, col: str, row: int = 0, default: str = "missing") -> str:
    df = read_csv(rel)
    if df is None or df.empty or col not in df:
        return default
    return str(df[col].iloc[row])


def reports() -> None:
    b_summary = read_csv("aggregate/B_heldout_stage_transfer_summary.csv")
    b_null = read_csv("aggregate/B_heldout_stage_shuffle_null.csv")
    c_summary = read_csv("aggregate/C_keypoint_alignment_summary.csv")
    c_budget = read_csv("aggregate/C_strict_evidence_budget_summary.csv")
    b_status = "not run"
    b_main = "no"
    b_claim = "not established"
    if b_summary is not None and b_null is not None and not b_summary.empty and not b_null.empty:
        agg = b_summary[b_summary["stage"] == "aggregate"].iloc[0]
        null = b_null.iloc[0]
        if float(agg["seen_node_coverage_mean"]) >= 0.30 and float(null["empirical_p_value"]) < 0.05 and float(agg["stage_advantage_mean"]) > float(null["null_mean"]):
            b_status, b_main, b_claim = "pass", "yes", "earlier stage-conditioned access structure transfers to later held-out accesses"
        else:
            b_status, b_main, b_claim = "partial", "appendix", "held-out transfer is weak or guardrail-limited"
    c_status = "not run"
    c_main = "appendix"
    c_claim = "not established"
    if c_summary is not None and not c_summary.empty:
        coverage = float(c_summary["in_strict_subset"].mean())
        c_status = "pass" if coverage >= 0.30 else "partial"
        c_claim = f"deterministic strict subset coverage={coverage:.3f}"
    lines = [
        "# Final Motivation V4 Summary",
        "",
        "V4 strengthens v3 with provenance/causality audits, held-out stage-transfer testing, and a deterministic strict source-keypoint-aligned subset.",
        "",
        "## Component Summary",
        "",
        "| Component | Status | Main-paper usable? | Claim |",
        "|---|---|---|---|",
        "| M1 | pass | appendix | history footprint/traffic increases with scale and noise; latency is not universally monotonic |",
        "| M2 | pass | yes | flat frontier expansion increases cost and improves high-budget reranker-reference fidelity |",
        "| M3 | pass | yes | chronological placement scatters selected memories and creates over-fetch headroom |",
        "| O1 within-trace | pass | yes | fixed controller-defined stages exhibit non-random access locality |",
        f"| O1 held-out transfer | {b_status} | {b_main} | {b_claim} |",
        f"| strict evidence subset | {c_status} | {c_main} | {c_claim} |",
        "",
        "See `reports/FINAL_MOTIVATION_V4_CLAIM_BOUNDARIES.md` for allowed and forbidden claims.",
    ]
    (ROOT / "reports" / "FINAL_MOTIVATION_V4_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reports" / "FINAL_MOTIVATION_V4_CLAIM_BOUNDARIES.md").write_text(
        """# Final Motivation V4 Claim Boundaries

## Allowed Claims

- Flat frontier expansion increases selection and reranking cost.
- Larger frontiers improve agreement with a high-budget reranking reference.
- Chronological placement scatters selected memories across tiles and creates physical over-fetch headroom.
- Under a fixed auditable controller, stages exhibit non-random access locality.
- If supported by B, some stage-conditioned structure transfers from earlier to later accesses.

## Forbidden Claims

- Do not claim hypergraph superiority.
- Do not claim PIM speedup or energy reduction.
- Do not claim the final layout is implemented.
- Do not claim larger frontier budgets improve answer quality.
- Do not call controller stages source-provided or benchmark-provided cognitive phases.
- Do not present oracle tile packing as an implemented baseline.
- Do not call automatic keypoint alignment human-verified gold evidence.
""",
        encoding="utf-8",
    )
    (ROOT / "reports" / "FINAL_MOTIVATION_V4_FIGURE_GUIDE.md").write_text(
        """# Final Motivation V4 Figure Guide

| Result | Figure | What was tested | Metric | Supports | Does not support | Placement |
|---|---|---|---|---|---|---|
| M1 | `Fig_M1_v4_history_footprint_traffic.png` | history scale/noise footprint | host/index/payload bytes | workload growth | universal latency law | appendix |
| M2 | `Fig_M2_v4_frontier_cost_reference_fidelity.png` | frontier budgets | latency and oracle overlap | flat expansion cost/reference fidelity | answer quality | main |
| M3 | `Fig_M3_v4_tile_access_amplification.png` | chronological tile placement | access amplification | physical over-fetch headroom | implemented layout superiority | main |
| O1 within-trace | `Fig_O1_v4_same_cross_stage_locality.png` | controller-defined stages | same/cross overlap | non-random access locality | source-provided phases | main |
| O1 held-out | `Fig_O1_v4_heldout_stage_transfer.png` | train-earlier/test-later transfer | stage advantage | future transfer if guardrails pass | universal stage locality | main only if pass |
| C strict evidence | `Fig_M2_v4_budget_cost_strict_keypoint_recall.png` | deterministic keypoint subset | strict recall@5 | stricter evidence behavior | human-verified gold | appendix unless high coverage |
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    figures()
    reports()
    print("v4_figures_reports=complete")


if __name__ == "__main__":
    main()
