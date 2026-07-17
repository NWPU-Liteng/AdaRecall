#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VERSION_ROOT = Path(__file__).resolve().parents[1]


def _collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted((root / "runs").glob("v4_*_budget*")):
        raw_path = run_dir / "raw_profile.jsonl"
        if not raw_path.exists():
            continue
        df = pd.read_json(raw_path, lines=True)
        clean = df[df["error"].fillna("") == ""].copy()
        if clean.empty:
            continue
        parts = run_dir.name.split("_")
        tier = parts[1]
        dataset = "_".join(parts[2:-1])
        budget = int(parts[-1].replace("budget", ""))
        rows.append(
            {
                "tier": tier,
                "dataset": dataset,
                "budget": budget,
                "n_queries": int(len(clean)),
                "avg_total_response_latency_ms": float(clean["total_response_latency_ms"].mean()),
                "p95_total_response_latency_ms": float(clean["total_response_latency_ms"].quantile(0.95)),
                "avg_selection_path_only_latency_ms": float(clean["selection_path_only_latency_ms"].mean()),
                "avg_phase_rerank_total_latency_ms": float(clean["phase_rerank_total_latency_ms"].mean()),
                "avg_evidence_recall": float(clean["evidence_recall"].mean()),
                "avg_answer_exact_match": float(clean["answer_exact_match"].dropna().mean()) if clean["answer_exact_match"].dropna().size else float("nan"),
                "avg_answer_extracted_exact_match": float(clean["answer_extracted_exact_match"].dropna().mean()) if "answer_extracted_exact_match" in clean.columns and clean["answer_extracted_exact_match"].dropna().size else float("nan"),
                "avg_answer_token_f1": float(clean["answer_token_f1"].dropna().mean()) if "answer_token_f1" in clean.columns and clean["answer_token_f1"].dropna().size else float("nan"),
                "avg_answer_extracted_token_f1": float(clean["answer_extracted_token_f1"].dropna().mean()) if "answer_extracted_token_f1" in clean.columns and clean["answer_extracted_token_f1"].dropna().size else float("nan"),
                "avg_answer_contains_reference": float(clean["answer_contains_reference"].dropna().mean()) if "answer_contains_reference" in clean.columns and clean["answer_contains_reference"].dropna().size else float("nan"),
                "avg_answer_extracted_contains_reference": float(clean["answer_extracted_contains_reference"].dropna().mean()) if "answer_extracted_contains_reference" in clean.columns and clean["answer_extracted_contains_reference"].dropna().size else float("nan"),
                "avg_answer_insufficient_information_rate": float(clean["answer_insufficient_information_rate"].dropna().mean()) if "answer_insufficient_information_rate" in clean.columns and clean["answer_insufficient_information_rate"].dropna().size else float("nan"),
                "avg_retrieval_invocation_count": float(clean["retrieval_invocation_count"].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V4 official-QA budget-grid runs.")
    parser.add_argument("--root", default=str(VERSION_ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = _collect_rows(root)
    if not rows:
        raise SystemExit("No completed budget-grid runs found.")
    df = pd.DataFrame(rows).sort_values(["tier", "dataset", "budget"])
    out_dir = root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "v4_official_qa_budget_grid_summary.csv"
    md_path = out_dir / "v4_official_qa_budget_grid_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text("# V4 Official QA Budget Grid Summary\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"summary_csv={csv_path}")
    print(f"summary_md={md_path}")


if __name__ == "__main__":
    main()
