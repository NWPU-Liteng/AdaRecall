#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parent.parent
SOURCE_MANIFEST_ROOT = (
    PROJECT_ROOT / "versions" / "20260621_real_trace_adaptive_phase_v4_formal" / "manifests"
)


FORMAL_BUDGETS = [8, 16, 32, 64]
DATASETS = ["locomo", "longmemeval_s"]
LONG_RANGE_QUANTILE = 0.25
FINAL_TOP_K = 5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_runtime_events(dataset: str) -> dict[str, dict[str, Any]]:
    path = SOURCE_MANIFEST_ROOT / dataset / "runtime_manifest.jsonl"
    rows = _read_jsonl(path)
    return {row["event_id"]: row for row in rows if row.get("event_type") == "official_qa"}


def _build_memory_rank_map(dataset: str) -> dict[str, int]:
    path = SOURCE_MANIFEST_ROOT / dataset / "runtime_memory_bank.jsonl"
    rank_map: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rank_map[row["memory_id"]] = idx
    return rank_map


def _load_labels(dataset: str) -> list[dict[str, Any]]:
    return _read_jsonl(SOURCE_MANIFEST_ROOT / dataset / "evaluation_labels.jsonl")


def _normalize_question_type(value: Any) -> str:
    return str(value) if value is not None else "unknown"


def _derive_primary_stratum(dataset: str, question_type: str, gold_turn_ids: list[str]) -> str:
    gold_count = len(gold_turn_ids)
    if dataset == "longmemeval_s":
        if question_type == "multi-session" or gold_count >= 2:
            return "multi_evidence"
        return "single_evidence"
    if question_type in {"1", "3"} or gold_count >= 2:
        return "multi_evidence"
    return "single_evidence"


def _derive_long_range_flag(
    event_id: str,
    runtime_event: dict[str, Any],
    gold_turn_ids: list[str],
    memory_rank_map: dict[str, int],
) -> tuple[bool, float | None, float | None]:
    available = list(runtime_event.get("memory_ids_available") or [])
    visible_count = len(available)
    if visible_count <= 0:
        return False, None, None

    visible_rank_map = {memory_id: idx for idx, memory_id in enumerate(available)}
    positions: list[float] = []
    fallback_positions: list[float] = []
    for memory_id in gold_turn_ids:
        if memory_id in visible_rank_map:
            positions.append(visible_rank_map[memory_id] / visible_count)
        elif memory_id in memory_rank_map:
            fallback_positions.append(memory_rank_map[memory_id] / max(len(memory_rank_map), 1))
    if positions:
        mean_visible_position = float(sum(positions) / len(positions))
        return False, mean_visible_position, None
    if fallback_positions:
        mean_fallback_position = float(sum(fallback_positions) / len(fallback_positions))
        return False, None, mean_fallback_position
    return False, None, None


def _collect_strata_rows(root: Path) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        runtime_events = _load_runtime_events(dataset)
        memory_rank_map = _build_memory_rank_map(dataset)
        label_rows = _load_labels(dataset)

        base_rows: list[dict[str, Any]] = []
        for label in label_rows:
            event_id = label["event_id"]
            runtime_event = runtime_events.get(event_id)
            if runtime_event is None:
                continue
            question_type = _normalize_question_type(label.get("official_question_type"))
            gold_turn_ids = [memory_id for memory_id in (label.get("gold_turn_ids") or []) if memory_id]
            primary_stratum = _derive_primary_stratum(dataset, question_type, gold_turn_ids)
            _, visible_position, fallback_position = _derive_long_range_flag(
                event_id=event_id,
                runtime_event=runtime_event,
                gold_turn_ids=gold_turn_ids,
                memory_rank_map=memory_rank_map,
            )
            base_rows.append(
                {
                    "dataset": dataset,
                    "event_id": event_id,
                    "tenant_id": runtime_event.get("tenant_id"),
                    "query_text": runtime_event.get("query_text", ""),
                    "official_question_type": question_type,
                    "gold_turn_count": len(gold_turn_ids),
                    "gold_session_count": len(label.get("gold_session_ids") or []),
                    "visible_memory_count": len(runtime_event.get("memory_ids_available") or []),
                    "gold_mean_visible_position": visible_position,
                    "gold_mean_global_position": fallback_position,
                    "primary_stratum": primary_stratum,
                    "is_multi_evidence": primary_stratum == "multi_evidence",
                    "is_single_evidence": primary_stratum == "single_evidence",
                }
            )

        base_df = pd.DataFrame(base_rows)
        if base_df.empty:
            continue

        position_series = base_df["gold_mean_visible_position"].dropna()
        long_range_threshold = (
            float(position_series.quantile(LONG_RANGE_QUANTILE)) if not position_series.empty else 0.0
        )
        base_df["long_range_threshold"] = long_range_threshold
        base_df["is_long_range"] = base_df["gold_mean_visible_position"].fillna(1.0) <= long_range_threshold

        reference_selected: dict[str, list[str]] = {}
        budget_frames: list[pd.DataFrame] = []
        for budget in FORMAL_BUDGETS:
            raw_path = root / "runs" / f"v4_formal_{dataset}_budget{budget}" / "raw_profile.jsonl"
            if not raw_path.exists():
                continue
            raw_df = pd.read_json(raw_path, lines=True)
            clean = raw_df[raw_df["error"].fillna("") == ""].copy()
            if clean.empty:
                continue
            clean["budget"] = budget
            if budget == max(FORMAL_BUDGETS):
                reference_selected = {
                    row["event_id"]: list(row.get("selected_memory_ids") or [])
                    for row in clean.to_dict("records")
                }
            budget_frames.append(clean)

        if not budget_frames:
            continue

        combined = pd.concat(budget_frames, ignore_index=True)
        combined["selected_memory_ids"] = combined["selected_memory_ids"].apply(
            lambda value: value if isinstance(value, list) else []
        )
        combined["selection_downstream_ms"] = (
            combined["selection_path_only_latency_ms"].astype(float)
            + combined["phase_rerank_total_latency_ms"].astype(float)
        )

        def _stability(event_id: str, selected_ids: list[str]) -> float:
            reference = reference_selected.get(event_id, [])
            return len(set(selected_ids) & set(reference)) / float(FINAL_TOP_K)

        combined["high_budget_reranker_reference_stability"] = combined.apply(
            lambda row: _stability(row["event_id"], row["selected_memory_ids"]),
            axis=1,
        )

        merged = combined.merge(base_df, on=["dataset", "event_id"], how="left")
        all_rows.extend(merged.to_dict("records"))

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def _summarize_query_strata(query_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata_specs = [
        ("single_evidence", query_df["is_single_evidence"]),
        ("multi_evidence", query_df["is_multi_evidence"]),
        ("long_range", query_df["is_long_range"]),
    ]
    for dataset in sorted(query_df["dataset"].dropna().unique()):
        dataset_df = query_df[query_df["dataset"] == dataset].copy()
        for budget in sorted(dataset_df["budget"].dropna().unique()):
            budget_df = dataset_df[dataset_df["budget"] == budget].copy()
            for stratum_name, mask in strata_specs:
                subset = budget_df[mask.loc[budget_df.index]]
                if subset.empty:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "query_stratum": stratum_name,
                        "budget": int(budget),
                        "n_queries": int(len(subset)),
                        "mean_downstream_selection_ms": float(subset["selection_downstream_ms"].mean()),
                        "mean_selected_memory_count": float(
                            subset["selected_memory_ids"].apply(len).mean()
                        ),
                        "mean_answer_exact_match": float(subset["answer_exact_match"].dropna().mean())
                        if subset["answer_exact_match"].dropna().size
                        else float("nan"),
                        "mean_answer_extracted_exact_match": float(
                            subset["answer_extracted_exact_match"].dropna().mean()
                        )
                        if "answer_extracted_exact_match" in subset.columns
                        and subset["answer_extracted_exact_match"].dropna().size
                        else float("nan"),
                        "mean_answer_token_f1": float(subset["answer_token_f1"].dropna().mean())
                        if "answer_token_f1" in subset.columns and subset["answer_token_f1"].dropna().size
                        else float("nan"),
                        "mean_answer_extracted_token_f1": float(
                            subset["answer_extracted_token_f1"].dropna().mean()
                        )
                        if "answer_extracted_token_f1" in subset.columns
                        and subset["answer_extracted_token_f1"].dropna().size
                        else float("nan"),
                        "mean_answer_contains_reference": float(
                            subset["answer_contains_reference"].dropna().mean()
                        )
                        if subset["answer_contains_reference"].dropna().size
                        else float("nan"),
                        "mean_answer_extracted_contains_reference": float(
                            subset["answer_extracted_contains_reference"].dropna().mean()
                        )
                        if "answer_extracted_contains_reference" in subset.columns
                        and subset["answer_extracted_contains_reference"].dropna().size
                        else float("nan"),
                        "mean_answer_insufficient_information_rate": float(
                            subset["answer_insufficient_information_rate"].dropna().mean()
                        )
                        if "answer_insufficient_information_rate" in subset.columns
                        and subset["answer_insufficient_information_rate"].dropna().size
                        else float("nan"),
                        "mean_evidence_recall": float(subset["evidence_recall"].mean()),
                        "mean_high_budget_reference_stability": float(
                            subset["high_budget_reranker_reference_stability"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "query_stratum", "budget"])


def _summarize_strata_population(query_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        query_df[
            [
                "dataset",
                "event_id",
                "official_question_type",
                "gold_turn_count",
                "gold_session_count",
                "visible_memory_count",
                "gold_mean_visible_position",
                "primary_stratum",
                "is_long_range",
            ]
        ]
        .drop_duplicates()
        .copy()
    )
    rows: list[dict[str, Any]] = []
    for dataset in sorted(base["dataset"].dropna().unique()):
        subset = base[base["dataset"] == dataset].copy()
        rows.append(
            {
                "dataset": dataset,
                "n_queries": int(len(subset)),
                "n_single_evidence": int((subset["primary_stratum"] == "single_evidence").sum()),
                "n_multi_evidence": int((subset["primary_stratum"] == "multi_evidence").sum()),
                "n_long_range": int(subset["is_long_range"].sum()),
                "mean_gold_turn_count": float(subset["gold_turn_count"].mean()),
                "mean_visible_memory_count": float(subset["visible_memory_count"].mean()),
                "long_range_threshold": float(
                    subset["gold_mean_visible_position"].dropna().quantile(LONG_RANGE_QUANTILE)
                )
                if subset["gold_mean_visible_position"].dropna().size
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V4 official-QA budget curves by auditable query strata.")
    parser.add_argument("--root", default=str(VERSION_ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    query_df = _collect_strata_rows(root)
    if query_df.empty:
        raise SystemExit("No formal V4 official-QA rows found for strata analysis.")

    query_csv = aggregate_dir / "v4_official_qa_query_strata.csv"
    strata_csv = aggregate_dir / "v4_official_qa_strata_summary.csv"
    population_csv = aggregate_dir / "v4_official_qa_strata_population.csv"
    strata_md = aggregate_dir / "v4_official_qa_strata_summary.md"

    query_df.to_csv(query_csv, index=False)
    strata_df = _summarize_query_strata(query_df)
    population_df = _summarize_strata_population(query_df)
    strata_df.to_csv(strata_csv, index=False)
    population_df.to_csv(population_csv, index=False)

    md_lines = [
        "# V4 Official-QA Strata Summary",
        "",
        "## Population",
        "",
        population_df.to_markdown(index=False),
        "",
        "## Budget Curves",
        "",
        strata_df.to_markdown(index=False),
        "",
    ]
    strata_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"query_strata_csv={query_csv}")
    print(f"population_csv={population_csv}")
    print(f"strata_summary_csv={strata_csv}")
    print(f"strata_summary_md={strata_md}")


if __name__ == "__main__":
    main()
