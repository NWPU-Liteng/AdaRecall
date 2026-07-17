#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medmmb_common import (
    VERSION_ROOT,
    first_present,
    infer_persona_id,
    infer_query_id,
    infer_query_text,
    infer_query_type,
    infer_session_id,
    infer_turn_id,
    parse_maybe_json,
    resolve_path,
    session_sort_key,
    utc_now,
    write_csv,
    write_jsonl,
)


def _read_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, list):
                        return pd.DataFrame(value)
                return pd.json_normalize(payload)
            raise
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _language_match(path: Path, language: str) -> bool:
    text = str(path).lower()
    if f"/{language.lower()}/" in text or f"_{language.lower()}." in text or f"-{language.lower()}." in text:
        return True
    if language.lower() == "en" and ("english" in text or "/en_" in text):
        return True
    return False


def _candidate_files(data_root: Path, language: str) -> list[Path]:
    patterns = ["*.parquet", "*.json", "*.jsonl", "*.ndjson", "*.csv"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(data_root.rglob(pattern))
    lang_matches = [path for path in files if _language_match(path, language)]
    return sorted(lang_matches or files)


def _column_examples(series, limit: int = 3) -> str:
    examples = []
    for value in series.dropna().head(50).tolist():
        parsed = parse_maybe_json(value)
        text = json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, (dict, list)) else str(parsed)
        text = " ".join(text.split())
        if text and text not in examples:
            examples.append(text[:160])
        if len(examples) >= limit:
            break
    return " | ".join(examples)


def _classify_file(path: Path, columns: list[str]) -> str:
    lowered = {col.lower() for col in columns}
    name = path.name.lower()
    if "query" in name or {"query", "question", "question_text"} & lowered:
        return "queries"
    if "dialogues_with_noise" in name:
        return "dialogues_with_noise"
    if "dialog" in name or "dialogue" in name or {"role", "speaker", "utterance", "message"} & lowered:
        return "dialogues"
    if "noise" in name:
        return "dialogues_with_noise"
    return "other"


def _extract_queries(file_frames: list[tuple[Path, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for path, df in file_frames:
        kind = _classify_file(path, list(df.columns))
        if kind != "queries":
            continue
        for idx, record in enumerate(df.to_dict(orient="records")):
            query_text = infer_query_text(record)
            if not query_text:
                continue
            query_id = infer_query_id(record, idx)
            if query_id in seen:
                query_id = f"{path.stem}_{query_id}_{idx}"
            seen.add(query_id)
            row = {
                "query_id": query_id,
                "dataset": "medmemorybench_en",
                "persona_id": infer_persona_id(record),
                "query_session_id": infer_session_id(record),
                "query_type": infer_query_type(record),
                "query_text": query_text,
                "answers": parse_maybe_json(first_present(record, ["answers", "answer", "reference_answer", "gold_answer"], None)),
                "source_key_points": parse_maybe_json(first_present(record, ["source_key_points", "source_key_point", "source_knowledge_points"], None)),
                "knowledge_points": parse_maybe_json(first_present(record, ["knowledge_points", "key_points"], None)),
                "event_info": parse_maybe_json(first_present(record, ["event_info", "event", "metadata"], None)),
                "source_file": str(path),
                "source_row_index": idx,
            }
            rows.append(row)
    return rows


def _write_split(queries: list[dict[str, Any]], output_root: Path) -> None:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        by_type[str(row["query_type"])].append(row)
    for values in by_type.values():
        values.sort(key=lambda row: (str(row["persona_id"]), session_sort_key(row["query_session_id"]), row["query_id"]))
    ordered: list[dict[str, Any]] = []
    while any(by_type.values()):
        for key in sorted(by_type):
            if by_type[key]:
                ordered.append(by_type[key].pop(0))
    def take(n: int) -> list[dict[str, Any]]:
        return ordered[: min(n, len(ordered))]
    split = {
        "smoke": take(24),
        "pilot": take(72),
        "formal": take(240),
        "formal_o1": take(120),
        "seed": 20260704,
        "sampling_note": "Deterministic stratified round-robin by native query_type before formal results are observed.",
    }
    path = output_root / "configs" / "query_split_medmemorybench_en_v1.json"
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe official MedMemoryBench English schema and write audit artifacts.")
    parser.add_argument("--data-root", required=True, help="Official MedMemoryBench local data root.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root, VERSION_ROOT)
    output_root = resolve_path(args.output_root, VERSION_ROOT)
    if not data_root.exists():
        raise FileNotFoundError(f"Official MedMemoryBench data root not found: {data_root}")

    files = _candidate_files(data_root, args.language)
    if not files:
        raise FileNotFoundError(f"No parquet/json/jsonl/csv files found under {data_root}")

    audit_rows: list[dict[str, Any]] = []
    file_frames = []
    stage_like_fields: set[str] = set()
    for path in files:
        try:
            df = _read_table(path)
        except Exception as exc:
            audit_rows.append({"file": str(path), "error": str(exc)})
            continue
        file_frames.append((path, df))
        kind = _classify_file(path, list(df.columns))
        for col in df.columns:
            lowered = str(col).lower()
            if any(token in lowered for token in ["plan", "reason", "reflect", "phase", "stage"]):
                stage_like_fields.add(str(col))
            series = df[col]
            audit_rows.append(
                {
                    "file": str(path),
                    "kind": kind,
                    "rows": len(df),
                    "column": str(col),
                    "dtype": str(series.dtype),
                    "null_rate": float(series.isna().mean()) if len(series) else 0.0,
                    "examples": _column_examples(series),
                    "error": "",
                }
            )

    queries = _extract_queries(file_frames)
    write_jsonl(output_root / "aggregate" / "queries_medmemorybench_en.jsonl", queries)
    _write_split(queries, output_root)

    stats = {
        "created_utc": utc_now(),
        "data_root": str(data_root),
        "language": args.language,
        "n_files": len(files),
        "n_loaded_files": len(file_frames),
        "n_queries": len(queries),
        "query_types": Counter(str(row["query_type"]) for row in queries),
        "n_personas": len({str(row["persona_id"]) for row in queries}),
        "stage_like_fields": sorted(stage_like_fields),
        "dialogue_files": [str(path) for path, df in file_frames if _classify_file(path, list(df.columns)) == "dialogues"],
        "query_files": [str(path) for path, df in file_frames if _classify_file(path, list(df.columns)) == "queries"],
        "dialogues_with_noise_available": any("noise" in str(path).lower() for path, _ in file_frames),
    }
    (output_root / "aggregate").mkdir(parents=True, exist_ok=True)
    (output_root / "aggregate" / "medmemorybench_schema_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "aggregate" / "medmemorybench_schema_audit.csv", audit_rows)

    lines = [
        "# MedMemoryBench Schema Audit",
        "",
        f"Created UTC: {stats['created_utc']}",
        f"Data root: `{data_root}`",
        f"Language requested: `{args.language}`",
        f"Loaded files: {len(file_frames)} / {len(files)}",
        f"Queries extracted: {len(queries)}",
        f"Personas in extracted queries: {stats['n_personas']}",
        f"dialogues_with_noise available: {stats['dialogues_with_noise_available']}",
        "",
        "## Query Types",
        "",
    ]
    for key, count in sorted(stats["query_types"].items()):
        lines.append(f"- `{key}`: {count}")
    lines.extend(["", "## Stage-Like Fields", ""])
    if stage_like_fields:
        lines.append("Stage-like fields were observed and must not be treated as source-provided execution-stage labels for O1:")
        for field in sorted(stage_like_fields):
            lines.append(f"- `{field}`")
    else:
        lines.append("No source field named `plan`, `reason`, `reflect`, `phase`, `stage`, or equivalent was observed in loaded columns.")
    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `aggregate/medmemorybench_schema_audit.csv`",
            "- `aggregate/medmemorybench_schema_stats.json`",
            "- `aggregate/queries_medmemorybench_en.jsonl`",
            "- `configs/query_split_medmemorybench_en_v1.json`",
            "",
            "This report must be reviewed before formal runs.",
        ]
    )
    report = output_root / "reports" / "01_medmemorybench_schema_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"schema_audit={output_root / 'aggregate' / 'medmemorybench_schema_audit.csv'}")
    print(f"queries={output_root / 'aggregate' / 'queries_medmemorybench_en.jsonl'}")
    print(f"split={output_root / 'configs' / 'query_split_medmemorybench_en_v1.json'}")


if __name__ == "__main__":
    main()
