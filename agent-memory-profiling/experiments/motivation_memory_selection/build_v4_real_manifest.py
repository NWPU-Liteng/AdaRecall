#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from real_dataset_adapters import (
    build_locomo_bundle,
    build_longmemeval_bundle,
    build_msc_bundle,
    write_manifest_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_raw_data_path(dataset: str) -> Path:
    base = PROJECT_ROOT / "data" / "public_datasets"
    mapping = {
        "locomo": base / "locomo10.json",
        "longmemeval_s": base / "longmemeval" / "longmemeval_s_cleaned.json",
        "longmemeval_m": base / "longmemeval" / "longmemeval_m_cleaned.json",
        "msc": base / "msc" / "msc_v0.1.tar.gz",
    }
    try:
        return mapping[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def _write_provenance_report(output_dir: Path, dataset: str, raw_data_path: Path, bundle_counts: dict) -> Path:
    digest = hashlib.sha256(raw_data_path.read_bytes()).hexdigest()
    report_path = output_dir / "data_provenance_report.md"
    lines = [
        "# Data Provenance Report",
        "",
        f"- dataset: `{dataset}`",
        f"- raw_data_path: `{raw_data_path}`",
        f"- raw_data_sha256: `{digest}`",
        f"- runtime_events: `{bundle_counts['runtime_events']}`",
        f"- evaluation_labels: `{bundle_counts['evaluation_labels']}`",
        f"- memory_bank_rows: `{bundle_counts['memory_bank_rows']}`",
        "",
        "## Runtime Fields",
        "",
        "- `query_text`, `visible_context`, `memory_ids_available`: directly derived from original public dataset turns or official benchmark questions.",
        "- `tenant_id`, `timeline_id`, `session_id`, `speaker`, `event_time`: dataset-native identifiers or stable replay metadata derived from raw sessions.",
        "- `source_is_synthetic=false`: formal V4 runtime manifest excludes synthetic workload generation.",
        "",
        "## Evaluator-only Fields",
        "",
        "- `reference_answer`, `gold_turn_ids`, `gold_session_ids`, `official_question_type`: written only to `evaluation_labels.jsonl` and not imported by runtime profiling code.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V4 real-data manifests with runtime/evaluator isolation.")
    parser.add_argument("--dataset", required=True, choices=["locomo", "longmemeval_s", "longmemeval_m", "msc"])
    parser.add_argument("--raw-data", default="", help="Optional explicit raw dataset path.")
    parser.add_argument("--output", required=True, help="Output directory for manifest files.")
    parser.add_argument("--msc-limit-dialogs", type=int, default=512, help="Limit MSC dialogs for tractable profiling.")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    raw_data_path = Path(args.raw_data).resolve() if args.raw_data else _default_raw_data_path(args.dataset)
    memory_source_path = str(output_dir / "runtime_memory_bank.jsonl")

    if args.dataset == "locomo":
        bundle = build_locomo_bundle(raw_data_path, memory_source_path)
    elif args.dataset == "longmemeval_s":
        bundle = build_longmemeval_bundle(raw_data_path, "longmemeval_s", memory_source_path)
    elif args.dataset == "longmemeval_m":
        bundle = build_longmemeval_bundle(raw_data_path, "longmemeval_m", memory_source_path)
    else:
        bundle = build_msc_bundle(raw_data_path, memory_source_path, limit_dialogs=args.msc_limit_dialogs)

    paths = write_manifest_bundle(output_dir, bundle)
    provenance_path = _write_provenance_report(
        output_dir,
        args.dataset,
        raw_data_path,
        {
            "runtime_events": len(bundle.runtime_events),
            "evaluation_labels": len(bundle.evaluation_labels),
            "memory_bank_rows": len(bundle.memory_bank),
        },
    )

    print(f"runtime_manifest={paths['runtime_manifest']}")
    print(f"evaluation_labels={paths['evaluation_labels']}")
    print(f"memory_bank={paths['memory_bank']}")
    print(f"data_provenance_report={provenance_path}")


if __name__ == "__main__":
    main()
