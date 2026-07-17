#!/usr/bin/env python3
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


RUNTIME_ALLOWED_FIELDS = ["text", "time", "session_id", "speaker", "tenant_id"]


@dataclass
class ManifestBundle:
    runtime_events: List[dict]
    evaluation_labels: List[dict]
    memory_bank: List[dict]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_timestamp(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _memory_record(
    *,
    memory_id: str,
    dataset: str,
    tenant_id: str,
    timeline_id: str,
    session_id: str,
    event_time: str,
    speaker: str,
    text: str,
    role: str,
    source_path: str,
    extra: dict | None = None,
) -> dict:
    record = {
        "memory_id": memory_id,
        "dataset": dataset,
        "tenant_id": tenant_id,
        "timeline_id": timeline_id,
        "session_id": session_id,
        "event_time": event_time,
        "speaker": speaker,
        "role": role,
        "text": text,
        "source_path": source_path,
    }
    if extra:
        record.update(extra)
    return record


def _runtime_event(
    *,
    dataset: str,
    tenant_id: str,
    timeline_id: str,
    event_id: str,
    event_type: str,
    event_time: str,
    query_text: str,
    visible_memory_ids: Sequence[str],
    memory_source_path: str,
) -> dict:
    return {
        "dataset": dataset,
        "tenant_id": tenant_id,
        "timeline_id": timeline_id,
        "event_id": event_id,
        "event_type": event_type,
        "event_time": event_time,
        "query_text": query_text,
        "visible_context": list(visible_memory_ids),
        "memory_ids_available": list(visible_memory_ids),
        "memory_source_paths": [memory_source_path],
        "runtime_allowed_fields": list(RUNTIME_ALLOWED_FIELDS),
        "source_is_synthetic": False,
    }


def _evaluation_label(
    *,
    dataset: str,
    event_id: str,
    reference_answer: object,
    gold_session_ids: Sequence[str],
    gold_turn_ids: Sequence[str],
    official_question_type: object,
) -> dict:
    return {
        "dataset": dataset,
        "event_id": event_id,
        "reference_answer": reference_answer,
        "gold_session_ids": list(gold_session_ids),
        "gold_turn_ids": list(gold_turn_ids),
        "official_question_type": official_question_type,
    }


def _iter_locomo_samples(raw_data_path: Path) -> list[dict]:
    payload = json.loads(raw_data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("samples", "data", "dialogues", "conversations"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported LoCoMo payload format: {raw_data_path}")


def build_locomo_bundle(raw_data_path: Path, memory_source_path: str) -> ManifestBundle:
    runtime_events: List[dict] = []
    evaluation_labels: List[dict] = []
    memory_bank: List[dict] = []
    for sample_index, sample in enumerate(_iter_locomo_samples(raw_data_path), start=1):
        tenant_id = str(sample.get("sample_id") or f"locomo_sample_{sample_index:04d}")
        timeline_id = tenant_id
        conversation = sample.get("conversation") or {}
        session_keys = sorted(
            [key for key in conversation.keys() if key.startswith("session_") and not key.endswith("_date_time")],
            key=lambda item: int(item.split("_")[1]),
        )
        visible_ids: List[str] = []
        last_event_time = f"2023-01-{sample_index:02d}"
        for session_key in session_keys:
            turns = conversation.get(session_key) or []
            session_id = session_key
            event_time = _safe_timestamp(conversation.get(f"{session_key}_date_time"), f"2023-01-{sample_index:02d}")
            if not isinstance(turns, list):
                continue
            for turn_index, turn in enumerate(turns, start=1):
                if not isinstance(turn, dict):
                    continue
                text = str(turn.get("text") or "").strip()
                if not text:
                    continue
                speaker = str(turn.get("speaker") or "speaker")
                dia_id = str(turn.get("dia_id") or f"{session_id}:{turn_index}")
                memory_id = f"{tenant_id}::{dia_id}"
                memory_bank.append(
                    _memory_record(
                        memory_id=memory_id,
                        dataset="locomo",
                        tenant_id=tenant_id,
                        timeline_id=timeline_id,
                        session_id=session_id,
                        event_time=event_time,
                        speaker=speaker,
                        role=speaker,
                        text=text,
                        source_path=memory_source_path,
                        extra={"dia_id": dia_id},
                    )
                )
                runtime_events.append(
                    _runtime_event(
                        dataset="locomo",
                        tenant_id=tenant_id,
                        timeline_id=timeline_id,
                        event_id=f"{tenant_id}::replay::{dia_id}",
                        event_type="replay_turn",
                        event_time=event_time,
                        query_text=text,
                        visible_memory_ids=list(visible_ids),
                        memory_source_path=memory_source_path,
                    )
                )
                visible_ids.append(memory_id)
                last_event_time = event_time
        for qa_index, qa in enumerate(sample.get("qa") or [], start=1):
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            event_id = f"{tenant_id}::qa::{qa_index:04d}"
            runtime_events.append(
                _runtime_event(
                    dataset="locomo",
                    tenant_id=tenant_id,
                    timeline_id=timeline_id,
                    event_id=event_id,
                    event_type="official_qa",
                    event_time=last_event_time,
                    query_text=question,
                    visible_memory_ids=list(visible_ids),
                    memory_source_path=memory_source_path,
                )
            )
            evidence = [f"{tenant_id}::{item}" for item in (qa.get("evidence") or [])]
            evaluation_labels.append(
                _evaluation_label(
                    dataset="locomo",
                    event_id=event_id,
                    reference_answer=qa.get("answer"),
                    gold_session_ids=[tenant_id],
                    gold_turn_ids=evidence,
                    official_question_type=qa.get("category"),
                )
            )
    return ManifestBundle(runtime_events=runtime_events, evaluation_labels=evaluation_labels, memory_bank=memory_bank)


def _iter_longmemeval_items(raw_data_path: Path) -> list[dict]:
    payload = json.loads(raw_data_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "samples", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported LongMemEval payload format: {raw_data_path}")


def build_longmemeval_bundle(raw_data_path: Path, dataset_name: str, memory_source_path: str) -> ManifestBundle:
    runtime_events: List[dict] = []
    evaluation_labels: List[dict] = []
    memory_bank: List[dict] = []
    for item_index, item in enumerate(_iter_longmemeval_items(raw_data_path), start=1):
        tenant_id = str(item.get("question_id") or f"{dataset_name}_{item_index:06d}")
        timeline_id = tenant_id
        visible_ids: List[str] = []
        gold_turn_ids: List[str] = []
        session_ids = list(item.get("haystack_session_ids") or [])
        session_dates = list(item.get("haystack_dates") or [])
        haystack_sessions = list(item.get("haystack_sessions") or [])
        for session_index, turns in enumerate(haystack_sessions):
            session_id = str(session_ids[session_index] if session_index < len(session_ids) else f"session_{session_index:03d}")
            session_time = _safe_timestamp(session_dates[session_index] if session_index < len(session_dates) else "", f"2023-05-{(session_index % 28) + 1:02d}")
            if not isinstance(turns, list):
                continue
            for turn_index, turn in enumerate(turns, start=1):
                if not isinstance(turn, dict):
                    continue
                text = str(turn.get("content") or turn.get("text") or "").strip()
                if not text:
                    continue
                role = str(turn.get("role") or turn.get("speaker") or "speaker")
                memory_id = f"{tenant_id}::{session_id}::{turn_index:04d}"
                memory_bank.append(
                    _memory_record(
                        memory_id=memory_id,
                        dataset=dataset_name,
                        tenant_id=tenant_id,
                        timeline_id=timeline_id,
                        session_id=session_id,
                        event_time=session_time,
                        speaker=role,
                        role=role,
                        text=text,
                        source_path=memory_source_path,
                    )
                )
                if bool(turn.get("has_answer")):
                    gold_turn_ids.append(memory_id)
                if role.lower() == "user":
                    runtime_events.append(
                        _runtime_event(
                            dataset=dataset_name,
                            tenant_id=tenant_id,
                            timeline_id=timeline_id,
                            event_id=f"{tenant_id}::replay::{session_id}::{turn_index:04d}",
                            event_type="replay_turn",
                            event_time=session_time,
                            query_text=text,
                            visible_memory_ids=list(visible_ids),
                            memory_source_path=memory_source_path,
                        )
                    )
                visible_ids.append(memory_id)
        event_id = f"{tenant_id}::qa"
        runtime_events.append(
            _runtime_event(
                dataset=dataset_name,
                tenant_id=tenant_id,
                timeline_id=timeline_id,
                event_id=event_id,
                event_type="official_qa",
                event_time=_safe_timestamp(item.get("question_date"), f"2023-05-{(item_index % 28) + 1:02d}"),
                query_text=str(item.get("question") or "").strip(),
                visible_memory_ids=list(visible_ids),
                memory_source_path=memory_source_path,
            )
        )
        evaluation_labels.append(
            _evaluation_label(
                dataset=dataset_name,
                event_id=event_id,
                reference_answer=item.get("answer"),
                gold_session_ids=item.get("answer_session_ids") or [],
                gold_turn_ids=gold_turn_ids,
                official_question_type=item.get("question_type"),
            )
        )
    return ManifestBundle(runtime_events=runtime_events, evaluation_labels=evaluation_labels, memory_bank=memory_bank)


def _iter_msc_dialog_records(raw_data_path: Path) -> Iterator[tuple[str, dict]]:
    with tarfile.open(raw_data_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if not name.startswith("msc/msc_dialogue/session_") or not name.endswith(".txt"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            for line_index, raw_line in enumerate(handle, start=1):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                yield f"{name}#{line_index}", json.loads(line)


def build_msc_bundle(raw_data_path: Path, memory_source_path: str, limit_dialogs: int | None = None) -> ManifestBundle:
    runtime_events: List[dict] = []
    evaluation_labels: List[dict] = []
    memory_bank: List[dict] = []
    for dialog_index, (source_id, dialog_obj) in enumerate(_iter_msc_dialog_records(raw_data_path), start=1):
        if limit_dialogs is not None and dialog_index > limit_dialogs:
            break
        dialog = list(dialog_obj.get("dialog") or [])
        personas = list(dialog_obj.get("personas") or [])
        tenant_id = str(dialog[0].get("convai2_id") if dialog and isinstance(dialog[0], dict) else f"msc_dialog_{dialog_index:06d}")
        tenant_id = tenant_id.replace("/", "_")
        timeline_id = tenant_id
        visible_ids: List[str] = []
        for persona_group_index, persona_group in enumerate(personas, start=1):
            if not isinstance(persona_group, list):
                continue
            for persona_index, persona_text in enumerate(persona_group, start=1):
                text = str(persona_text).strip()
                if not text:
                    continue
                memory_id = f"{tenant_id}::persona::{persona_group_index:02d}::{persona_index:02d}"
                memory_bank.append(
                    _memory_record(
                        memory_id=memory_id,
                        dataset="msc",
                        tenant_id=tenant_id,
                        timeline_id=timeline_id,
                        session_id="persona",
                        event_time=f"2023-01-{(dialog_index % 28) + 1:02d}",
                        speaker="persona",
                        role="persona",
                        text=text,
                        source_path=memory_source_path,
                    )
                )
                visible_ids.append(memory_id)
        for turn_index, turn in enumerate(dialog, start=1):
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            speaker = str(turn.get("id") or "speaker")
            memory_id = f"{tenant_id}::dialog::{turn_index:04d}"
            event_time = f"2023-02-{(turn_index % 28) + 1:02d}"
            memory_bank.append(
                _memory_record(
                    memory_id=memory_id,
                    dataset="msc",
                    tenant_id=tenant_id,
                    timeline_id=timeline_id,
                    session_id="dialog",
                    event_time=event_time,
                    speaker=speaker,
                    role=speaker,
                    text=text,
                    source_path=memory_source_path,
                )
            )
            runtime_events.append(
                _runtime_event(
                    dataset="msc",
                    tenant_id=tenant_id,
                    timeline_id=timeline_id,
                    event_id=f"{tenant_id}::replay::{turn_index:04d}",
                    event_type="replay_turn",
                    event_time=event_time,
                    query_text=text,
                    visible_memory_ids=list(visible_ids),
                    memory_source_path=memory_source_path,
                )
            )
            visible_ids.append(memory_id)
    return ManifestBundle(runtime_events=runtime_events, evaluation_labels=evaluation_labels, memory_bank=memory_bank)


def write_manifest_bundle(output_dir: Path, bundle: ManifestBundle) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_manifest_path = output_dir / "runtime_manifest.jsonl"
    evaluation_labels_path = output_dir / "evaluation_labels.jsonl"
    memory_bank_path = output_dir / "runtime_memory_bank.jsonl"
    _write_jsonl(runtime_manifest_path, bundle.runtime_events)
    _write_jsonl(evaluation_labels_path, bundle.evaluation_labels)
    _write_jsonl(memory_bank_path, bundle.memory_bank)
    return {
        "runtime_manifest": runtime_manifest_path,
        "evaluation_labels": evaluation_labels_path,
        "memory_bank": memory_bank_path,
    }
