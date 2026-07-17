#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from synthetic_memory_generator import write_jsonl


ENTITY_HINTS = {
    "knee",
    "sleep",
    "weight",
    "running",
    "diet",
    "fatigue",
    "hydration",
    "recovery",
    "blood",
    "pressure",
    "medication",
    "injury",
    "work",
    "travel",
    "family",
    "study",
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _entity_tags_from_text(text: str) -> List[str]:
    lowered = text.lower()
    tags = [token for token in sorted(ENTITY_HINTS) if token in lowered]
    if tags:
        return tags[:6]
    tokens = re.findall(r"[a-zA-Z]{4,}", lowered)
    return sorted(set(tokens[:4]))


def _task_tags_from_text(text: str, fallback: str) -> List[str]:
    lowered = text.lower()
    tags = []
    for keyword in ["plan", "schedule", "goal", "health", "risk", "work", "travel", "family", "study"]:
        if keyword in lowered:
            tags.append(keyword)
    return tags[:4] or [fallback]


def _importance_from_turn(turn_index: int, total_turns: int) -> float:
    if total_turns <= 1:
        return 0.65
    return round(0.45 + 0.45 * (turn_index / max(total_turns - 1, 1)), 4)


def _recency_from_turn(turn_index: int, total_turns: int) -> float:
    if total_turns <= 1:
        return 1.0
    return round(0.25 + 0.75 * (turn_index / max(total_turns - 1, 1)), 4)


def _to_timestamp(base_prefix: str, idx: int) -> str:
    month = 1 + ((idx // 28) % 12)
    day = 1 + (idx % 28)
    return f"{base_prefix}-{month:02d}-{day:02d}"


def _locomo_dialogues(payload) -> List[dict]:
    if isinstance(payload, dict):
        for key in ["conversations", "dialogues", "data", "samples"]:
            if isinstance(payload.get(key), list):
                return payload[key]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported LoCoMo payload format")


def convert_locomo(input_path: Path) -> List[dict]:
    payload = _read_json(input_path)
    dialogues = _locomo_dialogues(payload)
    memories: List[dict] = []
    memory_counter = 1
    for dialogue_index, dialogue in enumerate(dialogues, start=1):
        conv_id = (
            dialogue.get("conversation_id")
            or dialogue.get("dialogue_id")
            or dialogue.get("id")
            or f"locomo_conv_{dialogue_index:04d}"
        )
        conversation = dialogue.get("conversation") or {}
        speaker_a = str(conversation.get("speaker_a") or "speaker_a")
        speaker_b = str(conversation.get("speaker_b") or "speaker_b")
        previous_memory_id = None
        session_keys = sorted(
            [key for key in conversation.keys() if key.startswith("session_") and not key.endswith("_date_time")],
            key=lambda key: int(key.split("_")[1]),
        )
        total_turns = 0
        for key in session_keys:
            turns = conversation.get(key) or []
            if isinstance(turns, list):
                total_turns += len(turns)

        running_turn_index = 0
        for key in session_keys:
            turns = conversation.get(key) or []
            session_date = conversation.get(f"{key}_date_time") or _to_timestamp("2026", memory_counter)
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if isinstance(turn, dict):
                    raw_speaker = str(turn.get("speaker") or turn.get("role") or turn.get("participant") or "")
                    speaker = raw_speaker or speaker_a
                    text = str(turn.get("text") or turn.get("utterance") or turn.get("content") or "")
                elif isinstance(turn, list) and len(turn) >= 2:
                    raw_speaker = str(turn[0])
                    speaker = speaker_a if raw_speaker in {"speaker_a", "A"} else speaker_b if raw_speaker in {"speaker_b", "B"} else raw_speaker
                    text = str(turn[1])
                else:
                    speaker = speaker_a if running_turn_index % 2 == 0 else speaker_b
                    text = str(turn)
                speaker = {"speaker_a": speaker_a, "speaker_b": speaker_b}.get(speaker, speaker)
                if not text.strip():
                    continue
                memory_id = f"locomo_{memory_counter:07d}"
                relations = [previous_memory_id] if previous_memory_id else []
                memory = {
                    "memory_id": memory_id,
                    "text": f"{speaker}: {text}",
                    "summary": text[:160],
                    "timestamp": str(session_date),
                    "memory_type": "episodic",
                    "task_tags": _task_tags_from_text(text, "locomo_dialogue"),
                    "entity_tags": _entity_tags_from_text(text),
                    "importance": _importance_from_turn(running_turn_index, max(total_turns, 1)),
                    "recency": _recency_from_turn(running_turn_index, max(total_turns, 1)),
                    "relations": relations,
                    "gold_group_id": str(conv_id),
                }
                memories.append(memory)
                previous_memory_id = memory_id
                memory_counter += 1
                running_turn_index += 1
    return memories


def _msc_sessions(payload) -> List[dict]:
    if isinstance(payload, dict):
        for key in ["data", "dialogs", "dialogues", "episodes", "samples"]:
            if isinstance(payload.get(key), list):
                return payload[key]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported MSC payload format")


def convert_msc_from_export(input_path: Path) -> List[dict]:
    payload = _read_json(input_path) if input_path.suffix.lower() == ".json" else _read_jsonl(input_path)
    sessions = _msc_sessions(payload)
    memories: List[dict] = []
    memory_counter = 1
    for session_index, session in enumerate(sessions, start=1):
        session_id = (
            session.get("session_id")
            or session.get("dialog_id")
            or session.get("dialogue_id")
            or session.get("id")
            or f"msc_session_{session_index:05d}"
        )
        turns = (
            session.get("dialog")
            or session.get("dialogue")
            or session.get("messages")
            or session.get("turns")
            or []
        )
        if not isinstance(turns, list):
            continue
        previous_memory_id = None
        total_turns = len(turns)
        for turn_index, turn in enumerate(turns, start=1):
            if isinstance(turn, dict):
                speaker = str(turn.get("speaker") or turn.get("role") or turn.get("id") or "speaker")
                text = str(turn.get("text") or turn.get("utterance") or turn.get("content") or turn.get("label") or "")
            else:
                speaker = "speaker"
                text = str(turn)
            if not text.strip():
                continue
            memory_id = f"msc_{memory_counter:07d}"
            relations = [previous_memory_id] if previous_memory_id else []
            memory = {
                "memory_id": memory_id,
                "text": f"{speaker}: {text}",
                "summary": text[:160],
                "timestamp": _to_timestamp("2026", memory_counter),
                "memory_type": "episodic",
                "task_tags": _task_tags_from_text(text, "msc_session"),
                "entity_tags": _entity_tags_from_text(text),
                "importance": _importance_from_turn(turn_index - 1, total_turns),
                "recency": _recency_from_turn(turn_index - 1, total_turns),
                "relations": relations,
                "gold_group_id": str(session_id),
            }
            memories.append(memory)
            previous_memory_id = memory_id
            memory_counter += 1
    return memories


def _append_msc_dialog_memories(memories: List[dict], dialog_obj: dict, session_name: str, conv_id: str, memory_counter: int) -> int:
    dialog = dialog_obj.get("dialog") or []
    personas = dialog_obj.get("personas") or []
    init_personas = dialog_obj.get("init_personas") or []

    session_group_id = f"{session_name}:{conv_id}"
    previous_memory_id = None
    total_turns = len(dialog)

    # Persist persona/profile-like memories first.
    for persona_group_index, persona_group in enumerate(personas or init_personas, start=1):
        if not isinstance(persona_group, list):
            continue
        for persona_index, persona_text in enumerate(persona_group, start=1):
            text = str(persona_text).strip()
            if not text:
                continue
            memory_id = f"msc_{memory_counter:07d}"
            memories.append(
                {
                    "memory_id": memory_id,
                    "text": f"Persona {persona_group_index}: {text}",
                    "summary": text[:160],
                    "timestamp": _to_timestamp("2026", memory_counter),
                    "memory_type": "profile",
                    "task_tags": ["msc_persona"],
                    "entity_tags": _entity_tags_from_text(text),
                    "importance": 0.82,
                    "recency": 0.45,
                    "relations": [previous_memory_id] if previous_memory_id else [],
                    "gold_group_id": session_group_id,
                }
            )
            previous_memory_id = memory_id
            memory_counter += 1

    for turn_index, turn in enumerate(dialog, start=1):
        if not isinstance(turn, dict):
            continue
        text = str(turn.get("text") or turn.get("content") or "").strip()
        speaker = str(turn.get("id") or turn.get("speaker") or f"speaker_{turn_index % 2 + 1}")
        if not text:
            continue
        memory_id = f"msc_{memory_counter:07d}"
        memories.append(
            {
                "memory_id": memory_id,
                "text": f"{speaker}: {text}",
                "summary": text[:160],
                "timestamp": _to_timestamp("2026", memory_counter),
                "memory_type": "episodic",
                "task_tags": _task_tags_from_text(text, "msc_dialogue"),
                "entity_tags": _entity_tags_from_text(text),
                "importance": _importance_from_turn(turn_index - 1, max(total_turns, 1)),
                "recency": _recency_from_turn(turn_index - 1, max(total_turns, 1)),
                "relations": [previous_memory_id] if previous_memory_id else [],
                "gold_group_id": session_group_id,
            }
        )
        previous_memory_id = memory_id
        memory_counter += 1
    return memory_counter


def convert_msc_raw(root_path: Path) -> List[dict]:
    root = root_path.resolve()
    if root.is_file():
        return convert_msc_from_export(root)

    dialogue_root = root / "msc_dialogue"
    if not dialogue_root.exists():
        raise ValueError(f"MSC raw root missing msc_dialogue/: {root}")

    memories: List[dict] = []
    memory_counter = 1

    persona_path = dialogue_root / "init_persona_all.json"
    if persona_path.exists():
        persona_obj = _read_json(persona_path)
        for conv_id, dialog_obj in persona_obj.items():
            if isinstance(dialog_obj, dict):
                memory_counter = _append_msc_dialog_memories(memories, dialog_obj, "init_persona_all", conv_id, memory_counter)

    for session_dir in sorted(dialogue_root.glob("session_*")):
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name
        for split_name in ["train.txt", "valid.txt", "test.txt"]:
            split_path = session_dir / split_name
            if not split_path.exists():
                continue
            for row_index, row in enumerate(_read_jsonl(split_path), start=1):
                conv_id = str(
                    row.get("dialog_id")
                    or row.get("convai2_id")
                    or row.get("id")
                    or f"{session_name}:{split_name}:{row_index:07d}"
                )
                memory_counter = _append_msc_dialog_memories(memories, row, session_name, conv_id, memory_counter)
    return memories


def convert_longmemeval(input_path: Path) -> List[dict]:
    payload = _read_json(input_path) if input_path.suffix.lower() == ".json" else _read_jsonl(input_path)
    records = payload if isinstance(payload, list) else payload.get("data") or payload.get("examples") or []
    memories: List[dict] = []
    counter = 1
    for item_index, item in enumerate(records, start=1):
        sample_id = item.get("id") or item.get("sample_id") or f"longmemeval_{item_index:05d}"
        history = (
            item.get("history")
            or item.get("memory")
            or item.get("conversation")
            or item.get("messages")
            or item.get("haystack_sessions")
            or []
        )
        if isinstance(history, str):
            history = [history]
        previous_memory_id = None
        total_turns = len(history)
        for turn_index, entry in enumerate(history, start=1):
            if isinstance(entry, dict):
                speaker = str(entry.get("role") or entry.get("speaker") or "speaker")
                text = str(entry.get("text") or entry.get("content") or entry.get("message") or "")
                timestamp = str(entry.get("timestamp") or entry.get("time") or _to_timestamp("2026", counter))
            else:
                speaker = "history"
                text = str(entry)
                if isinstance(item.get("haystack_dates"), list) and turn_index - 1 < len(item["haystack_dates"]):
                    timestamp = str(item["haystack_dates"][turn_index - 1])
                else:
                    timestamp = _to_timestamp("2026", counter)
            if not text.strip():
                continue
            memory_id = f"longmemeval_{counter:07d}"
            relations = [previous_memory_id] if previous_memory_id else []
            memories.append(
                {
                    "memory_id": memory_id,
                    "text": f"{speaker}: {text}",
                    "summary": text[:160],
                    "timestamp": timestamp,
                    "memory_type": "episodic",
                    "task_tags": _task_tags_from_text(text, "longmemeval_sample"),
                    "entity_tags": _entity_tags_from_text(text),
                    "importance": _importance_from_turn(turn_index - 1, total_turns),
                    "recency": _recency_from_turn(turn_index - 1, total_turns),
                    "relations": relations,
                    "gold_group_id": str(sample_id),
                }
            )
            previous_memory_id = memory_id
            counter += 1
    return memories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["locomo", "msc", "longmemeval"], required=True)
    parser.add_argument("--msc-format", choices=["auto", "raw", "export"], default="auto")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if args.dataset == "locomo":
        memories = convert_locomo(input_path)
    elif args.dataset == "msc":
        if args.msc_format == "raw":
            memories = convert_msc_raw(input_path)
        elif args.msc_format == "export":
            memories = convert_msc_from_export(input_path)
        else:
            if input_path.is_dir():
                memories = convert_msc_raw(input_path)
            else:
                memories = convert_msc_from_export(input_path)
    else:
        memories = convert_longmemeval(input_path)

    write_jsonl(output_path, memories)
    print(f"dataset={args.dataset}")
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"memories={len(memories)}")


if __name__ == "__main__":
    main()
