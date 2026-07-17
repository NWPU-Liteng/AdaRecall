#!/usr/bin/env python3
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MEMORY_TYPES = ["episodic", "semantic", "procedural", "profile", "constraint"]
QUERY_TYPES = ["fact", "temporal", "relation", "plan_continuity", "constraint"]
TASK_CATALOG = [
    ("marathon_rebuild", ["running", "recovery", "fatigue"]),
    ("sleep_stability", ["sleep", "fatigue", "hydration"]),
    ("blood_pressure_guard", ["blood_pressure", "diet", "recovery"]),
    ("knee_rehab", ["knee", "running", "recovery"]),
    ("strength_progression", ["strength", "fatigue", "sleep"]),
    ("nutrition_cut", ["diet", "weight", "hydration"]),
]


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    query_type: str
    text: str
    task_tags: List[str]
    entity_tags: List[str]
    preferred_memory_types: List[str]
    temporal_window_days: int
    gold_group_ids: List[str]


def _memory_type_for_index(idx: int) -> str:
    return MEMORY_TYPES[idx % len(MEMORY_TYPES)]


def _choose_hot_offset(rng: random.Random) -> int:
    if rng.random() < 0.22:
        return rng.randint(0, 21)
    if rng.random() < 0.55:
        return rng.randint(22, 90)
    return rng.randint(91, 420)


def _memory_text(memory_type: str, task_tag: str, entity_tags: List[str], step_idx: int, group_id: str) -> Tuple[str, str]:
    primary = entity_tags[0]
    secondary = entity_tags[1] if len(entity_tags) > 1 else entity_tags[0]
    templates = {
        "episodic": (
            f"Episode {step_idx} for {task_tag}: the user reported {primary} changes after training, and {secondary} remained relevant.",
            f"{task_tag} event {step_idx} with {primary} and {secondary}.",
        ),
        "semantic": (
            f"Semantic note for {task_tag}: patterns involving {primary} and {secondary} usually predict the next adjustment.",
            f"{task_tag} pattern note about {primary}.",
        ),
        "procedural": (
            f"Procedure update for {task_tag}: resume the plan by checking {primary}, then tune workload using {secondary}.",
            f"{task_tag} recovery procedure using {primary}.",
        ),
        "profile": (
            f"Long-term profile for {task_tag}: the user baseline links {primary} with {secondary}, and stable progress depends on both.",
            f"{task_tag} profile baseline for {primary}.",
        ),
        "constraint": (
            f"Constraint for {task_tag}: when {primary} worsens, keep the next step conservative and verify {secondary} before adding load.",
            f"{task_tag} safety constraint with {primary}.",
        ),
    }
    text, summary = templates[memory_type]
    return f"{text} Group {group_id}.", summary


def generate_memories(size: int, seed: int = 42) -> List[dict]:
    rng = random.Random(seed + size)
    base_date = date(2026, 6, 1)
    group_count = max(40, min(size // 18, 2000))
    groups = []
    for group_idx in range(group_count):
        task_tag, entity_pool = TASK_CATALOG[group_idx % len(TASK_CATALOG)]
        group_id = f"group_{group_idx + 1:05d}"
        target_size = max(8, min(26, size // group_count + rng.randint(-2, 4)))
        groups.append(
            {
                "group_id": group_id,
                "task_tag": task_tag,
                "entity_pool": entity_pool,
                "target_size": target_size,
            }
        )

    memories: List[dict] = []
    group_members: Dict[str, List[str]] = defaultdict(list)
    for idx in range(size):
        group = groups[idx % len(groups)]
        group_id = group["group_id"]
        task_tag = group["task_tag"]
        entity_pool = group["entity_pool"]
        primary = entity_pool[idx % len(entity_pool)]
        secondary = entity_pool[(idx + 1) % len(entity_pool)]
        if rng.random() < 0.35:
            secondary = rng.choice(entity_pool)
        entity_tags = sorted({primary, secondary})
        memory_type = _memory_type_for_index(idx)
        offset_days = _choose_hot_offset(rng)
        timestamp = (base_date - timedelta(days=offset_days)).isoformat()
        importance = round(0.35 + 0.64 * rng.random(), 4)
        recency = round(max(0.0, 1.0 - offset_days / 420.0), 4)
        memory_id = f"m_{idx + 1:07d}"
        text, summary = _memory_text(memory_type, task_tag, entity_tags, len(group_members[group_id]) + 1, group_id)

        relations: List[str] = []
        existing = group_members[group_id]
        if existing:
            relations.append(existing[-1])
        if len(existing) > 1:
            relations.append(existing[-2])
        if len(existing) > 3 and rng.random() < 0.7:
            relations.append(existing[0])

        # Add sparse cross-group links that share the primary entity to make
        # relation-heavy queries produce wider expansions at larger scales.
        if idx >= len(groups) and rng.random() < 0.25:
            linked_group = groups[(idx + len(entity_tags)) % len(groups)]["group_id"]
            linked_members = group_members.get(linked_group, [])
            if linked_members:
                relations.append(linked_members[min(len(linked_members) - 1, rng.randint(0, 2))])

        memory = {
            "memory_id": memory_id,
            "text": text,
            "summary": summary,
            "timestamp": timestamp,
            "memory_type": memory_type,
            "task_tags": [task_tag],
            "entity_tags": entity_tags,
            "importance": importance,
            "recency": recency,
            "relations": sorted(set(relations)),
            "gold_group_id": group_id,
        }
        memories.append(memory)
        group_members[group_id].append(memory_id)

    # Make relations bidirectional for traversal.
    id_to_memory = {memory["memory_id"]: memory for memory in memories}
    for memory in memories:
        for related_id in list(memory["relations"]):
            related = id_to_memory.get(related_id)
            if related is None:
                continue
            if memory["memory_id"] not in related["relations"]:
                related["relations"].append(memory["memory_id"])
                related["relations"].sort()

    return memories


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_or_generate_memories(path: Path, size: int, seed: int = 42) -> List[dict]:
    if path.exists():
        return read_jsonl(path)
    memories = generate_memories(size=size, seed=seed)
    write_jsonl(path, memories)
    return memories


def build_query_workload(memories: List[dict], query_type: str, num_queries: int, seed: int = 42) -> List[QuerySpec]:
    rng = random.Random(seed * 17 + len(memories) + len(query_type))
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for memory in memories:
        grouped[memory["gold_group_id"]].append(memory)

    groups = list(grouped.items())
    if not groups:
        return []

    type_preferences = {
        "fact": ["profile", "semantic"],
        "temporal": ["episodic", "procedural"],
        "relation": ["episodic", "semantic", "constraint"],
        "plan_continuity": ["procedural", "profile", "episodic"],
        "constraint": ["constraint", "profile", "episodic"],
    }
    windows = {
        "fact": 365,
        "temporal": 21,
        "relation": 120,
        "plan_continuity": 90,
        "constraint": 180,
    }
    prompts = {
        "fact": "What long-term preference should I remember about {task} when {entity} comes up again?",
        "temporal": "How has the user's status around {entity} changed recently while working on {task}?",
        "relation": "Which linked factors most likely caused the recent {entity} decline during {task}?",
        "plan_continuity": "The user paused {task}. How should I resume the plan while accounting for {entity} and previous progress?",
        "constraint": "Is it safe to push harder on {task} when {entity} is elevated, and which constraints matter most?",
    }

    selected = []
    sample_groups = groups if len(groups) <= num_queries else rng.sample(groups, num_queries)
    for idx, (group_id, rows) in enumerate(sample_groups, start=1):
        anchor = rng.choice(rows)
        task_tag = anchor["task_tags"][0]
        entity_pool = anchor.get("entity_tags") or [task_tag]
        entity = rng.choice(entity_pool)
        selected.append(
            QuerySpec(
                query_id=f"{query_type}_{idx:04d}",
                query_type=query_type,
                text=prompts[query_type].format(task=task_tag, entity=entity),
                task_tags=[task_tag],
                entity_tags=[entity],
                preferred_memory_types=type_preferences[query_type],
                temporal_window_days=windows[query_type],
                gold_group_ids=[group_id],
            )
        )
    while len(selected) < num_queries:
        selected.extend(selected[: max(1, num_queries - len(selected))])
    return selected[:num_queries]
