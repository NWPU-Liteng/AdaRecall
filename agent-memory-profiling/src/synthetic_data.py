import json
import random
from datetime import date, timedelta
from typing import Iterable, List


MEMORY_TYPES = [
    "health_constraint",
    "episodic",
    "feedback",
    "profile",
    "planning",
]
MODULE_TAGS = ["planning", "reflection", "risk_control", "profile_update"]
ENTITY_TAGS = [
    "knee",
    "sleep",
    "weight",
    "running",
    "diet",
    "fatigue",
    "hydration",
    "recovery",
]


def _sample_content(rng: random.Random, memory_type: str, entity: str, idx: int) -> str:
    templates = {
        "health_constraint": "User reported {entity} discomfort after training and should reduce intensity for the next 3 days.",
        "episodic": "Session {idx} summary: user mentioned {entity} changes after a morning workout and evening meal.",
        "feedback": "Coach feedback for record {idx}: adjust plan because {entity} trend suggests lower recovery tolerance.",
        "profile": "Long-term profile update {idx}: user baseline around {entity} should be considered in future training plans.",
        "planning": "Plan item {idx}: schedule a moderate routine focusing on {entity} stability, sleep, and nutrition.",
    }
    sentence = templates[memory_type].format(entity=entity, idx=idx)
    suffixes = [
        "Recent observations indicate mild fatigue.",
        "Prioritize consistency over intensity.",
        "Track the next two sessions closely.",
        "Avoid abrupt volume increases.",
        "This note may affect planning and risk control.",
    ]
    return f"{sentence} {rng.choice(suffixes)}"


def generate_memories(size: int, seed: int = 42) -> List[dict]:
    rng = random.Random(seed + size)
    base_date = date(2026, 3, 14)
    memories: List[dict] = []
    for idx in range(size):
        memory_type = MEMORY_TYPES[idx % len(MEMORY_TYPES)]
        primary_module = MODULE_TAGS[idx % len(MODULE_TAGS)]
        extra_module = MODULE_TAGS[(idx + rng.randint(1, 3)) % len(MODULE_TAGS)]
        modules = sorted({primary_module, extra_module})
        entity = ENTITY_TAGS[idx % len(ENTITY_TAGS)]
        extra_entity = ENTITY_TAGS[(idx + rng.randint(1, 5)) % len(ENTITY_TAGS)]
        day = base_date - timedelta(days=idx % 365)
        memories.append(
            {
                "memory_id": f"m_{idx + 1:06d}",
                "user_id": "u_001",
                "content": _sample_content(rng, memory_type, entity, idx + 1),
                "memory_type": memory_type,
                "module_tags": modules,
                "entity_tags": sorted({entity, extra_entity}),
                "time": day.isoformat(),
                "importance": round(rng.uniform(0.35, 0.99), 4),
            }
        )
    return memories


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
