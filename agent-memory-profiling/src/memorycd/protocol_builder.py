from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .loader import iter_user_records, safe_text, split_last_n
from .schema import DOMAINS, TASKS, MemoryCDMemory, MemoryCDTaskEvent


def _memory_id(user_id: str, domain: str, inter: dict[str, Any], ordinal: int) -> str:
    asin = safe_text(inter.get("parent_asin") or inter.get("asin"))
    ts = int(inter.get("timestamp") or 0)
    return f"{user_id}|{domain}|{ts}|{asin}|{ordinal}"


def format_memory_text(domain: str, inter: dict[str, Any]) -> str:
    parts = [
        f"[Domain] {domain}",
        f"[Item] {safe_text(inter.get('parent_asin') or inter.get('asin'))}",
    ]
    if inter.get("rating") is not None:
        parts.append(f"[Rating] {inter.get('rating')}/5")
    title = safe_text(inter.get("title"), 300)
    text = safe_text(inter.get("text"), 1200)
    if title:
        parts.append(f"[Review Title] {title}")
    if text:
        parts.append(f"[Review] {text}")
    return "\n".join(parts)


def interaction_to_memory(user_id: str, domain: str, inter: dict[str, Any], ordinal: int) -> MemoryCDMemory:
    text = format_memory_text(domain, inter)
    return MemoryCDMemory(
        memory_id=_memory_id(user_id, domain, inter, ordinal),
        tenant_id=user_id,
        domain=domain,
        timestamp=int(inter.get("timestamp") or 0),
        parent_asin=safe_text(inter.get("parent_asin") or inter.get("asin")),
        rating=float(inter["rating"]) if inter.get("rating") is not None else None,
        review_title=safe_text(inter.get("title")),
        review_text=safe_text(inter.get("text")),
        memory_text=text,
        text_bytes=len(text.encode("utf-8")),
    )


def _target_item_text(inter: dict[str, Any], task: str, include_review: bool) -> str:
    parts = [
        f"Target item: {safe_text(inter.get('parent_asin') or inter.get('asin'))}",
    ]
    title = safe_text(inter.get("title"), 300)
    if task != "review_summarization" and title:
        # Target review title is hidden for all tasks except the evaluator side.
        pass
    if task in {"review_generation", "review_summarization"} and inter.get("rating") is not None:
        parts.append(f"Target rating: {inter.get('rating')}/5")
    if include_review:
        parts.append(f"Target review text: {safe_text(inter.get('text'), 1600)}")
    return "\n".join(parts)


def build_query_text(task: str, inter: dict[str, Any], candidate_asins: list[str] | None = None) -> str:
    if task == "rating_prediction":
        return (
            "Predict the user's rating from 1 to 5 for the target item.\n"
            + _target_item_text(inter, task, include_review=False)
        )
    if task == "item_ranking":
        candidates = ", ".join(candidate_asins or [])
        return (
            "Rank all candidate item ASINs from most to least relevant for this user.\n"
            f"Candidate ASINs: {candidates}"
        )
    if task == "review_generation":
        return (
            "Generate the user's review body for the target item.\n"
            + _target_item_text(inter, task, include_review=False)
        )
    if task == "review_summarization":
        return (
            "Generate a 3-10 word review title for the target review.\n"
            + _target_item_text(inter, task, include_review=True)
        )
    raise ValueError(f"Unsupported task: {task}")


def _event_id(protocol: str, task: str, setting: str, user_id: str, inter: dict[str, Any], test_ordinal: int) -> str:
    raw = "|".join(
        [
            protocol,
            task,
            setting,
            user_id,
            str(inter.get("timestamp") or 0),
            safe_text(inter.get("parent_asin") or inter.get("asin")),
            str(test_ordinal),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _label_for_task(task: str, inter: dict[str, Any], candidates: list[str]) -> dict[str, Any]:
    label = {"event_id": None, "task": task}
    if task == "rating_prediction":
        label["gold_rating"] = inter.get("rating")
    elif task == "item_ranking":
        label["positive_parent_asin"] = inter.get("parent_asin") or inter.get("asin")
        label["candidate_asins"] = candidates
    elif task == "review_generation":
        label["gold_review_text"] = inter.get("text")
    elif task == "review_summarization":
        label["gold_review_title"] = inter.get("title")
    return label


def build_manifests(
    *,
    users_path: Path,
    output_root: Path,
    protocol: str,
    max_users: int | None = None,
    num_test: int = 3,
) -> dict[str, Any]:
    if protocol not in {"official_single", "official_cross", "causal_cross"}:
        raise ValueError(f"Unsupported protocol: {protocol}")

    manifest_dir = output_root / "manifests" / f"memorycd_{protocol}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = manifest_dir / "runtime_manifest.jsonl"
    memory_path = manifest_dir / "memory_bank.jsonl"
    labels_path = manifest_dir / "evaluation_labels.jsonl"
    subset_path = manifest_dir / "task_curve_subset_event_ids.jsonl"

    event_count = 0
    memory_count = 0
    users_seen = 0
    settings: set[str] = set()
    subset_rows: list[dict[str, str]] = []

    with runtime_path.open("w", encoding="utf-8") as runtime_out, memory_path.open("w", encoding="utf-8") as memory_out, labels_path.open("w", encoding="utf-8") as labels_out:
        for record in iter_user_records(users_path):
            user_id = str(record.get("user_id") or "")
            if not user_id:
                continue
            users_seen += 1
            if max_users is not None and users_seen > max_users:
                break
            by_domain = record.get("interactions") or {}

            all_memories: dict[str, MemoryCDMemory] = {}
            memory_id_by_object: dict[tuple[str, int], str] = {}
            for domain in DOMAINS:
                ordered_domain = sorted(by_domain.get(domain, []), key=lambda x: int(x.get("timestamp") or 0))
                for ordinal, inter in enumerate(ordered_domain):
                    mem = interaction_to_memory(user_id, domain, inter, ordinal)
                    all_memories[mem.memory_id] = mem
                    memory_id_by_object[(domain, id(inter))] = mem.memory_id
                    memory_out.write(json.dumps(mem.to_runtime_dict(), ensure_ascii=False) + "\n")
                    memory_count += 1

            for target_domain in DOMAINS:
                target_interactions = by_domain.get(target_domain, [])
                if len(target_interactions) < num_test + 1:
                    continue
                own_memory, tests = split_last_n(target_interactions, num_test)
                for test_ordinal, test_inter in enumerate(tests):
                    source_domains: list[str]
                    memory_inters: list[tuple[str, dict[str, Any]]]
                    if protocol == "official_single":
                        source_domains = [target_domain]
                        memory_inters = [(target_domain, inter) for inter in own_memory]
                        setting = f"single_{target_domain.lower()}"
                    else:
                        source_domains = [domain for domain in DOMAINS if domain != target_domain]
                        memory_inters = []
                        for domain in source_domains:
                            for inter in by_domain.get(domain, []):
                                if protocol == "causal_cross" and int(inter.get("timestamp") or 0) >= int(test_inter.get("timestamp") or 0):
                                    continue
                                memory_inters.append((domain, inter))
                        setting = f"cross_to_{target_domain.lower()}"

                    scope_ids: list[str] = []
                    for domain, inter in sorted(memory_inters, key=lambda item: int(item[1].get("timestamp") or 0)):
                        mem_id = memory_id_by_object.get((domain, id(inter)), "")
                        if mem_id in all_memories:
                            scope_ids.append(mem_id)
                    if not scope_ids:
                        continue

                    for task in TASKS:
                        candidates: list[str] = []
                        if task == "item_ranking":
                            positive = safe_text(test_inter.get("parent_asin") or test_inter.get("asin"))
                            negatives = [safe_text(item) for item in test_inter.get("negative_parent_asins", [])]
                            if not positive or not negatives:
                                continue
                            candidates = [positive] + negatives

                        event_id = _event_id(protocol, task, setting, user_id, test_inter, test_ordinal)
                        event = MemoryCDTaskEvent(
                            event_id=event_id,
                            tenant_id=user_id,
                            protocol=protocol,
                            task=task,
                            setting=setting,
                            target_domain=target_domain,
                            source_domains=tuple(source_domains),
                            target_timestamp=int(test_inter.get("timestamp") or 0),
                            query_text=build_query_text(task, test_inter, candidates),
                            candidate_item_metadata=tuple(candidates),
                            runtime_memory_ids=tuple(scope_ids),
                            runtime_memory_scope_size=len(scope_ids),
                        )
                        runtime_out.write(json.dumps(event.to_runtime_dict(), ensure_ascii=False) + "\n")
                        label = _label_for_task(task, test_inter, candidates)
                        label["event_id"] = event_id
                        labels_out.write(json.dumps(label, ensure_ascii=False) + "\n")
                        digest = hashlib.sha256(f"{event_id}|20260622".encode("utf-8")).hexdigest()
                        subset_rows.append({"event_id": event_id, "hash": digest, "task": task, "setting": setting})
                        event_count += 1
                        settings.add(setting)

    subset_rows.sort(key=lambda row: row["hash"])
    # Small fixed subset for V5.1 task-curve smoke; full scripts can raise per-stratum caps.
    per_stratum: dict[tuple[str, str], int] = {}
    with subset_path.open("w", encoding="utf-8") as subset_out:
        for row in subset_rows:
            key = (row["task"], row["setting"])
            if per_stratum.get(key, 0) >= 200:
                continue
            subset_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            per_stratum[key] = per_stratum.get(key, 0) + 1

    summary = {
        "protocol": protocol,
        "users_seen": users_seen if max_users is None else min(users_seen, max_users),
        "memory_count": memory_count,
        "event_count": event_count,
        "settings": sorted(settings),
        "runtime_manifest": str(runtime_path),
        "memory_bank": str(memory_path),
        "evaluation_labels": str(labels_path),
        "subset_event_ids": str(subset_path),
    }
    (manifest_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
