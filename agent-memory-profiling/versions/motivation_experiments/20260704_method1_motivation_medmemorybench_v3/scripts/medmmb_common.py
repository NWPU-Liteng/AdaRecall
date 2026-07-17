#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


def resolve_path(path: str | Path, base: Path = VERSION_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return dict(data)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(resolve_path(path), "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def stable_hash_int(*parts: Any, modulo: int = 2**31 - 1) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def normalize_text(value: Any) -> str:
    raw = str(value or "").lower()
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return " ".join(raw.split())


def parse_maybe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import ast

        return ast.literal_eval(text)
    except Exception:
        return value


def flatten_strings(value: Any) -> list[str]:
    parsed = parse_maybe_json(value)
    output: list[str] = []
    if parsed is None:
        return output
    if isinstance(parsed, str):
        return [parsed] if parsed.strip() else []
    if isinstance(parsed, dict):
        for key, item in parsed.items():
            if isinstance(item, (str, int, float)):
                output.append(str(item))
            else:
                output.extend(flatten_strings(item))
        return output
    if isinstance(parsed, list):
        for item in parsed:
            output.extend(flatten_strings(item))
        return output
    return [str(parsed)]


def first_present(row: dict[str, Any], names: list[str], default: Any = "") -> Any:
    lower_map = {str(key).lower(): key for key in row.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            value = row.get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return default


def infer_persona_id(row: dict[str, Any]) -> Any:
    return first_present(row, ["persona_id", "person_id", "patient_id", "user_id", "profile_id", "subject_id"], 0)


def infer_session_id(row: dict[str, Any]) -> Any:
    return first_present(row, ["session_id", "session", "session_idx", "visit_id", "dialogue_session_id"], 0)


def infer_turn_id(row: dict[str, Any], default: int = 0) -> Any:
    return first_present(row, ["turn", "turn_id", "turn_idx", "utterance_id", "exchange_id"], default)


def infer_query_id(row: dict[str, Any], default: int = 0) -> str:
    value = first_present(row, ["query_id", "question_id", "id", "qid"], "")
    return str(value) if str(value).strip() else f"query_{default:06d}"


def infer_query_text(row: dict[str, Any]) -> str:
    value = first_present(row, ["query", "question", "question_text", "input", "prompt"], "")
    return str(value).strip()


def infer_query_type(row: dict[str, Any]) -> str:
    value = first_present(row, ["query_type", "question_type", "type", "category", "task_type"], "unknown")
    return str(value or "unknown").strip() or "unknown"


def infer_role(row: dict[str, Any]) -> str:
    role = str(first_present(row, ["role", "speaker", "speaker_role", "from"], "")).strip().lower()
    if role in {"user", "patient", "human"}:
        return "patient"
    if role in {"assistant", "doctor", "physician", "clinician"}:
        return "doctor"
    text = str(first_present(row, ["speaker", "name"], "")).lower()
    if "patient" in text:
        return "patient"
    if any(token in text for token in ["doctor", "physician", "clinician"]):
        return "doctor"
    return role or "unknown"


def infer_text(row: dict[str, Any]) -> str:
    value = first_present(row, ["text", "content", "utterance", "message", "dialogue", "response"], "")
    return str(value or "").strip()


def infer_timestamp(row: dict[str, Any], fallback: str) -> str:
    value = first_present(row, ["timestamp", "time", "date", "created_at", "event_time"], "")
    text = str(value or "").strip()
    return text if text else fallback


def session_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(float(text)):012d}")
    except Exception:
        return (1, text)


def session_lt(left: Any, right: Any) -> bool:
    return session_sort_key(left) < session_sort_key(right)


def session_le(left: Any, right: Any) -> bool:
    return session_sort_key(left) <= session_sort_key(right)


def count_tokens(text: str) -> int:
    return len(str(text or "").split())


def set_global_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def require_cuda_if(config: dict[str, Any], *, formal: bool = False) -> None:
    if bool(config.get("require_cuda", False)) or formal:
        if not cuda_available():
            raise RuntimeError("CUDA unavailable; v3 formal/pilot GPU run cannot proceed.")


def cuda_sync() -> None:
    if cuda_available():
        torch.cuda.synchronize()


def timed_call(fn, *args, **kwargs):
    cuda_sync()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    cuda_sync()
    return result, (time.perf_counter() - start) * 1000.0


def environment_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python": sys.version.replace("\n", " "),
        "platform": sys.platform,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_qwen_cuda_device": os.environ.get("LOCAL_QWEN_CUDA_DEVICE"),
        "cuda_available": cuda_available(),
        "device_count": int(torch.cuda.device_count()) if cuda_available() else 0,
        "torch_version": getattr(torch, "__version__", "unavailable") if torch is not None else "unavailable",
        "cuda_version": getattr(torch.version, "cuda", None) if torch is not None else None,
    }
    if cuda_available():
        payload["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if extra:
        payload.update(extra)
    return payload


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_progress(
    *,
    experiment: str,
    mode: str,
    total_queries: int,
    completed_queries: int,
    total_units: int,
    completed_units: int,
    current_query_id: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "experiment": experiment,
        "mode": mode,
        "total_queries": int(total_queries),
        "completed_queries": int(completed_queries),
        "total_units": int(total_units),
        "completed_units": int(completed_units),
        "current_query_id": current_query_id,
        "updated_at": utc_now(),
    }
    if extra:
        payload.update(extra)
    path = VERSION_ROOT / "runs" / "progress" / f"{experiment}_{mode}_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_embedding_client(config: dict[str, Any]):
    from src.embedding_client import TransformerEmbeddingClient

    return TransformerEmbeddingClient(
        model_dir=str(resolve_path(config["embedding_model_dir"])),
        dim=int(config.get("embedding_dim", 768)),
        max_length=int(config.get("embedding_max_length", 512)),
        batch_size=int(config.get("embedding_batch_size", 16)),
    )


def build_reranker_client(config: dict[str, Any]):
    from src.reranker_client import resolve_reranker_client

    cfg = dict(config)
    if "reranker_model_dir" in cfg:
        cfg["reranker_model_dir"] = str(resolve_path(cfg["reranker_model_dir"]))
    client = resolve_reranker_client(cfg, str(PROJECT_ROOT))
    if bool(config.get("require_true_reranker", True)) and getattr(client, "mode", "") != "cross_encoder":
        reason = getattr(client, "reason", "")
        raise RuntimeError(f"Formal v3 requires cross_encoder reranker; got {getattr(client, 'mode', '')}. {reason}")
    return client


def build_llm_client(config: dict[str, Any], max_tokens_key: str = "answer_max_new_tokens"):
    from src.llm_client import LocalQwenLLMClient

    return LocalQwenLLMClient(
        model_dir=str(resolve_path(config.get("controller_answer_model_dir", "../../../../dir"))),
        max_new_tokens=int(config.get(max_tokens_key, config.get("answer_max_new_tokens", 128))),
    )


def dense_retrieve(query_vector: np.ndarray, candidate_vectors: np.ndarray, top_r: int) -> list[tuple[int, float]]:
    if candidate_vectors.size == 0:
        return []
    scores = candidate_vectors @ query_vector.astype(np.float32)
    count = min(int(top_r), len(scores))
    if count <= 0:
        return []
    if count == len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -count)[-count:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def ensure_embedding_cache(memory_items: list[dict[str, Any]], retrieval_config: dict[str, Any], cache_stem: str) -> tuple[np.ndarray, list[str], float]:
    cache_dir = VERSION_ROOT / "aggregate" / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(retrieval_config.get("embedding_model_dir", ""))
    stem = hashlib.sha256((cache_stem + "::" + model_id).encode("utf-8")).hexdigest()[:16]
    vector_path = cache_dir / f"{stem}.npy"
    ids_path = cache_dir / f"{stem}.ids.json"
    if vector_path.exists() and ids_path.exists():
        return np.load(vector_path).astype(np.float32), json.loads(ids_path.read_text(encoding="utf-8")), 0.0
    client = build_embedding_client(retrieval_config)
    vectors, elapsed_ms = timed_call(client.embed_batch, [row["text"] for row in memory_items])
    ids = [str(row["memory_id"]) for row in memory_items]
    np.save(vector_path, vectors.astype(np.float32))
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    return vectors.astype(np.float32), ids, elapsed_ms


def run_flat_selection(
    *,
    query_text: str,
    candidate_memories: list[dict[str, Any]],
    candidate_vectors: np.ndarray,
    retrieval_config: dict[str, Any],
    embed_client,
    reranker_client,
    frontier_budget: int,
    final_top_k: int,
) -> dict[str, Any]:
    ann_top_r = max(int(frontier_budget), int(frontier_budget) * int(retrieval_config.get("ann_top_r_multiplier", 4)))
    query_vector, t_query_encode_ms = timed_call(embed_client.embed, query_text)
    ann_pairs, t_ann_ms = timed_call(dense_retrieve, query_vector, candidate_vectors, min(ann_top_r, len(candidate_memories)))
    ann_candidate_ids = [str(candidate_memories[idx]["memory_id"]) for idx, _ in ann_pairs]
    filtered_pairs = ann_pairs
    seen: set[str] = set()
    deduped_pairs: list[tuple[int, float]] = []
    for idx, score in filtered_pairs:
        memory_id = str(candidate_memories[idx]["memory_id"])
        if memory_id in seen:
            continue
        seen.add(memory_id)
        deduped_pairs.append((idx, score))
    frontier_pairs = deduped_pairs[: min(int(frontier_budget), len(deduped_pairs))]
    materialized = [(candidate_memories[idx], score) for idx, score in frontier_pairs]
    payload_bytes_read = sum(int(memory.get("payload_bytes", len(memory.get("text", "").encode("utf-8")))) for memory, _ in materialized)
    metadata_bytes_read = int(retrieval_config.get("header_size_bytes", 128)) * len(deduped_pairs)
    if materialized:
        rerank_input = [(query_text, memory, score) for memory, score in materialized]
        rerank_scores, t_rerank_ms = timed_call(reranker_client.score_pairs, rerank_input)
        reranked = sorted(
            [
                {
                    "memory": memory,
                    "vector_score": float(vector_score),
                    "rerank_score": float(rerank_score),
                }
                for (memory, vector_score), rerank_score in zip(materialized, rerank_scores)
            ],
            key=lambda row: row["rerank_score"],
            reverse=True,
        )
    else:
        reranked = []
        t_rerank_ms = 0.0
    selected = reranked[: int(final_top_k)]
    selected_ids = [str(row["memory"]["memory_id"]) for row in selected]
    reranked_ids = [str(row["memory"]["memory_id"]) for row in reranked]
    t_filter_ms = 0.0
    t_materialize_ms = 0.0
    return {
        "ann_candidate_ids": ann_candidate_ids,
        "reranked_candidate_ids": reranked_ids,
        "final_selected_ids": selected_ids,
        "selected_memories": [row["memory"] for row in selected],
        "n_ann_raw": len(ann_pairs),
        "n_after_filter": len(filtered_pairs),
        "n_after_dedup": len(deduped_pairs),
        "n_materialized": len(materialized),
        "n_reranked": len(reranked),
        "n_final_selected": len(selected),
        "t_query_encode_ms": t_query_encode_ms,
        "t_ann_ms": t_ann_ms,
        "t_filter_ms": t_filter_ms,
        "t_materialize_ms": t_materialize_ms,
        "t_rerank_ms": t_rerank_ms,
        "t_total_selection_ms": t_query_encode_ms + t_ann_ms + t_filter_ms + t_materialize_ms + t_rerank_ms,
        "metadata_bytes_read": metadata_bytes_read,
        "payload_bytes_read": payload_bytes_read,
        "index_bytes_read": int(candidate_vectors.nbytes),
        "host_memory_bytes": metadata_bytes_read + payload_bytes_read + int(candidate_vectors.nbytes),
    }


def eligible_memories_for_query(
    memories_by_persona: dict[str, list[dict[str, Any]]],
    query: dict[str, Any],
    *,
    allow_current_session: bool = False,
    include_noise: bool = False,
) -> list[dict[str, Any]]:
    persona = str(query["persona_id"])
    query_session = query["query_session_id"]
    rows = []
    for memory in memories_by_persona.get(persona, []):
        if str(memory.get("persona_id")) != persona:
            continue
        if allow_current_session:
            valid_time = session_le(memory.get("session_id"), query_session)
        else:
            valid_time = session_lt(memory.get("session_id"), query_session)
        if not valid_time:
            continue
        session_type = str(memory.get("session_type", "unknown"))
        if not include_noise and session_type.startswith("noise"):
            continue
        rows.append(memory)
    return rows


def validate_no_future_or_cross_persona(query: dict[str, Any], memories: list[dict[str, Any]], *, allow_current_session: bool = False) -> None:
    persona = str(query["persona_id"])
    query_session = query["query_session_id"]
    for memory in memories:
        if str(memory.get("persona_id")) != persona:
            raise RuntimeError(f"Cross-persona memory access: query={query['query_id']} memory={memory.get('memory_id')}")
        valid_time = session_le(memory.get("session_id"), query_session) if allow_current_session else session_lt(memory.get("session_id"), query_session)
        if not valid_time:
            raise RuntimeError(f"Future memory access: query={query['query_id']} memory={memory.get('memory_id')}")


def build_nested_history(
    *,
    query: dict[str, Any],
    eligible: list[dict[str, Any]],
    gold_memory_ids: set[str],
    scale: int | str,
    seed: int,
) -> list[dict[str, Any]]:
    if scale == "full":
        selected = list(eligible)
    else:
        target = int(scale)
        gold = [row for row in eligible if str(row["memory_id"]) in gold_memory_ids]
        if len(gold) > target:
            selected = gold[:target]
        else:
            remaining = [row for row in eligible if str(row["memory_id"]) not in gold_memory_ids]
            buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for row in remaining:
                buckets[row.get("session_id")].append(row)
            rng = random.Random(stable_hash_int(query["query_id"], scale, seed))
            fill: list[dict[str, Any]] = []
            session_keys = sorted(buckets.keys(), key=session_sort_key)
            while len(fill) + len(gold) < target and session_keys:
                progressed = False
                for session in session_keys:
                    bucket = buckets[session]
                    if not bucket:
                        continue
                    idx = rng.randrange(len(bucket))
                    fill.append(bucket.pop(idx))
                    progressed = True
                    if len(fill) + len(gold) >= target:
                        break
                if not progressed:
                    break
            selected = gold + fill
    selected.sort(key=lambda row: int(row.get("chronological_rank", 0)))
    missing_gold = gold_memory_ids - {str(row["memory_id"]) for row in selected}
    if missing_gold:
        raise RuntimeError(f"Nested history failed to retain gold memories for query {query['query_id']}: {sorted(missing_gold)[:5]}")
    return selected


def evidence_metrics(selected_ids: list[str], gold_ids: set[str]) -> dict[str, float]:
    selected = set(selected_ids)
    if not gold_ids:
        return {
            "evidence_recall_at_k": float("nan"),
            "evidence_completion_at_k": float("nan"),
            "candidate_precision": float("nan"),
        }
    overlap = len(selected & gold_ids)
    return {
        "evidence_recall_at_k": overlap / float(len(gold_ids)),
        "evidence_completion_at_k": 1.0 if gold_ids <= selected else 0.0,
        "candidate_precision": overlap / float(len(selected)) if selected else 0.0,
    }


def read_split(path: str | Path, mode: str, subset_name: str | None = None) -> list[str]:
    split_path = resolve_path(path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if subset_name and subset_name in payload:
        rows = payload[subset_name]
    elif mode in payload:
        rows = payload[mode]
    else:
        raise KeyError(f"Split {mode!r} not found in {split_path}")
    if rows and isinstance(rows[0], dict):
        return [str(row["query_id"]) for row in rows]
    return [str(row) for row in rows]


def group_memories_by_persona(memory_items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in memory_items:
        grouped[str(row["persona_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("chronological_rank", 0)))
    return grouped


def build_context(selected_memories: list[dict[str, Any]]) -> str:
    if not selected_memories:
        return "No retrieved memory."
    parts = []
    for memory in selected_memories:
        parts.append(
            "[memory_id={mid} | session={session} | rank={rank} | type={stype}]\n{text}".format(
                mid=memory["memory_id"],
                session=memory.get("session_id", ""),
                rank=memory.get("chronological_rank", ""),
                stype=memory.get("session_type", "unknown"),
                text=memory.get("text", ""),
            )
        )
    return "\n\n".join(parts)


def build_answer_prompt(query: dict[str, Any], selected_memories: list[dict[str, Any]]) -> str:
    return (
        "You are evaluating a text-only healthcare memory benchmark. This is research data, not clinical advice.\n"
        "Answer the user query using only the provided memory context.\n"
        "Do not use external knowledge or tools.\n"
        "Do not mention source key points, gold answers, or hidden labels.\n"
        "If the memory context is insufficient, answer exactly: insufficient information\n"
        "Output one concise final answer line.\n\n"
        f"Query type: {query.get('query_type', 'unknown')}\n"
        f"Current session: {query.get('query_session_id', '')}\n"
        f"User query: {query.get('query_text', '')}\n\n"
        f"Memory context:\n{build_context(selected_memories)}\n\n"
        "Final answer only:"
    )


def simple_answer_normalize(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    answer = lines[0] if lines else raw
    answer = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def bootstrap_ci(values: list[float], seed: int = 20260704, reps: int = 1000) -> tuple[float, float]:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def summarize_numeric(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str], seed: int = 20260704) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: str(item[0])):
        base = {name: value for name, value in zip(group_keys, key)}
        base["n"] = len(items)
        for metric in metrics:
            vals = [float(row[metric]) for row in items if row.get(metric) not in ("", None) and not math.isnan(float(row[metric]))]
            if vals:
                lo, hi = bootstrap_ci(vals, seed=seed)
                base[f"{metric}_mean"] = sum(vals) / len(vals)
                base[f"{metric}_median"] = float(np.median(vals))
                base[f"{metric}_p95"] = float(np.quantile(vals, 0.95))
                base[f"{metric}_ci95_low"] = lo
                base[f"{metric}_ci95_high"] = hi
            else:
                base[f"{metric}_mean"] = ""
                base[f"{metric}_median"] = ""
                base[f"{metric}_p95"] = ""
                base[f"{metric}_ci95_low"] = ""
                base[f"{metric}_ci95_high"] = ""
        output.append(base)
    return output
