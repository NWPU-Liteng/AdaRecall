from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ROOT.parent
PROJECT = VERSIONS.parent
V28 = VERSIONS / "20260718_prism_aligned_method1_graph_only_v28"
V24 = VERSIONS / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24 / "src"))

from v24.dataset import load_canonical  # noqa: E402

FRONTIER = V28 / "runs" / "v28_frontier_oof_20260719_attempt003_913cd3f" / "per_query_frontier.jsonl"
OUTPUT = ROOT / "artifacts" / "v28_qwen_truncation_audit.json"
MODEL = PROJECT / "models" / "models" / "Qwen3-Reranker-0.6B"
EXPECTED_FRONTIER_SHA = "834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98"
INSTRUCTION = "Given a question about a user's history, retrieve memory records that provide the evidence needed to answer it."
SYSTEM = "Judge whether the Document meets the requirements based on the Query and Instruct. Answer only yes or no."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def format_prompt(tokenizer, query: str, document: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}",
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def summarize(values: Sequence[int]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": int(array.max()),
        "over_512_count": int(np.sum(array > 512)),
        "over_512_rate": float(np.mean(array > 512)),
        "over_1024_count": int(np.sum(array > 1024)),
        "over_1024_rate": float(np.mean(array > 1024)),
        "over_2048_count": int(np.sum(array > 2048)),
        "over_2048_rate": float(np.mean(array > 2048)),
    }


def main() -> None:
    if sha256(FRONTIER) != EXPECTED_FRONTIER_SHA:
        raise RuntimeError("immutable v28 Frontier rows changed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL),
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )
    dataset = load_canonical(V24 / "inputs", split="train")
    query_by_id = {query.query_id: query for query in dataset.queries["train"]}
    memory_by_id = {memory.memory_id: memory for memory in dataset.memories}
    evaluations = dataset.evaluations["train"]
    dense_by_query = {
        row["query_id"]: set(row["frontier_ids"])
        for row in iter_jsonl(FRONTIER)
        if row["method"] == "dense_hnsw_32"
    }
    semantic_rows = [row for row in iter_jsonl(FRONTIER) if row["method"] == "semantic_graph_only_32"]
    prompts = []
    roles = []
    for row in semantic_rows:
        query_id = row["query_id"]
        gold = set(evaluations[query_id].gold_memory_ids)
        dense = dense_by_query[query_id]
        for memory_id in row["frontier_ids"]:
            if memory_id in gold and memory_id not in dense:
                role = "graph_novel_gold"
            elif memory_id in gold:
                role = "shared_gold"
            else:
                role = "selected_false_positive"
            prompts.append(format_prompt(tokenizer, query_by_id[query_id].query, memory_by_id[memory_id].content))
            roles.append(role)
    lengths = []
    for start in range(0, len(prompts), 128):
        encoded = tokenizer(
            prompts[start : start + 128],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths.extend(int(value) for value in encoded["length"])
        if start % 4096 == 0:
            print(f"tokenized {min(start + 128, len(prompts))}/{len(prompts)}", flush=True)
    by_role = defaultdict(list)
    for role, length in zip(roles, lengths):
        by_role[role].append(length)

    marker = "__METHOD1_DOCUMENT_BOUNDARY_MARKER_20260719__"
    marker_prompt = format_prompt(tokenizer, "audit query", marker)
    if marker not in marker_prompt:
        raise RuntimeError("chat template changed the document boundary marker")
    prefix, suffix = marker_prompt.split(marker, maxsplit=1)
    prefix_tokens = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    suffix_tokens = len(tokenizer(suffix, add_special_tokens=False)["input_ids"])
    output = {
        "schema_version": "method1-v30-qwen-truncation-audit-1",
        "status": "TRUNCATION_AUDIT_COMPLETE",
        "scientific_scope": "post-hoc engineering validity audit of immutable v28 Qwen input formatting",
        "frontier_rows_sha256": EXPECTED_FRONTIER_SHA,
        "model_path": str(MODEL),
        "tokenizer_truncation_side": tokenizer.truncation_side,
        "v28_max_length": 512,
        "chat_template_prefix_tokens_for_short_audit_query": prefix_tokens,
        "chat_template_suffix_tokens": suffix_tokens,
        "right_truncation_removes_generation_suffix_when_over_budget": bool(tokenizer.truncation_side == "right"),
        "all_semantic_candidates": summarize(lengths),
        "by_candidate_role": {role: summarize(values) for role, values in sorted(by_role.items())},
        "validity_decision": (
            "INVALID_YES_NO_POSITION_FOR_TRUNCATED_PROMPTS"
            if tokenizer.truncation_side == "right" and any(length > 512 for length in lengths)
            else "NO_SUFFIX_TRUNCATION_DETECTED"
        ),
        "required_fix": "preserve chat prefix and assistant generation suffix; apply identical document-window policy to every retrieval method",
        "fresh_authorized": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "validity_decision": output["validity_decision"], "all": output["all_semantic_candidates"]}, indent=2))


if __name__ == "__main__":
    main()
