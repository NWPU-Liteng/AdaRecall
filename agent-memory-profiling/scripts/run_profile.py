#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import statistics
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.embedding_client import LocalQwenEmbeddingClient, OllamaEmbeddingClient
from src.llm_client import LocalQwenLLMClient, OllamaLLMClient
from src.memory_invocation import run_langgraph_memory_pipeline, run_memory_pipeline
from src.vector_store import AnnNumpyVectorStore, NumpyVectorStore, get_vector_store


QUERIES = {
    "simple_semantic": ("general", "How should I adjust training after poor sleep and fatigue?"),
    "planning": ("planning", "Plan a 3-day routine while considering knee pain, sleep, and recovery."),
    "reflection": ("reflection", "Reflect on recent workout feedback and summarize what patterns are emerging."),
    "risk_control": ("risk_control", "Identify risks if the user keeps increasing running volume despite knee discomfort."),
    "profile_update": ("profile_update", "Update the long-term profile based on recent diet, weight, and fatigue notes."),
}


class QueryTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    _ = signum, frame
    raise QueryTimeout("query timeout")


def ensure_index(size: int, prefer_qdrant: bool, dim: int, retrieval_backend: str):
    input_path = Path("data") / f"synthetic_memories_{size}.jsonl"
    vector_path = Path("data") / f"index_vectors_{size}.npy"
    metadata_path = Path("data") / f"index_metadata_{size}.jsonl"
    ann_index_path = Path("data") / f"index_ann_{size}.npz"
    if vector_path.exists() and metadata_path.exists():
        if retrieval_backend != "ann_numpy" or ann_index_path.exists():
            return
    _ = prefer_qdrant, dim
    if retrieval_backend == "ann_numpy":
        raise RuntimeError(
            f"Missing ANN index for size {size}. Run build_index.py --retrieval-backend ann_numpy first for {input_path}."
        )
    if vector_path.exists() and metadata_path.exists():
        return
    raise RuntimeError(
        f"Missing index for size {size}. Run build_index.py first for {input_path}."
    )


def store_for_size(size: int, prefer_qdrant: bool, dim: int, retrieval_backend: str):
    ann_index_exists = (Path("data") / f"index_ann_{size}.npz").exists()

    def local_fallback_backend() -> str:
        return "ann_numpy" if ann_index_exists else "numpy"

    try:
        store = get_vector_store(
            prefer_qdrant,
            size=size,
            dim=dim,
            base_dir="data",
            retrieval_backend=retrieval_backend,
        )
    except Exception:
        fallback_backend = local_fallback_backend() if retrieval_backend == "qdrant" else "numpy"
        store = get_vector_store(
            False,
            size=size,
            dim=dim,
            base_dir="data",
            retrieval_backend=fallback_backend,
        )
    if getattr(store, "mode", "") == "qdrant":
        try:
            store.client.get_collection(f"agent_memory_{size}")
            return store
        except Exception:
            fallback_backend = local_fallback_backend() if retrieval_backend in {"auto", "qdrant"} else "numpy"
            store = get_vector_store(
                False,
                size=size,
                dim=dim,
                base_dir="data",
                retrieval_backend=fallback_backend,
            )
    if retrieval_backend == "ann_numpy":
        ann_store = AnnNumpyVectorStore(
            vector_path=str(Path("data") / f"index_vectors_{size}.npy"),
            metadata_path=str(Path("data") / f"index_metadata_{size}.jsonl"),
            ann_index_path=str(Path("data") / f"index_ann_{size}.npz"),
        )
        ann_store.load()
        return ann_store
    numpy_store = NumpyVectorStore(
        vector_path=str(Path("data") / f"index_vectors_{size}.npy"),
        metadata_path=str(Path("data") / f"index_metadata_{size}.jsonl"),
    )
    numpy_store.load()
    return numpy_store


def write_summary(rows, output_path: Path):
    if not rows:
        return
    grouped = {}
    for row in rows:
        if row["error"]:
            continue
        key = (
            row["memory_size"],
            row["query_type"],
            row["module"],
            row["execution_mode"],
            row["framework_mode"],
            row["memory_rounds"],
            row["candidate_k"],
            row["top_k"],
            row["backend_mode"],
            row["llm_mode"],
            row["embedding_mode"],
        )
        grouped.setdefault(key, []).append(row)

    fieldnames = [
        "memory_size",
        "query_type",
        "module",
        "execution_mode",
        "framework_mode",
        "memory_rounds",
        "candidate_k",
        "top_k",
        "backend_mode",
        "llm_mode",
        "embedding_mode",
        "runs",
        "avg_t_total",
        "p95_t_total",
        "avg_t_memory",
        "avg_num_memory_calls",
        "avg_memory_invocation_ratio",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, items in sorted(grouped.items()):
            totals = [float(item["t_total"]) for item in items]
            ratios = [float(item["memory_invocation_ratio"]) for item in items]
            writer.writerow(
                {
                    "memory_size": key[0],
                    "query_type": key[1],
                    "module": key[2],
                    "execution_mode": key[3],
                    "framework_mode": key[4],
                    "memory_rounds": key[5],
                    "candidate_k": key[6],
                    "top_k": key[7],
                    "backend_mode": key[8],
                    "llm_mode": key[9],
                    "embedding_mode": key[10],
                    "runs": len(items),
                    "avg_t_total": statistics.mean(totals),
                    "p95_t_total": max(totals) if len(totals) < 20 else statistics.quantiles(totals, n=20)[18],
                    "avg_t_memory": statistics.mean(float(item["t_memory"]) for item in items),
                    "avg_num_memory_calls": statistics.mean(float(item["num_memory_calls"]) for item in items),
                    "avg_memory_invocation_ratio": statistics.mean(ratios),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-sizes", nargs="+", type=int, required=True)
    parser.add_argument("--candidate-k", nargs="+", type=int, required=True)
    parser.add_argument("--top-k", nargs="+", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--timeout-per-query", type=int, default=30)
    parser.add_argument("--prefer-qdrant", type=int, default=1)
    parser.add_argument("--retrieval-backend", choices=["auto", "numpy", "ann_numpy", "qdrant"], default="auto")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--execution-mode", choices=["single", "agent", "langgraph"], default="single")
    parser.add_argument("--memory-rounds", type=int, default=1)
    parser.add_argument("--print-index-status", type=int, default=0)
    parser.add_argument("--llm-max-new-tokens", type=int, default=24)
    parser.add_argument("--llm-backend", choices=["auto", "local_qwen", "ollama_mock"], default="auto")
    parser.add_argument("--embedding-backend", choices=["auto", "local_qwen", "ollama_hash"], default="auto")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    fieldnames = [
        "memory_size",
        "query_type",
        "module",
        "execution_mode",
        "framework_mode",
        "memory_rounds",
        "candidate_k",
        "top_k",
        "t_embed",
        "t_search",
        "t_filter",
        "t_rerank",
        "t_pack",
        "t_memory",
        "t_llm",
        "t_total",
        "memory_invocation_ratio",
        "num_candidates",
        "num_after_filter",
        "num_memory_calls",
        "prompt_chars",
        "backend_mode",
        "llm_mode",
        "embedding_mode",
        "error",
    ]
    rows = []
    signal.signal(signal.SIGALRM, timeout_handler)

    local_qwen_dir = os.getenv("LOCAL_QWEN_DIR", "../dir")

    def make_embed_client():
        if args.embedding_backend == "local_qwen":
            return LocalQwenEmbeddingClient(model_dir=local_qwen_dir, dim=args.dim)
        if args.embedding_backend == "auto":
            try:
                return LocalQwenEmbeddingClient(model_dir=local_qwen_dir, dim=args.dim)
            except Exception:
                return OllamaEmbeddingClient(dim=args.dim)
        return OllamaEmbeddingClient(dim=args.dim)

    def make_llm_client():
        if args.llm_backend == "local_qwen":
            return LocalQwenLLMClient(model_dir=local_qwen_dir, max_new_tokens=args.llm_max_new_tokens)
        if args.llm_backend == "auto":
            try:
                return LocalQwenLLMClient(model_dir=local_qwen_dir, max_new_tokens=args.llm_max_new_tokens)
            except Exception:
                return OllamaLLMClient()
        return OllamaLLMClient()

    framework_mode = "langgraph" if args.execution_mode == "langgraph" else "builtin"

    for size in args.memory_sizes:
        ensure_index(size, bool(args.prefer_qdrant), args.dim, args.retrieval_backend)
        store = store_for_size(size, bool(args.prefer_qdrant), args.dim, args.retrieval_backend)
        for candidate_k in args.candidate_k:
            for top_k in args.top_k:
                for query_type, (module, query) in QUERIES.items():
                    for _ in range(args.repeat):
                        embed_client = make_embed_client()
                        llm_client = make_llm_client()
                        row = {
                            "memory_size": size,
                            "query_type": query_type,
                            "module": module,
                            "execution_mode": args.execution_mode,
                            "framework_mode": framework_mode,
                            "memory_rounds": args.memory_rounds,
                            "candidate_k": candidate_k,
                            "top_k": top_k,
                            "t_embed": "",
                            "t_search": "",
                            "t_filter": "",
                            "t_rerank": "",
                            "t_pack": "",
                            "t_memory": "",
                            "t_llm": "",
                            "t_total": "",
                            "memory_invocation_ratio": "",
                            "num_candidates": "",
                            "num_after_filter": "",
                            "num_memory_calls": "",
                            "prompt_chars": "",
                            "backend_mode": getattr(store, "mode", "numpy_fallback"),
                            "llm_mode": "",
                            "embedding_mode": "",
                            "error": "",
                        }
                        try:
                            signal.alarm(args.timeout_per_query)
                            if args.execution_mode == "langgraph":
                                stats = run_langgraph_memory_pipeline(
                                    query=query,
                                    module=module,
                                    candidate_k=candidate_k,
                                    top_k=top_k,
                                    embed_client=embed_client,
                                    llm_client=llm_client,
                                    vector_store=store,
                                    memory_rounds=args.memory_rounds,
                                )
                            else:
                                stats = run_memory_pipeline(
                                    query=query,
                                    module=module,
                                    candidate_k=candidate_k,
                                    top_k=top_k,
                                    embed_client=embed_client,
                                    llm_client=llm_client,
                                    vector_store=store,
                                    execution_mode=args.execution_mode,
                                    memory_rounds=args.memory_rounds,
                                )
                            signal.alarm(0)
                            row.update(
                                {
                                    "t_embed": stats["t_embed"],
                                    "t_search": stats["t_search"],
                                    "t_filter": stats["t_filter"],
                                    "t_rerank": stats["t_rerank"],
                                    "t_pack": stats["t_pack"],
                                    "t_memory": stats["t_memory"],
                                    "t_llm": stats["t_llm"],
                                    "t_total": (
                                        stats["t_embed"]
                                        + stats["t_search"]
                                        + stats["t_filter"]
                                        + stats["t_rerank"]
                                        + stats["t_pack"]
                                        + stats["t_llm"]
                                    ),
                                    "memory_invocation_ratio": stats["memory_invocation_ratio"],
                                    "num_candidates": stats["num_candidates"],
                                    "num_after_filter": stats["num_after_filter"],
                                    "num_memory_calls": stats["num_memory_calls"],
                                    "prompt_chars": stats["prompt_chars"],
                                    "backend_mode": getattr(store, "mode", "numpy_fallback"),
                                    "llm_mode": llm_client.mode,
                                    "embedding_mode": embed_client.mode,
                                }
                            )
                        except Exception as exc:
                            signal.alarm(0)
                            row["error"] = str(exc)
                            row["llm_mode"] = getattr(llm_client, "mode", "mock")
                            row["embedding_mode"] = getattr(embed_client, "mode", "hash_embedding")
                        rows.append(row)

    with open("results/raw_profile.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, Path("results/profile_summary.csv"))

    status_path = Path("results/index_status.json")
    if args.print_index_status and status_path.exists():
        with open(status_path, "r", encoding="utf-8") as handle:
            print(json.dumps(json.load(handle), ensure_ascii=False))
    print(f"Wrote {len(rows)} rows to results/raw_profile.csv")


if __name__ == "__main__":
    main()
