#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parent.parent
CONFIG_DIR = VERSION_ROOT / "config" / "v4_official_qa_budget_grid"


BASE_CONFIG = {
    "seed": 42,
    "embedding_dim": 768,
    "embedding_backend": "transformer",
    "embedding_model_dir": str(PROJECT_ROOT / "models" / "bge-base-en-v1.5"),
    "embedding_max_length": 512,
    "embedding_batch_size": 16,
    "retrieval_top_r": 256,
    "header_size_bytes": 128,
    "reranker_backend": "cross_encoder",
    "reranker_model_dir": str(PROJECT_ROOT / "models" / "cross-encoder-ms-marco-MiniLM-L6-v2"),
    "reranker_batch_size": 1,
    "reranker_max_length": 256,
    "generation_backend": "local_qwen",
    "controller_max_new_tokens": 8,
    "answer_max_new_tokens": 48,
    "agent_memory_mode": "shared_once",
    "selection_only": False,
    "event_type_filter": "official_qa",
    "final_top_k": 5,
}

DATASETS = {
    "locomo": {
        "manifest": "versions/20260621_real_trace_adaptive_phase_v4_formal/manifests/locomo/runtime_manifest.jsonl",
        "label_manifest": "versions/20260621_real_trace_adaptive_phase_v4_formal/manifests/locomo/evaluation_labels.jsonl",
    },
    "longmemeval_s": {
        "manifest": "versions/20260621_real_trace_adaptive_phase_v4_formal/manifests/longmemeval_s/runtime_manifest.jsonl",
        "label_manifest": "versions/20260621_real_trace_adaptive_phase_v4_formal/manifests/longmemeval_s/evaluation_labels.jsonl",
    },
}

BUDGETS = [8, 16, 32, 64]

TIERS = {
    "smoke": {
        "max_events": 8,
        "event_sort_by": "visible_memory_count_desc",
        "min_visible_memory_count": 64,
    },
    "formal": {
        "max_events": 64,
        "event_sort_by": "visible_memory_count_desc",
        "min_visible_memory_count": 64,
    },
}


def build_config(dataset: str, budget: int, tier: str) -> dict:
    config = dict(BASE_CONFIG)
    config["dataset_name"] = dataset
    config["pruner_top_b"] = budget
    config["frontier_budget"] = budget
    config["tier"] = tier
    config["runtime_manifest"] = DATASETS[dataset]["manifest"]
    config["evaluation_manifest"] = DATASETS[dataset]["label_manifest"]
    config.update(TIERS[tier])
    return config


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tier in TIERS:
        for dataset in DATASETS:
            for budget in BUDGETS:
                config = build_config(dataset, budget, tier)
                filename = f"{tier}_{dataset}_budget{budget}.json"
                path = CONFIG_DIR / filename
                path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                manifest.append(
                    {
                        "tier": tier,
                        "dataset": dataset,
                        "budget": budget,
                        "config_path": str(path.relative_to(PROJECT_ROOT)),
                        "runtime_manifest": config["runtime_manifest"],
                    }
                )
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_configs={len(manifest)}")
    print(f"manifest={CONFIG_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
