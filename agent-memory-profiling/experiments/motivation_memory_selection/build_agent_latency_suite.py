#!/usr/bin/env python3
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "experiments" / "motivation_memory_selection" / "configs" / "agent_latency_suite_20260615"
MODEL_DIR = PROJECT_ROOT / "models" / "cross-encoder-ms-marco-MiniLM-L6-v2"


DATASETS = {
    "locomo": {
        "input_path": "data/motivation_memory_selection/locomo_memories_1000.jsonl",
        "memory_size": 5882,
        "index_namespace": "locomo_full",
    },
    "longmemeval": {
        "input_path": "data/motivation_memory_selection/longmemeval_memories_1000.jsonl",
        "memory_size": 23867,
        "index_namespace": "longmemeval_full",
    },
    "msc": {
        "input_path": "data/motivation_memory_selection/msc_memories_5000.jsonl",
        "memory_size": 395934,
        "index_namespace": "msc_full",
    },
}


MODES = {
    "no_rerank": {
        "baseline_modes": ["retrieval_only", "retrieval_plus_expansion"],
        "reranker_backend": "heuristic",
    },
    "heuristic_rerank": {
        "baseline_modes": ["retrieval_only", "retrieval_plus_expansion", "full_selection", "oracle_pruned_rerank"],
        "reranker_backend": "heuristic",
    },
    "cross_encoder_rerank": {
        "baseline_modes": ["retrieval_only", "retrieval_plus_expansion", "full_selection", "oracle_pruned_rerank"],
        "reranker_backend": "cross_encoder",
    },
}


AGENT_VARIANTS = {
    "mock_agent": {
        "generation_backend": "mock",
        "enable_generation": False,
        "agent_execution_backend": "calibrated_mock",
    },
    "real_agent": {
        "generation_backend": "local_qwen",
        "enable_generation": False,
        "agent_execution_backend": "llm",
        "agent_enable_intermediate_llm_steps": True,
    },
    "real_agent_v2": {
        "generation_backend": "local_qwen",
        "enable_generation": False,
        "agent_execution_backend": "llm",
        "agent_enable_intermediate_llm_steps": False,
        "agent_planning_llm_rounds": 1,
        "agent_reasoning_llm_rounds": 1,
        "agent_output_llm_rounds": 1,
        "agent_execution_steps": 1,
        "agent_tool_calls_per_step": 1,
        "agent_memory_mode": "shared_once",
        "llm_max_new_tokens": 16,
        "embedding_backend": "local_qwen",
    },
}


BASE_CONFIG = {
    "seed": 42,
    "embedding_dim": 384,
    "embedding_backend": "hash",
    "retrieval_backend": "ann_numpy",
    "enable_agent_execution": True,
    "agent_execution_steps": 3,
    "agent_tool_calls_per_step": 1,
    "agent_mock_planning_ms": 4.0,
    "agent_mock_reasoning_ms_per_step": 8.0,
    "agent_mock_tool_ms_per_call": 3.0,
    "agent_mock_output_ms": 12.0,
    "llm_max_new_tokens": 24,
    "query_types": ["fact", "temporal", "relation", "plan_continuity", "constraint"],
    "retrieval_topk_values": [50, 100],
    "relation_expansion_depth_values": [0, 1],
    "rerank_topk_input_values": ["all", 50],
    "num_queries_per_setting": 20,
    "final_topk": 8,
    "oracle_prune_limit": 50,
    "header_size_bytes": 128,
    "edge_entry_size_bytes": 8,
    "candidate_payload_size_bytes": 512,
    "embedding_size_bytes": 1536,
    "external_memory_sampling": "head",
    "reranker_batch_size": 8,
    "reranker_max_length": 256,
}


def build_config(dataset_name: str, mode_name: str, agent_variant_name: str, run_tier: str = "formal") -> dict:
    dataset = DATASETS[dataset_name]
    mode = MODES[mode_name]
    agent_variant = AGENT_VARIANTS[agent_variant_name]
    config = dict(BASE_CONFIG)
    config.update(
        {
            "memory_sizes": [dataset["memory_size"]],
            "input_memory_paths": {
                str(dataset["memory_size"]): dataset["input_path"],
            },
            "index_namespace": f"{dataset['index_namespace']}_{agent_variant_name}",
            "baseline_modes": mode["baseline_modes"],
            "reranker_backend": mode["reranker_backend"],
            "reranker_model_dir": str(MODEL_DIR),
            "output_dir": f"outputs/agent_latency_suite_20260615/{run_tier}/{agent_variant_name}/{dataset_name}/{mode_name}",
        }
    )
    config.update(agent_variant)
    return config


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run_tier in ["formal", "smoke"]:
        for agent_variant_name in AGENT_VARIANTS:
            for dataset_name in DATASETS:
                for mode_name in MODES:
                    config = build_config(dataset_name, mode_name, agent_variant_name, run_tier=run_tier)
                    path = CONFIG_DIR / f"{run_tier}_{agent_variant_name}_{dataset_name}_{mode_name}.yaml"
                    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    manifest.append(
                        {
                            "run_tier": run_tier,
                            "agent_variant": agent_variant_name,
                            "dataset": dataset_name,
                            "mode": mode_name,
                            "config_path": str(path.relative_to(PROJECT_ROOT)),
                            "output_dir": config["output_dir"],
                        }
                    )

    manifest_path = CONFIG_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote_configs={len(manifest)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
