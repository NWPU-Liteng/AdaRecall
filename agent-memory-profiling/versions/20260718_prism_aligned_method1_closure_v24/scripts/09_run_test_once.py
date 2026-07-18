from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from common import PROJECT_ROOT, VERSION_ROOT, add_common_args, artifact_root, config_hash, load_config
from v24.dataset import load_canonical
from v24.io import write_json, write_jsonl
from v24.pipeline import Method1Runner, run_query_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--index-run-id", default="v24_index_20260718")
    args = parser.parse_args()
    if not args.run_id:
        args.run_id = "v24_test_once_20260718"
    frozen_path = VERSION_ROOT / "artifacts" / "frozen" / "development_config.json"
    if not frozen_path.exists():
        status = {"status": "NOT_RUN", "reason": "development_gate_not_passed_or_freeze_missing", "test_read_status": "not_read"}
        write_json(VERSION_ROOT / "negative_results" / "test_not_run.json", status)
        print(json.dumps(status, sort_keys=True))
        return
    lock_path = VERSION_ROOT / "artifacts" / "frozen" / "test_read_lock.json"
    status_path = VERSION_ROOT / "aggregate" / "test_status.json"
    existing_results = list((VERSION_ROOT / "runs").glob("v24_test_once_*/query_results.jsonl"))
    if lock_path.exists() or status_path.exists() or existing_results:
        raise RuntimeError("test split is already locked or has a completed attempt; refusing a second run")
    frozen = __import__("v24.io", fromlist=["read_json"]).read_json(frozen_path)
    config = frozen["resolved_config"]
    write_json(lock_path, {
        "status": "LOCKED_FOR_SINGLE_TEST_READ", "run_id": args.run_id,
        "config_hash": config_hash(config), "locked_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    config["reranker"] = dict(config["reranker"])
    config["reranker"]["model_dir"] = str(PROJECT_ROOT / config["model"]["reranker_model_dir"])
    config["reranker"]["device"] = args.device
    data = load_canonical(VERSION_ROOT / "inputs")
    runner = Method1Runner(data, artifact_root(args.index_run_id), config, reranker_backend="qwen3")
    results = run_query_matrix(
        runner, data.queries["test"], run_id=args.run_id, full_n=int(config["budget"]["full_n"]), seed_s=int(config["budget"]["seed_s"]),
        frontier_c=int(config["budget"]["frontier_c"]), final_k=int(config["budget"]["final_k"]),
    )
    output = VERSION_ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "query_results.jsonl", (result.to_mapping() for result in results))
    write_json(VERSION_ROOT / "aggregate" / "test_status.json", {"status": "COMPLETED_ONCE", "run_id": args.run_id, "config_hash": frozen.get("config_hash", config_hash(config)), "test_read_status": "read_once", "rows": len(results)})
    print(json.dumps({"status": "COMPLETED_ONCE", "run_id": args.run_id, "rows": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
