from __future__ import annotations

import argparse
import json

from common import VERSION_ROOT, add_common_args, config_hash, load_config
from v24.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    gate_path = VERSION_ROOT / "aggregate" / "development_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError("development_gate.json is missing; run development first")
    gate = read_json(gate_path)
    frozen = {
        "status": "FROZEN_FOR_TEST" if gate.get("passed") else "NOT_PROMOTED",
        "config_hash": config_hash(config), "development_run_id": gate.get("run_id"),
        "development_gate": gate, "test_read_status": "not_read",
        "resolved_config": config,
    }
    if gate.get("passed"):
        write_json(VERSION_ROOT / "artifacts" / "frozen" / "development_config.json", frozen)
    else:
        write_json(VERSION_ROOT / "negative_results" / "development_not_promoted.json", frozen)
    print(json.dumps({"status": frozen["status"], "config_hash": frozen["config_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
