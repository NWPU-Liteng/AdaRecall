#!/usr/bin/env bash
set -euo pipefail

VERSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$VERSION_ROOT/aggregate/migrated_manifests"
mkdir -p "$OUT"

OLD_ROOT="/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions"
DATA_HASH="$(sha256sum "$VERSION_ROOT/configs/v21_data.json" | awk '{print $1}')"

migrate() {
  local source="$1" target="$2"
  jq --arg old_root "$OLD_ROOT" \
     --arg v3 "inputs/memory_medmemorybench_v3" \
     --arg qa "inputs/qa_basis_v1" \
     --arg v20 "provenance/v20_parent/v20r1" \
     --arg data_hash "$DATA_HASH" \
     'walk(if type == "string" then
      gsub($old_root + "/20260704_method1_motivation_medmemorybench_v3/aggregate/"; $v3 + "/") |
      gsub($old_root + "/20260711_method1_qa_basis_formal_v1/"; $qa + "/") |
        gsub($old_root + "/20260713_method1_fca_dynamic_hypergraph_v20_r1/"; $v20 + "/") |
        gsub("../20260711_method1_qa_basis_formal_v1/"; $qa + "/") |
        gsub("20260711_method1_qa_basis_formal_v1/"; $qa + "/") |
        gsub("20260704_method1_motivation_medmemorybench_v3/"; $v3 + "/") |
        gsub("inputs/qa_basis_v1/artifacts/keypoint_basis_v6_schedule_layout_candidate/query_split/"; $qa + "/query_split/") |
        gsub("inputs/qa_basis_v1/artifacts/keypoint_basis_v15_structured_selector/embeddings/"; $qa + "/query_embeddings/") |
        gsub("inputs/qa_basis_v1/artifacts/keypoint_basis_v13_ann64_semantic_root/train/"; $qa + "/frozen_train/") |
        gsub("inputs/qa_basis_v1/configs/keypoint_basis_v6_schedule_layout_candidate.yaml"; $qa + "/config/keypoint_basis_v6_schedule_layout_candidate.yaml") |
        gsub("inputs/memory_medmemorybench_v3/embedding_cache/f5a55a7f678df5fa.ids.json"; $v3 + "/embedding_cache/ids.json") |
        gsub("inputs/memory_medmemorybench_v3/embedding_cache/f5a55a7f678df5fa.npy"; $v3 + "/embedding_cache/embeddings.npy")
      else . end) |
      .consolidated_from = input_filename |
      .consolidated_on = "2026-07-17" |
      (if has("config_hashes") then .config_hashes["v21_data.json"] = $data_hash else . end)' \
    "$source" > "$target"
}

migrate "$VERSION_ROOT/aggregate/v21_manifest.json" "$OUT/v21_manifest_consolidated_20260717.json"
migrate "$VERSION_ROOT/aggregate/v21_formal_test_once_r1_20260714.json" "$OUT/v21_formal_test_once_consolidated_20260717.json"
migrate "$VERSION_ROOT/aggregate/v21_eligible_memory_provenance_audit.json" "$OUT/v21_eligible_memory_provenance_audit_consolidated_20260717.json"

printf 'migrated_manifests=%s\n' "$OUT"
