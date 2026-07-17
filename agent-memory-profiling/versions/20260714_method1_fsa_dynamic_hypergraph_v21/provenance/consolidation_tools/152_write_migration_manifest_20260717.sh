#!/usr/bin/env bash
set -euo pipefail

VERSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$VERSION_ROOT/../.." && pwd)"
VERSIONS_ROOT="$PROJECT_ROOT/versions"
OUTPUT="$VERSION_ROOT/aggregate/MIGRATION_MANIFEST_20260717.tsv"

printf 'kind\tsource\tcanonical\tbytes\tsha256\tnotes\n' > "$OUTPUT"

record() {
  local kind="$1" source="$2" canonical="$3" notes="$4"
  if [[ ! -f "$canonical" ]]; then
    printf 'missing\t%s\t%s\t0\t-\t%s\n' "$source" "$canonical" "$notes" >> "$OUTPUT"
    return
  fi
  local bytes digest
  bytes="$(stat -c '%s' "$canonical")"
  read -r digest _ < <(sha256sum "$canonical")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$kind" "$source" "$canonical" "$bytes" "$digest" "$notes" >> "$OUTPUT"
}

record input "$VERSIONS_ROOT/20260704_method1_motivation_medmemorybench_v3/aggregate/memory_items_turn_pair.jsonl" \
  "$VERSION_ROOT/inputs/memory_medmemorybench_v3/memory_items_turn_pair.jsonl" memory_corpus
record input "$VERSIONS_ROOT/20260704_method1_motivation_medmemorybench_v3/aggregate/embedding_cache/f5a55a7f678df5fa.npy" \
  "$VERSION_ROOT/inputs/memory_medmemorybench_v3/embedding_cache/embeddings.npy" memory_embeddings
record input "$VERSIONS_ROOT/20260704_method1_motivation_medmemorybench_v3/aggregate/embedding_cache/f5a55a7f678df5fa.ids.json" \
  "$VERSION_ROOT/inputs/memory_medmemorybench_v3/embedding_cache/ids.json" memory_embedding_ids
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v6_schedule_layout_candidate/query_split/query_plan_train.jsonl" \
  "$VERSION_ROOT/inputs/qa_basis_v1/query_split/query_plan_train.jsonl" query_plan_train
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v6_schedule_layout_candidate/query_split/query_plan_calibration.jsonl" \
  "$VERSION_ROOT/inputs/qa_basis_v1/query_split/query_plan_calibration.jsonl" query_plan_calibration
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v6_schedule_layout_candidate/query_split/query_plan_test.jsonl" \
  "$VERSION_ROOT/inputs/qa_basis_v1/query_split/query_plan_test.jsonl" query_plan_test_locked
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v15_structured_selector/embeddings/query_vectors.npy" \
  "$VERSION_ROOT/inputs/qa_basis_v1/query_embeddings/query_vectors.npy" query_embeddings
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v15_structured_selector/embeddings/query_index.json" \
  "$VERSION_ROOT/inputs/qa_basis_v1/query_embeddings/query_index.json" query_embedding_index
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v13_ann64_semantic_root/train/retrieved_contexts_flat_graph_hyper.jsonl" \
  "$VERSION_ROOT/inputs/qa_basis_v1/frozen_train/retrieved_contexts_flat_graph_hyper.jsonl" frozen_train_retrieval
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/configs/keypoint_basis_v6_schedule_layout_candidate.yaml" \
  "$VERSION_ROOT/inputs/qa_basis_v1/config/keypoint_basis_v6_schedule_layout_candidate.yaml" eligibility_source
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/aggregate/v19_stop_record.json" \
  "$VERSION_ROOT/inputs/qa_basis_v1/aggregate/v19_stop_record.json" qa_parent_stop_record
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/aggregate/v19_next_gate_manifest.json" \
  "$VERSION_ROOT/inputs/qa_basis_v1/aggregate/v19_next_gate_manifest.json" qa_parent_gate_manifest
record input "$VERSIONS_ROOT/20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v6_schedule_layout_candidate/query_split/query_split_manifest.json" \
  "$VERSION_ROOT/inputs/qa_basis_v1/artifacts/keypoint_basis_v6_schedule_layout_candidate/query_split/query_split_manifest.json" qa_split_manifest

while IFS=$'\t' read -r status version relative bytes source; do
  [[ "$status" == "retained" ]] || continue
  canonical="$VERSION_ROOT/provenance/legacy_versions/$version/$relative"
  if [[ ! -f "$canonical" ]]; then
    dedup_row="$(awk -F '\t' -v p="$canonical" '$1 == p {print; exit}' "$VERSION_ROOT/provenance/legacy_versions/DEDUP_INDEX.tsv")"
    if [[ -n "$dedup_row" ]]; then
      IFS=$'\t' read -r removed kept digest dedup_bytes <<< "$dedup_row"
      printf 'deduplicated\t%s\t%s\t%s\t%s\tbyte_identical_to_kept_path\n' "$source" "$kept" "$dedup_bytes" "$digest" >> "$OUTPUT"
      continue
    fi
  fi
  record legacy "$source" "$canonical" "compact_archive"
done < "$VERSION_ROOT/provenance/legacy_versions/SELECTION_INDEX.tsv"

while IFS= read -r source_file; do
  relative="${source_file#"$VERSION_ROOT/provenance/v20_parent/"}"
  case "$relative" in
    v20/*) source="$VERSIONS_ROOT/20260713_method1_fca_dynamic_hypergraph_v20/${relative#v20/}" ;;
    v20r1/*) source="$VERSIONS_ROOT/20260713_method1_fca_dynamic_hypergraph_v20_r1/${relative#v20r1/}" ;;
    *) continue ;;
  esac
  record parent "$source" "$source_file" "v20_parent_provenance"
done < <(find "$VERSION_ROOT/provenance/v20_parent" -type f | sort)

printf 'migration_manifest=%s\n' "$OUTPUT"
