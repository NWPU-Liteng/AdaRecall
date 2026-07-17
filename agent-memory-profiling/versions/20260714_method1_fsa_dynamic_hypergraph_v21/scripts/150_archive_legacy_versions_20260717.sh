#!/usr/bin/env bash
set -euo pipefail

# Copy only auditable, non-cache material into the v21 provenance package.
# Files are hard-linked while the old packages still exist, so deleting an old
# package does not duplicate the retained bytes. Large raw run outputs and
# checkpoints are intentionally excluded; their summaries/manifests remain.

VERSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSIONS_ROOT="${VERSION_ROOT%/versions/*}/versions"
DEST_ROOT="$VERSION_ROOT/provenance/legacy_versions"
INDEX="$DEST_ROOT/SELECTION_INDEX.tsv"

versions=(
  20260602_172552
  20260603_154211
  20260616_phase_aware_real_agent_v3_formal
  20260621_real_trace_adaptive_phase_v4_formal
  20260622_memorycd_candidate_frontier_v5_1_formal
  20260622_memorycd_native_v5_formal
  20260703_method1_motivation
  20260704_method1_motivation_medmemorybench_v3
  20260705_method1_motivation_medmemorybench_v4_completion
  20260710_method1_access_basis_v1
  20260710_method1_end_to_end_pilot_v1
  20260711_method1_qa_basis_formal_v1
  20260715_method1_motivation_medmemorybench_v5_alignment
  20260716_method1_motivation_medmemorybench_v5p1_controlled
  v0.1
  v0.2
  v0.3
  v0.4
)

mkdir -p "$DEST_ROOT"
printf 'status\tversion\trelative_path\tbytes\tsource\n' > "$INDEX"

for version in "${versions[@]}"; do
  source_root="$VERSIONS_ROOT/$version"
  target_root="$DEST_ROOT/$version"
  [[ -d "$source_root" ]] || continue
  while IFS= read -r -d '' source_file; do
    [[ -L "$source_file" ]] && continue
    relative="${source_file#"$source_root"/}"
    case "$relative" in
      *.md|README*|STATUS*|RESULTS*|LIMITATIONS*|FORMAL_RUN_RESULTS*|*.sha256|\
      aggregate/*|reports/*|configs/*|manifests/*|scripts/*|src/*|code/*|figures/*|\
      */reports/*|*/aggregate/*|*/configs/*|*/manifests/*|*/scripts/*|*/src/*|*/code/*|\
      */figures/*)
        ;;
      *)
        continue
        ;;
    esac
    bytes="$(stat -c '%s' "$source_file")"
    # Keep compact summaries and figures; avoid raw traces, shards and model files.
    (( bytes <= 20971520 )) || continue
    target_file="$target_root/$relative"
    mkdir -p "$(dirname "$target_file")"
    if [[ ! -e "$target_file" ]]; then
      cp -al "$source_file" "$target_file"
      printf 'retained\t%s\t%s\t%s\t%s\n' "$version" "$relative" "$bytes" "$source_file" >> "$INDEX"
    else
      printf 'same_source_path\t%s\t%s\t%s\t%s\n' "$version" "$relative" "$bytes" "$source_file" >> "$INDEX"
    fi
  done < <(find "$source_root" -type f -print0)
done

printf 'archive_root=%s\n' "$DEST_ROOT"
printf 'selection_index=%s\n' "$INDEX"
