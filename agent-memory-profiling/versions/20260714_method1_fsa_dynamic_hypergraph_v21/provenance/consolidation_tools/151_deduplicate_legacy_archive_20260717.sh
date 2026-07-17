#!/usr/bin/env bash
set -euo pipefail

VERSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$VERSION_ROOT/provenance/legacy_versions"
INDEX="$ARCHIVE/DEDUP_INDEX.tsv"

printf 'removed_path\tkept_path\tsha256\tbytes\n' > "$INDEX"
declare -A first_path

while IFS= read -r -d '' file; do
  [[ "$file" == "$ARCHIVE/SELECTION_INDEX.tsv" ]] && continue
  [[ "$file" == "$ARCHIVE/DEDUP_INDEX.tsv" ]] && continue
  read -r digest _ < <(sha256sum "$file")
  if [[ -n "${first_path[$digest]+x}" ]]; then
    kept="${first_path[$digest]}"
    bytes="$(stat -c '%s' "$file")"
    printf '%s\t%s\t%s\t%s\n' "$file" "$kept" "$digest" "$bytes" >> "$INDEX"
    rm -f -- "$file"
  else
    first_path[$digest]="$file"
  fi
done < <(find "$ARCHIVE" -type f -print0 | sort -z)

printf 'removed=%s\n' "$(($(wc -l < "$INDEX") - 1))"
printf 'index=%s\n' "$INDEX"
