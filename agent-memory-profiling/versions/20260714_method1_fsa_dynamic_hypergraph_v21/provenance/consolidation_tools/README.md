# Consolidation Tools

These four shell scripts were used once on 2026-07-17 to construct the
consolidated package, deduplicate the compact archive, and write the migration
manifests. They are retained as audit evidence only. They are not part of the
active v21 pipeline and must not be rerun after the former sibling version
directories have been removed or the motivation packages have been moved to
`../../../motivation_experiments/`.

The final outputs are:

- `../legacy_versions/SELECTION_INDEX.tsv`
- `../legacy_versions/DEDUP_INDEX.tsv`
- `../../aggregate/MIGRATION_MANIFEST_20260717.tsv`
- `../../aggregate/migrated_manifests/`
