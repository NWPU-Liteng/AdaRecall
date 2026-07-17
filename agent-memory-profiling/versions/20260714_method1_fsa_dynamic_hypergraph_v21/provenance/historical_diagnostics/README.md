# Historical Diagnostics

`135_audit_v21_legacy_hyper_layout_headroom.py` produced the retained
`aggregate/v21_legacy_hyper_layout_headroom_*.json` and `.csv` comparison before
the compact consolidation. The all-train legacy layout used by that diagnostic
was intentionally not retained because it overlaps every v21 OOF target fold
and is not deployable.

The script therefore exits with an explicit explanation when the layout file is
absent. The aggregate results remain valid as historical evidence; this folder
is not an active v21 experiment entry point.

