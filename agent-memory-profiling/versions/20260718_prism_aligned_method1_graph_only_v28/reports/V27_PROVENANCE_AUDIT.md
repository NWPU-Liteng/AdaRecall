# v27 Provenance Audit

Status: `PROVENANCE_CONFLICT`

The v27 checkpoint is retained only as an exploratory record. It is forbidden
for v28 promotion, parameter selection, confirmation, or paper numbers.

## Machine evidence

The handoff file was written at `2026-07-18 22:30:56 +0800` and records a
16-seed + 16-graph aggregate SHA256 of
`51f3a05e0679c27f7c2ef1016a17113a45690eee3d05353ed6b01973cf5b094e`.
The current aggregate was written later, at `2026-07-18 23:17:16 +0800`, has
SHA256 `09203e196d2c183ca7daf9b9cac4ea085ba65964c4ede468f4b2dd542929949b`,
and contains the obsolete `mean_additions` schema rather than aggregate
`graph_supported_slots`, `graph_novel_vs_dense32`, and
`dense_fallback_slots` fields.

The config, source, and runner still match the handoff hashes:

| Object | SHA256 |
|---|---|
| `configs/v27.json` | `92f3393cf88e36f898419ea21e18b897f5b29d996defee8ea055b1ff658bb61c` |
| `src/v27/core.py` | `b4175a1dbe385c0f049a803421b864efbb30933ad6d27d3d004933e6b63328fa` |
| `scripts/01_run_dual_objective_oof.py` | `4d32d1a4a8da40ee1229d6f0fa72bfe6704ca63c75291c503596c5f11d1b6c66` |
| current `relevance_rows.jsonl` | `f19aaddbd5081e2302cc1c5b7e8aa938a57d5908e61daa3939e8f96f2446f0cb` |
| current `layout_rows.jsonl` | `a5ea98b84301a436d05781506fe7a72548740ecb37d0686f9200335b41debb66` |
| current aggregate | `09203e196d2c183ca7daf9b9cac4ea085ba65964c4ede468f4b2dd542929949b` |

The v27 directory is Git-ignored and has no tracked commit or preserved formal
stdout/stderr. No rows SHA was registered in the handoff. Although the current
relevance rows predate the handoff, there is no immutable machine link from
those rows to the recorded `51f3a05e...` aggregate. The later layout and
aggregate timestamps demonstrate an overwrite after the handoff.

## Decision

The conflict cannot be resolved by choosing the more favorable file or by
reconstructing an unregistered aggregate. v28 starts from a new config, source
manifest, immutable run ID, and causal implementation. Its aggregator refuses
to write into an existing run directory.
