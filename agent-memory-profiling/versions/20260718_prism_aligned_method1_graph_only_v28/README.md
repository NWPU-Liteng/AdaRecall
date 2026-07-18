# Method 1 v28: Causal Graph-only Closure

Status: `GRAPH_ONLY_QWEN_FAILED`

This version is the non-overwriting successor to the v27 exploratory screen.
It implements the frozen Method 1 path without memory-level ANN, query-memory
candidate scoring, Dense protected slots, or Dense fallback. Query vectors may
only score causal semantic edge descriptors. Every returned candidate must
carry an active-edge graph provenance record.

The v27 aggregate is quarantined as `PROVENANCE_CONFLICT`: its current SHA and
schema do not match the handoff checkpoint, and the conflicting file was
written after the handoff. It is not promotion or paper evidence. See
`reports/V27_PROVENANCE_AUDIT.md`.

Scientific boundaries:

- MedMemoryBench train is development evidence via strict five-fold OOF only.
- MedMemoryBench development/test and the consumed LoCoMo source are forbidden.
- Formal runs require an immutable, previously absent `runs/<run_id>` directory.
- Gate failure is preserved. Dense retrieval cannot repair Graph-only output.

The immutable attempt-003 Frontier OOF run passed Gate C for both Semantic and
Dual Graph-only, with all 32 slots graph-owned and no Dense fallback. The
schema-corrected full-ranking Qwen run then failed Gate D for both methods.
Fresh evaluation is therefore locked. Compact result hashes and the immutable
failure decision are registered in
`artifacts/frozen/qwen_fullranking_failure_receipt.json`.
