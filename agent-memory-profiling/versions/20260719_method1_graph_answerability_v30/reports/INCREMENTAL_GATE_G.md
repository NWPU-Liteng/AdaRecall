# Causal Delta-Tile and Agent Replay Gate G

Run: `v30_incremental_gate_g_20260719`

Gate G passed its pre-frozen contract. The formal replay processed 30,755
memories in arrival order, 148 legal query/cutoff boundaries, all 1,355 train
OOF queries, and 1,355 frozen-policy Top-5 co-retrieval events. All 148 exact
snapshot hashes matched the independent v28 manifest. Candidate identity,
layout consistency, future, namespace, duplicate, and past-snapshot mutation
violations were all zero.

A separately committed, read-only cross-run verifier rebuilt all 148 snapshots
and compared them with the earlier Gate E run. All 148 immutable semantic
layout hashes and all 1,355 per-query semantic Tile ID lists matched exactly;
candidate and snapshot violations were also zero. This distinguishes logical
cross-run layout identity from the formal runner's physical range/byte
conservation check.

Each insertion performed online Top-3 fuzzy prototype sufficient-statistic
updates and appended the payload, embedding, and metadata to a physical delta
Tile. Insertion P50/P95 was 0.285/0.407 ms. One delta Tile was affected per
insertion, 2.948 semantic edges were affected on average, and mean/P95 write
amplification was 1.021/1.017x. The higher maximum of 1.829x comes from the
first entry in an aligned Tile carrying its padding cost. Total physical delta
writes were 228.66 MB for 224.05 MB of logical payload plus embeddings.

At every query boundary the delta journal was flushed and `fsync`ed, followed
by a deterministic exact graph rebuild and semantic layout materialization.
Fsync P50/P95 was 5.61/109.49 ms. Exact rebuild mean/P95 was
1,342.98/2,681.13 ms; semantic Tile materialization averaged 49.54 ms. Online
insertion time between boundaries averaged 7.41% of exact rebuild time.

This is deliberately named `full_exact_boundary_rebuild`. It is not a local
compaction result. The delta path demonstrates cheap causal ingestion,
bounded affected state, physical write accounting, and exact reproducibility;
the remaining systems optimization is to replace full boundary rebuilding
with a locally exact split/merge compactor without changing graph semantics.

Agent replay events are explicitly `co_retrieval` from a frozen Qwen policy,
are applied only after the current snapshot is frozen, and do not feed current
candidate generation. No native co-use trace exists, so
`native_agent_access_claim=false`.
