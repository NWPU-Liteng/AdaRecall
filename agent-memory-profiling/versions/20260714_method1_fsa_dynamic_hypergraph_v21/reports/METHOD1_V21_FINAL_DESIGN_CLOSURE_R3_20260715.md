# Method 1 v21 Final Design Closure r3

This revision closes the deterministic static new-memory publication path in addition to the retrieval, Tile, matched e2e, and Agent train-OOF results in r2.

## Dynamic-partition design

Memory IDs are assigned to a frozen base or a new-memory pool by deterministic SHA-256 partitioning. No timestamp, access order, co-access trace, or causal update interpretation is used. The base contains `33627` memories and the new-memory pool contains `8407`.

At 1%, 5%, 10%, and 20% growth, the persona-sharded serving adapter, overlay lookup, compacted lookup, and B=8 selection all matched exactly for `256` probes per point. At 20% growth, overlay lookup averaged `2.897 ms`; atomic publish was `11.461 ms`; full snapshot build was `6181.4 ms`; compaction was `125.6 ms`. All mismatch counters were zero.

This closes the **static incremental publication** requirement. It does not establish causal dynamic-memory behavior, which the source data cannot support.

## Overall state

The default retrieval tier remains `cap=64`, active16, B8. Active24 is an optional quality tier; active32 remains diagnostic because of its latency. Persona sharding, corrected host/logical Tile accounting, matched e2e controls, static partition publication, and Agent train-OOF execution are complete. Agent answer accuracy remains `0.259041`; balanced packing and lexical handoff did not improve it. Formal heldout performance remains unavailable.

The remaining work is optimization and independent generalization validation, not a missing static publication contract. No calibration/formal split was read and no minimum-effect threshold was used.
