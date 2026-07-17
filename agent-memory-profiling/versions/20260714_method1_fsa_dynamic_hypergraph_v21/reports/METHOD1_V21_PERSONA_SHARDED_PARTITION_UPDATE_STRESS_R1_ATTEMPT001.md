# Method 1 v21 Persona-Sharded Static New-Memory Partition Stress

Memory rows were deterministically partitioned by SHA-256 memory ID into a frozen base and a new-memory pool. This validates static insertion, deletion, atomic publication, and compaction equivalence; it makes no timestamp, access-order, causal, or PIM claim.

| growth | appended | deleted | overlay lookup ms | compaction ms | dispatch mismatch | compact mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 420 | 42 | 1.506 | 149.569 | 0 | 0 |
| 5% | 2102 | 210 | 1.714 | 108.630 | 0 | 0 |
| 10% | 4203 | 420 | 2.023 | 109.420 | 0 | 0 |
| 20% | 8407 | 841 | 2.897 | 125.604 | 0 | 0 |

The serving adapter and compacted snapshot returned identical bounded persona/prototype edges and B=8 selections for every probe.
