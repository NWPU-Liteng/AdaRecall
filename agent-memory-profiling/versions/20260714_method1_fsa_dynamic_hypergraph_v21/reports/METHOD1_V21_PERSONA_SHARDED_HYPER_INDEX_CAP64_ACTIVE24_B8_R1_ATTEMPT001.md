# Method 1 v21 Persona-Sharded Hyper-Index OOF

All 25 outer-fold units completed without calibration or formal-test access. The persona-sharded index returned exactly the same members as the global prototype rank index for all 6,775 query-seed rows, so retrieval quality and payload are unchanged.

## Core B=32 quality retained

- group recall@32: `0.687491`
- evidence-member recall@32: `0.538553`
- exact task success: `0.613727`
- payload: `118689.5 B`

## Corrected logical storage result

| metric | global prototype rank index | persona x prototype shards | relative change |
|---|---:|---:|---:|
| rank entries visited | 77408.8 | 3415.6 | -95.59% |
| metadata bytes accessed | 309635.3 | 13662.3 | -95.59% |
| metadata 4KiB Tile activations | 87.641 | 24.001 | -72.61% |
| candidate feature Tile activations | 53.282 | 53.282 | +0.00% |
| selected payload Tile activations | 6.137 | 6.137 | +0.00% |
| corrected total logical Tiles | 140.929 | 77.290 | -45.16% |
| sparse lookup mean | 1.900 ms | 0.974 ms | -48.72% |

The old `52.61` value is not directly comparable: it mapped returned members to neutral payload Tiles while charging the full 10.76 MB rank index on every query. The corrected metric separates actual rank-prefix blocks, candidate feature Tiles, and selected payload Tiles.

The rank storage size is unchanged because sharding reorders the same entries. No feature or payload replication is used in r2. These are host/logical measurements, not PIM hardware results.
