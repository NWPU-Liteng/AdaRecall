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
| rank entries visited | 50818.7 | 2239.0 | -95.59% |
| metadata bytes accessed | 203274.6 | 8956.2 | -95.59% |
| metadata 4KiB Tile activations | 57.668 | 16.001 | -72.25% |
| candidate feature Tile activations | 52.577 | 52.577 | +0.00% |
| selected payload Tile activations | 6.156 | 6.156 | +0.00% |
| corrected total logical Tiles | 110.276 | 68.609 | -37.78% |
| sparse lookup mean | 1.296 ms | 0.716 ms | -44.70% |

The old `52.61` value is not directly comparable: it mapped returned members to neutral payload Tiles while charging the full 10.76 MB rank index on every query. The corrected metric separates actual rank-prefix blocks, candidate feature Tiles, and selected payload Tiles.

The rank storage size is unchanged because sharding reorders the same entries. No feature or payload replication is used in r2. These are host/logical measurements, not PIM hardware results.
