# M3 Interpretation

M3 formal completed with 2160 rows: 240 formal queries, 3 tile capacities, and 3 layouts.

## Main Result

Chronological placement scatters selected top-5 memories across multiple tiles:

| Tile capacity | Chronological activated tiles | Chronological access amplification | Oracle tile-packing lower-bound activated tiles |
|---:|---:|---:|---:|
| 8 | 4.50 | 7.21 | 1.00 |
| 16 | 4.21 | 13.48 | 1.00 |
| 32 | 3.84 | 24.56 | 1.00 |

Random per-persona placement is worse than chronological at all tested capacities.

## Supported Claim

Chronological physical placement creates over-fetch headroom: top-ranked memories that are jointly selected by flat retrieval are spread across several chronological tiles.

## Claim Boundary

The oracle tile-packing lower bound is not an implemented method and not a deployable layout. M3 motivates structural placement but does not show that any proposed layout or PIM runtime is superior.
