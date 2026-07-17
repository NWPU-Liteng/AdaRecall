# A3 Candidate Path Occupancy Audit

## Frontier budget 8

- `n_ann_raw`: min=32.000, p50=32.000, p90=32.000, p95=32.000, max=32.000, mean=32.000
- `n_after_metadata_filter`: min=32.000, p50=32.000, p90=32.000, p95=32.000, max=32.000, mean=32.000
- `n_after_causal_filter`: min=32.000, p50=32.000, p90=32.000, p95=32.000, max=32.000, mean=32.000
- `n_after_dedup`: min=32.000, p50=32.000, p90=32.000, p95=32.000, max=32.000, mean=32.000
- `n_after_materialization`: min=8.000, p50=8.000, p90=8.000, p95=8.000, max=8.000, mean=8.000
- `n_reranked`: min=8.000, p50=8.000, p90=8.000, p95=8.000, max=8.000, mean=8.000
- `n_final_selected`: min=5.000, p50=5.000, p90=5.000, p95=5.000, max=5.000, mean=5.000
- `frontier_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- `selection_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- fraction `n_reranked < K_f`: 0.0000
- fraction `n_final_selected < K_o`: 0.0000

## Frontier budget 16

- `n_ann_raw`: min=64.000, p50=64.000, p90=64.000, p95=64.000, max=64.000, mean=64.000
- `n_after_metadata_filter`: min=64.000, p50=64.000, p90=64.000, p95=64.000, max=64.000, mean=64.000
- `n_after_causal_filter`: min=64.000, p50=64.000, p90=64.000, p95=64.000, max=64.000, mean=64.000
- `n_after_dedup`: min=64.000, p50=64.000, p90=64.000, p95=64.000, max=64.000, mean=64.000
- `n_after_materialization`: min=16.000, p50=16.000, p90=16.000, p95=16.000, max=16.000, mean=16.000
- `n_reranked`: min=16.000, p50=16.000, p90=16.000, p95=16.000, max=16.000, mean=16.000
- `n_final_selected`: min=5.000, p50=5.000, p90=5.000, p95=5.000, max=5.000, mean=5.000
- `frontier_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- `selection_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- fraction `n_reranked < K_f`: 0.0000
- fraction `n_final_selected < K_o`: 0.0000

## Frontier budget 32

- `n_ann_raw`: min=128.000, p50=128.000, p90=128.000, p95=128.000, max=128.000, mean=128.000
- `n_after_metadata_filter`: min=128.000, p50=128.000, p90=128.000, p95=128.000, max=128.000, mean=128.000
- `n_after_causal_filter`: min=128.000, p50=128.000, p90=128.000, p95=128.000, max=128.000, mean=128.000
- `n_after_dedup`: min=128.000, p50=128.000, p90=128.000, p95=128.000, max=128.000, mean=128.000
- `n_after_materialization`: min=32.000, p50=32.000, p90=32.000, p95=32.000, max=32.000, mean=32.000
- `n_reranked`: min=32.000, p50=32.000, p90=32.000, p95=32.000, max=32.000, mean=32.000
- `n_final_selected`: min=5.000, p50=5.000, p90=5.000, p95=5.000, max=5.000, mean=5.000
- `frontier_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- `selection_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- fraction `n_reranked < K_f`: 0.0000
- fraction `n_final_selected < K_o`: 0.0000

## Frontier budget 64

- `n_ann_raw`: min=200.000, p50=256.000, p90=256.000, p95=256.000, max=256.000, mean=251.062
- `n_after_metadata_filter`: min=200.000, p50=256.000, p90=256.000, p95=256.000, max=256.000, mean=251.062
- `n_after_causal_filter`: min=200.000, p50=256.000, p90=256.000, p95=256.000, max=256.000, mean=251.062
- `n_after_dedup`: min=200.000, p50=256.000, p90=256.000, p95=256.000, max=256.000, mean=251.062
- `n_after_materialization`: min=64.000, p50=64.000, p90=64.000, p95=64.000, max=64.000, mean=64.000
- `n_reranked`: min=64.000, p50=64.000, p90=64.000, p95=64.000, max=64.000, mean=64.000
- `n_final_selected`: min=5.000, p50=5.000, p90=5.000, p95=5.000, max=5.000, mean=5.000
- `frontier_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- `selection_fill_ratio`: min=1.000, p50=1.000, p90=1.000, p95=1.000, max=1.000, mean=1.000
- fraction `n_reranked < K_f`: 0.0000
- fraction `n_final_selected < K_o`: 0.0000

Conclusion:

The audit verifies whether `frontier_budget` changes the real reranked frontier, not only a config label.
