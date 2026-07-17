# Method 1 v21 cap64 Agent Answer OOF

Train-only OOF evaluation with local Qwen, a fixed 1,800-word context cap, and at most 48 generated tokens. Calibration and formal test were not read.

| budget | answer accuracy | prompt tokens | completion tokens | retrieval ms | generation ms | total Agent ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.259041 | 2538.4 | 48.0 | 28.637 | 3284.509 | 3313.146 |
| 16 | 0.259041 | 2538.4 | 48.0 | 28.637 | 3285.134 | 3313.772 |
| 32 | 0.259041 | 2538.4 | 48.0 | 28.653 | 3283.446 | 3312.099 |

This is final-answer execution on the authorized train-OOF split, not formal heldout performance. Token cost is reported as local-model token counts, not API currency cost.
