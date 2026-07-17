# Method 1 v21 Calibration CRC/LTT Mapping

This is the single permitted calibration read: 290 calibration queries were evaluated after the pre-calibration freeze. The fixed policy was selected from OOF before this read; calibration only maps CRC/LTT risk and selects B.

- calibration access count: `1`
- formal-test runs: `0`
- fixed grid: `(4, 8, 8)`
- CRC-selected B: `32`
- ANN seed sources: `{'frozen_memory_embedding_flat_prefix_fallback': 290}`

| B | Frontier risk | 95% upper | CRC target | CRC pass | Final top-5 risk | 95% upper | LTT target | LTT after Holm |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 8 | 0.565517 | 0.614453 | 0.376199 | False | 0.589655 | 0.638042 | 0.295062 | False |
| 16 | 0.555172 | 0.604308 | 0.376199 | False | 0.589655 | 0.638042 | 0.295062 | False |
| 32 | 0.551724 | 0.600922 | 0.376199 | False | 0.589655 | 0.638042 | 0.295062 | False |

No model, prototype, selector, hyperparameter, payload layout, or dynamic policy was changed. Formal test remains locked for the final one-run operation.
