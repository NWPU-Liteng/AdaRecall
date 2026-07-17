# Method 1 v21 Final Freeze

The Method 1 policy is frozen after the single calibration CRC/LTT read. All pre-calibration source hashes remain unchanged; the calibration result did not pass the declared risk target, so B=32 is retained as the fallback budget.

- fixed grid: `[4, 8, 8]`
- frozen budget: `32`
- CRC pass: `False`
- LTT after Holm: `False`
- dynamic policy: `static_fsa_qehg_fallback`
- calibration access count: `1`
- formal-test runs: `0`

No model, prototype, selector, hyperparameter, split, payload layout, or dynamic policy may change after this freeze. The formal test is permitted exactly once and will be recorded even if the frozen corpus cannot provide a legal test embedding.
