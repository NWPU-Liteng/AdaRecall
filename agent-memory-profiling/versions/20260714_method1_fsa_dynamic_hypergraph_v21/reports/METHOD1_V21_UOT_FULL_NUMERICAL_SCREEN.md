# Method 1 v21 UOT Full Numerical Screen

Run `v21_uot_screen_r7_20260714_cuda2` mechanically merged all 27 preregistered train-only numerical recipes. The merge validates recipe coverage, configuration and input hashes, per-recipe artifacts, finite/nonnegative transport diagnostics, and H_sem row normalization. It performs no UOT recipe or model selection.

| Measure | Value |
| --- | ---: |
| recipes | 27 |
| primary converged (`potential_delta <= 1e-6` within 300 iterations) | 6 |
| primary nonconverged but retained as numerical results | 21 |
| calibration reads | 0 |
| formal-test reads | 0 |

The raw unbalanced transport plan is not required to have balanced row or column marginals. Normalization is checked only after conversion to the H_sem semantic incidence.

## Legacy Exclusion

- `aggregate/v21_uot_full_preregistered_numerical_screen.csv`: legacy monolithic aggregate has no independent recipe attempts, provenance hashes, or required secondary diagnostics
- `aggregate/v21_uot_full_preregistered_numerical_screen_part_01_03.csv`: legacy overlapping partial range
- `aggregate/v21_uot_full_preregistered_numerical_screen_part_01_04.csv`: legacy overlapping partial range
- `aggregate/v21_uot_full_preregistered_numerical_screen_part_04_04.csv`: preserved legacy recipe-4 numerical reference; not an r2 compliant shard
