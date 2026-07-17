# Method 1 v21 QESA Failure Diagnosis

This is a train-only post-Gate diagnostic. It does not revise Gate B and does not read calibration or formal test data. The failed OOF artifacts remain immutable.

| Budget | vs prior | vs flat cosine | vs shuffled QESA | vs same-capacity MLP |
| ---: | ---: | ---: | ---: | ---: |
| 8 | -0.002460 | -0.050012 | +0.002645 | -0.023333 |
| 16 | -0.003752 | -0.123715 | +0.005068 | -0.050959 |
| 32 | -0.009680 | -0.177663 | +0.005547 | -0.078057 |

Prediction diagnostics:

- Mean latent effective-count ratio: `0.122955`.
- Mean prototype L1 distance from its unit mean: `0.042986`.
- Mean prototype-coordinate standard deviation: `0.000784`.
- Near-query-independent warning: `True`.
- Diffuse-latent warning: `False`.

A revision is scientifically justified only if this audit identifies a predeclared representational failure. It must use a new run ID and preserve this negative result. QHDS remains blocked while Gate B is not passed.
