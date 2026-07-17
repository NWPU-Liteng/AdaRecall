# Method 1 v21 Oracle Headroom Results

The gold-coverage oracle is a non-deployable train-query OOF upper-bound analysis. It may read heldout train evidence groups within the five session-blocked folds, but it never reads calibration or formal-test queries. Protected ANN seeds are mandatory in every budget.

| B | Group coverage | All groups covered | Budget-infeasible rate | Residual groups after protected seeds |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.999631 | 0.998524 | 0.001476 | 0.782 |
| 16 | 1.000000 | 1.000000 | 0.000000 | 0.782 |
| 32 | 1.000000 | 1.000000 | 0.000000 | 0.782 |

This oracle reports headroom only and is not an online method, a retrieval claim, or a deployable selector.
