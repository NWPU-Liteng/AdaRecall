# Method 1 v21 UOT-FSA Results

Gate A status: `passed`. All 25 train-only outer OOF units completed with converged, non-collapsed final transport: `True`.

| Seed | UOT-FSA | cosine-kNN | Difference | Paired-query 95% CI | Session-clustered 95% CI |
| --- | ---: | ---: | ---: | --- | --- |
| 20260712 | 0.861070 | 0.769496 | +0.091574 | [+0.068141, +0.115194] | [+0.066111, +0.117693] |
| 20260713 | 0.857565 | 0.769496 | +0.088069 | [+0.064207, +0.111625] | [+0.062924, +0.114220] |
| 20260714 | 0.853198 | 0.769496 | +0.083702 | [+0.060023, +0.107320] | [+0.057238, +0.110214] |
| 20260715 | 0.866790 | 0.769496 | +0.097294 | [+0.075769, +0.119434] | [+0.071880, +0.123190] |
| 20260716 | 0.870910 | 0.769496 | +0.101415 | [+0.078167, +0.125217] | [+0.077216, +0.126295] |

The numerical screen only excluded unstable recipes. Adapter and UOT choices were made independently inside each outer unit using train-only inner validation. The metric is evidence-group co-location, not final retrieval quality or a temporal/co-access claim. Calibration and formal test remain unread.

Selected recipe counts: `{'7': 7, '16': 10, '25': 8}`.

Gate A passing permits QESA development; it does not establish the later hypergraph quality-cost attribution claim.
