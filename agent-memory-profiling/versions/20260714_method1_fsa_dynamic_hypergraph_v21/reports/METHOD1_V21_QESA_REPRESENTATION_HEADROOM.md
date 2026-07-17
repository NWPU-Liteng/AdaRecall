# Method 1 v21 QESA Representation Headroom

This is Oracle A, a gold evidence coverage oracle used only for train-OOF representation headroom and error attribution. It is not deployable and cannot be used by QESA inference, calibration, or formal test.

| Budget | Oracle coverage | Oracle K>=2 coverage | Oracle minus flat | K>=2 oracle minus flat |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.583752 | 0.333478 | +0.050295 | +0.027601 |
| 16 | 0.737970 | 0.510838 | +0.114600 | +0.115848 |
| 32 | 0.869397 | 0.717437 | +0.161956 | +0.207081 |

The oracle reads heldout OOF evidence groups only to ask whether the frozen prototype incidence can express useful rankings. It does not establish learned generalization.
