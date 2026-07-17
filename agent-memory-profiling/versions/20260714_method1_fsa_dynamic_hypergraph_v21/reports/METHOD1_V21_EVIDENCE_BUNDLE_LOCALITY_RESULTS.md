# Method 1 v21 Evidence-Bundle Locality OOF Results

This is train-only supervision-derived compositional evidence locality, not observed runtime co-access. The scorer is evaluated outer-OOF; inference uses no evidence group, acceptable member, query ID, or gold bundle.

| Comparator | Subset | Metric | Effect | Session-clustered 95% CI |
| --- | --- | --- | ---: | --- |
| independent_member | 2 | gold_compatible_bundle_hit_rate | -0.002521 | [-0.007595, +0.002488] |
| independent_member | 2 | all_groups_covered_rate | -0.010084 | [-0.016639, -0.004146] |
| independent_member | 2 | bundle_recall | -0.031092 | [-0.043318, -0.017854] |
| independent_member | 2 | bundle_member_redundancy | -0.004062 | [-0.006879, -0.001535] |
| independent_member | 3 | gold_compatible_bundle_hit_rate | -0.002041 | [-0.006356, +0.000000] |
| independent_member | 3 | all_groups_covered_rate | -0.004082 | [-0.010331, +0.000000] |
| independent_member | 3 | bundle_recall | -0.010884 | [-0.025956, +0.003381] |
| independent_member | 3 | bundle_member_redundancy | -0.015986 | [-0.025102, -0.007728] |
| independent_member | 4 | gold_compatible_bundle_hit_rate | +0.000000 | [+0.000000, +0.000000] |
| independent_member | 4 | all_groups_covered_rate | +0.000000 | [+0.000000, +0.000000] |
| independent_member | 4 | bundle_recall | +0.030000 | [+0.000000, +0.070000] |
| independent_member | 4 | bundle_member_redundancy | -0.010000 | [-0.043333, +0.020000] |
| independent_member | K>=2 | gold_compatible_bundle_hit_rate | -0.002312 | [-0.006002, +0.001161] |
| independent_member | K>=2 | all_groups_covered_rate | -0.008092 | [-0.013258, -0.003466] |
| independent_member | K>=2 | bundle_recall | -0.023603 | [-0.033702, -0.013785] |
| independent_member | K>=2 | bundle_member_redundancy | -0.007611 | [-0.011003, -0.004585] |
| shuffled_bundle_deepsets | 2 | gold_compatible_bundle_hit_rate | -0.002521 | [-0.006938, +0.001676] |
| shuffled_bundle_deepsets | 2 | all_groups_covered_rate | -0.005042 | [-0.010934, +0.000000] |
| shuffled_bundle_deepsets | 2 | bundle_recall | +0.006723 | [-0.004738, +0.019600] |
| shuffled_bundle_deepsets | 2 | bundle_member_redundancy | +0.001821 | [-0.000139, +0.003853] |
| shuffled_bundle_deepsets | 3 | gold_compatible_bundle_hit_rate | -0.002041 | [-0.006356, +0.000000] |
| shuffled_bundle_deepsets | 3 | all_groups_covered_rate | -0.002041 | [-0.006276, +0.000000] |
| shuffled_bundle_deepsets | 3 | bundle_recall | +0.000680 | [-0.013662, +0.015129] |
| shuffled_bundle_deepsets | 3 | bundle_member_redundancy | -0.003741 | [-0.008827, +0.001029] |
| shuffled_bundle_deepsets | 4 | gold_compatible_bundle_hit_rate | +0.000000 | [+0.000000, +0.000000] |
| shuffled_bundle_deepsets | 4 | all_groups_covered_rate | +0.000000 | [+0.000000, +0.000000] |
| shuffled_bundle_deepsets | 4 | bundle_recall | +0.010000 | [-0.015000, +0.040000] |
| shuffled_bundle_deepsets | 4 | bundle_member_redundancy | -0.006667 | [-0.020000, +0.000000] |
| shuffled_bundle_deepsets | K>=2 | gold_compatible_bundle_hit_rate | -0.002312 | [-0.005760, +0.000591] |
| shuffled_bundle_deepsets | K>=2 | all_groups_covered_rate | -0.004046 | [-0.008523, +0.000000] |
| shuffled_bundle_deepsets | K>=2 | bundle_recall | +0.005106 | [-0.003939, +0.014724] |
| shuffled_bundle_deepsets | K>=2 | bundle_member_redundancy | +0.000000 | [-0.001919, +0.002133] |
| pairwise_only | 2 | gold_compatible_bundle_hit_rate | -0.001681 | [-0.005988, +0.002604] |
| pairwise_only | 2 | all_groups_covered_rate | +0.001681 | [-0.004102, +0.007576] |
| pairwise_only | 2 | bundle_recall | +0.020168 | [+0.008152, +0.032498] |
| pairwise_only | 2 | bundle_member_redundancy | +0.000140 | [-0.001882, +0.002202] |
| pairwise_only | 3 | gold_compatible_bundle_hit_rate | +0.000000 | [+0.000000, +0.000000] |
| pairwise_only | 3 | all_groups_covered_rate | -0.004082 | [-0.010352, +0.000000] |
| pairwise_only | 3 | bundle_recall | +0.010204 | [-0.005219, +0.025696] |
| pairwise_only | 3 | bundle_member_redundancy | -0.006122 | [-0.012058, -0.000643] |
| pairwise_only | 4 | gold_compatible_bundle_hit_rate | +0.000000 | [+0.000000, +0.000000] |
| pairwise_only | 4 | all_groups_covered_rate | +0.000000 | [+0.000000, +0.000000] |
| pairwise_only | 4 | bundle_recall | +0.000000 | [-0.045000, +0.045000] |
| pairwise_only | 4 | bundle_member_redundancy | -0.013333 | [-0.033333, +0.000000] |
| pairwise_only | K>=2 | gold_compatible_bundle_hit_rate | -0.001156 | [-0.004554, +0.002229] |
| pairwise_only | K>=2 | all_groups_covered_rate | +0.000000 | [-0.004108, +0.004140] |
| pairwise_only | K>=2 | bundle_recall | +0.016763 | [+0.007209, +0.025914] |
| pairwise_only | K>=2 | bundle_member_redundancy | -0.002023 | [-0.004122, +0.000186] |

Calibration and formal test remained unread. Dense QHDS remains a quality reference; this report does not claim a sparse-path or hardware result.
