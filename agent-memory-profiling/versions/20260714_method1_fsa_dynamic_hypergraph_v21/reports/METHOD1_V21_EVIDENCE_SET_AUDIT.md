# Method 1 v21 Evidence-Set Audit

The train-only causal availability audit completed. Every accepted evidence member is in the corresponding query's `eligible_memory_ids`; this validates the training incidence as a past-available evidence relation. It does not create a temporal read order.

| Measure | Value |
| --- | ---: |
| queries | 1355 |
| evidence_groups | 1819 |
| evidence_member_incidences | 3638 |
| queries_with_two_or_more_groups | 346 |
| members_per_group | `{2: 1819}` |
| groups_per_query | `{1: 1009, 2: 238, 3: 98, 4: 10}` |
| source_session_lags | `{'count': 3638, 'min': 0, 'p50': 14, 'max': 79, 'same_session': 196}` |

Same-session items are not treated as leakage: the supplied eligibility list is the causal authority and can include earlier turns in the same session. Calibration and formal test were not read.
