# Motivation v5 Validation

Validation is recorded in `aggregate/v5_validation.json` and
`aggregate/v5_validation.csv`.

- S1 retains all 1,355 train query plans and all six query types.
- S1 has 34.24% multi-group queries; the multi-hop B=8 all-group rate is below
  the preregistered workload guardrail.
- S2 covers 1/5/10/20% growth, has zero lookup mismatches, and stays below the
  changed-posting guardrail at 20% growth.
- M3 non-oracle layouts remain no better than the oracle packing lower bound.
- Five v5 figures are present.
- Stage/workflow, method superiority, PIM, causal access, and dynamic access
  chain claims are disabled.

The O1 controller-stage artifacts from v4 are retained as appendix material and
are excluded from the main motivation closure.
