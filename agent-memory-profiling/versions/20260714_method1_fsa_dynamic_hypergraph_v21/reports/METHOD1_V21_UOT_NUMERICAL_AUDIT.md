# Method 1 v21 UOT Numerical Audit

## Status

`partial_preselection_numerical_audit_only`. This is not Gate A and does not
select a final UOT recipe.

## Retained Smoke Result

The initial three-recipe train-only implementation smoke ran on the 42,034 by
768 frozen memory embedding matrix and 1,819 train evidence groups. All three
transport plans were finite and nonnegative, but none met the predeclared
potential-delta tolerance of `1e-6` within 300 Sinkhorn iterations. Their
records remain in `aggregate/v21_uot_numerical_audit.csv`; none is eligible for
model selection.

## Full Predeclared Screen Progress

The configuration declares a 27-recipe numerical grid: three diversity
settings, three epsilon cost-scale multipliers, and three coupled
`(tau_memory, tau_prototype)` pairs. The execution wrapper imposed a short
foreground time limit, so the full grid is being saved as recipe-ID shards.
The completed `screen_04` shard is a predeclared recipe with no diversity,
epsilon multiplier 0.02, and coupled taus (0.25, 0.25). It converged in 263
iterations with potential delta `9.5367431640625e-07`, effective mass domination
0.019435, and finite nonnegative incidence. The remaining recipes have not
been evaluated by this partial artifact.

The GPU invocation initializes the CUDA context in the same Python process
before the screen script runs. This is an execution-environment workaround
only; the tensors, recipe, input hashes, split, and numerical criterion are
unchanged.

## Next Numerical Rule

All 27 recipe IDs must be materialized and mechanically combined before nested
session-blocked OOF Pareto selection. No row in this report, including
`screen_04`, may be promoted based on all-train evidence co-location. No
calibration or formal-test path was read.
