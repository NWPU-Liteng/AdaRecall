# B Held-Out Stage Transfer

Chronological split is performed independently per persona: earliest 60% train, latest 40% held-out test.

## Summary

- `ground`: n=106, seen-node coverage=0.449, stage advantage=0.0759, 95% bootstrap CI=[0.0380, 0.1194]
- `plan`: n=106, seen-node coverage=0.515, stage advantage=0.3700, 95% bootstrap CI=[0.2342, 0.5210]
- `reason`: n=106, seen-node coverage=0.425, stage advantage=0.0358, 95% bootstrap CI=[-0.0025, 0.0736]
- `reflect`: n=106, seen-node coverage=0.475, stage advantage=0.0752, 95% bootstrap CI=[0.0368, 0.1155]
- `aggregate`: n=424, seen-node coverage=0.466, stage advantage=0.1392, 95% bootstrap CI=[0.1012, 0.1816]

## Stage-Shuffle Null

- observed aggregate statistic: 0.1392
- null mean: -0.0004
- null 95% interval: [-0.0214, 0.0185]
- empirical p-value: 0.0010
- standardized effect size: 13.6842

## Guardrail Interpretation

Earlier controller-defined stage-conditioned access structure shows measurable transfer to later held-out accesses.
Transfer is concentrated in planning-stage retrieval.
