# Status

Updated: 2026-07-05

## Initial State

The v4 package was created as a sibling of v3. Immutable v3 artifacts are reused by symlink where possible. New v4 scripts, reports, figures, manifests, and derived aggregates will be written under this directory only.

## Claim Boundary

Allowed:

- Flat frontier expansion increases selection and reranking cost.
- Larger frontiers improve agreement with a high-budget reranking reference.
- Chronological placement scatters selected memories across tiles and creates physical over-fetch headroom.
- Under a fixed auditable controller, stages exhibit non-random access locality.
- If validated by held-out testing, some stage-conditioned structure transfers from earlier to later accesses.

Forbidden:

- hypergraph superiority;
- PIM speedup or energy reduction;
- final layout implementation;
- answer-quality improvement from larger frontier budgets;
- source-provided or benchmark-provided cognitive phases;
- oracle tile packing as an implemented baseline;
- automatic keypoint alignment as human-verified gold evidence.
