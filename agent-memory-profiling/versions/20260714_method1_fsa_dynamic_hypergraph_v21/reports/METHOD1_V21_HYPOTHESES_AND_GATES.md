# Method 1 v21 Hypotheses And Gates

## Predeclared Scope

FSA-DyHG uses memory content, train-only evidence groups, frozen query/memory
embeddings, and protected ANN seeds. Stage/workflow labels, access-order
proxies, recency score weights, calibration labels, and test labels are not
model inputs. The projection adapter is trained before UOT and frozen before
prototype learning. The reranker remains frozen.

## H1: Fuzzy Semantic Headroom

At matched candidate/structure size, UOT-FSA with a train-only multi-positive
projection produces better heldout evidence-group co-location than raw cosine
kNN, hard K-means, and a clique-expanded pairwise semantic graph. A memory may
have multiple continuous memberships; no top-1 hard assignment is used.

**Gate A.** All five session-blocked OOF seeds must have finite, converged
transport, no collapse diagnostic violation, and a positive paired-bootstrap
95% lower bound for a preregistered UOT-FSA minus matched-kNN evidence metric.
The final recipe is chosen by nested train-only Pareto selection over
co-location, multi-group coverage, diversity, index size, and build latency.
Failure stops QHDS, CRC, and dynamic experiments.

## H2: Query-Conditioned Evidence-Set Headroom

The train-only evidence schema expresses an unordered set-of-sets relation:
`q -> {A_q1, ..., A_qG}`, where each `A_qg` is the two-member set of
alternative acceptable memories for one labeled evidence aspect and every
member is causally eligible for the query. QESA learns to cover these sets with
query-conditioned bundles without receiving evidence groups at inference. Its
inference path is `q -> latent aspect -> FSA hyperedge -> bundle -> memories`.
This is a compositional evidence hierarchy, not a temporal or reasoning trace.
The two members inside one group are alternatives, not a joint access.

**Gate B.** At identical budget and payload, the hyperedge-first QESA selector
must improve query-level per-group coverage or complete multi-group coverage
over a flat selector and the matched clique expansion. The predefined
multi-group/compositional subset comprises queries with two or more labeled evidence groups;
it is analyzed separately. If the effect is confined to the full data and not
present in that structural subset, the compositional-evidence claim is reduced
to a semantic-selection claim.

## H3: Semantic Hypergraph-Native Selection

Under the same ANN protected seeds, memory corpus, frozen reranker, payload
representation, and B in {8,16,32}, a bundle-first selector retains evidence
groups better than both (a) the clique expansion with the identical encoder and
(b) a selector that flattens semantic hyperedge memberships into single-memory
features. The gain must not come from a larger payload.

**Gate C.** Five fixed seeds must show a non-dominated quality-cost result
against the matched pairwise and flattened selectors. On a preregistered small
candidate pool, deterministic greedy must report its utility regret against an
exact branch-and-bound or ILP oracle. If matched pairwise dominates all seeds,
the hypergraph-native superiority claim stops.

## H4: Risk-Controlled Adaptive Budget

After all FSA and selector architecture choices are frozen, CRC can select the
smallest B in {8,16,32} whose frontier-risk upper bound is within the
predeclared Dense-32 tolerance. Learn-Then-Test is separately used for the
non-monotone final top-5 risk. Dense B=32 is the fallback.

**Gate D.** Calibration must meet its risk target, average selected frontier
must be below 32, fallback cannot account for the apparent efficiency result,
and final top-5 LTT must pass after multiple-policy correction. Calibration
cannot alter H1/H2 models or hyperparameters.

## H5: Content-Causal Dynamic Semantic Update

For chronological content insertion, Online UOT sufficient-statistic updates,
Bayesian prototype model selection, and atomic overlays can maintain semantic
retrieval quality near periodic full rebuild while changing only bounded local
incidence lists and consuming less update time. This is an update claim about
memory content; it does not use inferred future access chains.

**Gate E.** At 1/5/10/20% growth, the complete local update must be close to
the full-rebuild reference in preregistered quality/structure measures, faster
in p50/p95 update latency, bounded in affected entries, and pass atomic reader
consistency checks. Otherwise dynamic claims are reduced to a negative result.

## Explicitly Excluded Hypotheses

There is no v21 ordered-access-chain, dual-association complementarity, or
access-chain dynamic-update hypothesis. Those claims require independently
logged, recurrent causal `read(memory_id)` events and are governed by the
separate extension audit.

## Stop Rules

No failed gate may be repaired by changing the data split, replacing fixed
seeds, changing the membership rule after calibration, adding stage features,
or counting retrieval/reranker output as agent demand. A failure is saved as a
negative result and blocks downstream gates whose premise has failed.
