# Method 1 v21 Query-Conditioned Evidence-Set Association Specification

## What The Existing Data Actually Observes

For each query q, the fixed plan provides a causally eligible memory set
`M_q^past` and zero or more labeled evidence groups
`G(q) = {A_q1, ..., A_qR}`. In the train partition there are 1,355 queries,
1,819 groups, and every group has two alternative acceptable memory IDs. All
3,638 group-member incidences are contained in their query's
`eligible_memory_ids`. Query group counts are 1 for 1,009 queries, 2 for 238,
3 for 98, and 4 for 10. Thus 346 queries form the predeclared
multi-group/compositional subset.

An `A_qr` set means that either contained item is accepted support for one
labeled evidence aspect. It is not an event log and does not mean both items
were read, that they were read in a particular order, or that they reflect the
model's private chain of thought. Same-session eligible items occur in the
data, so the explicit `eligible_memory_ids` field, not a coarse session number,
is the causality authority.

## Association And Inference Path

During training only, QESA supervises a query-conditioned set-coverage model.
The model has no persistent test-time access to `G(q)`. Given a query embedding
and protected ANN seeds, it learns the hierarchical association path

`q -> z_l(q) -> e_k(FSA) -> b_k(q) -> union memory IDs`.

Here `z_l(q)` is a learned latent evidence aspect, `e_k` is an activated fuzzy
semantic hyperedge, and `b_k(q)` is an adaptive local bundle selected from that
edge. This is deliberately called a **hierarchical evidence path**, rather
than a chain. It captures query-conditioned coordination across a set of
evidence aspects without inventing a time direction absent from the data. Any
later "higher-order" paper claim remains conditional on matched factor-graph
results.

For a candidate union C, the observed train utility is

`U_evidence(C, q) = (1 / |G(q)|) sum_{A in G(q)} 1[C intersects A]`.

This records coverage of each labeled aspect and handles the two members of an
acceptable set as alternatives. The primary multi-group/compositional analysis uses
`|G(q)| >= 2`; complete multi-group coverage is separately reported as
`prod_{A in G(q)} 1[C intersects A]` but is not silently substituted for final
answer quality.

## Selector Objective

The learned bundle utility has the structurally monotone submodular form

`F(B | q) = sum_l alpha_l(q) phi_l(sum_{b in B} w_lb(q)) + sum_{b in B} u_b(q)`,

where `alpha_l(q) >= 0`, `w_lb(q) >= 0`, and each `phi_l` is nondecreasing and
concave. The selected bundles obey protected seeds, frontier, and payload
constraints through primal-dual optimization. Evidence coverage is the primary
training utility; frozen-reranker promotability may only break near-ties in
evidence utility. No semantic/co-access/recency linear score is introduced.

## Matched Attribution

The pairwise baseline uses clique expansion of the exact same train-only
query-evidence incidence; it is not weakened by a different corpus or smaller
candidate pool. The four required comparisons are: modular memory ranking,
pairwise clique expansion, flattened incidence features, and the
hyperedge-first QESA bundle selector. A compositional-evidence benefit requires
improvement on the multi-group subset without adding payload. An independent
hypergraph contribution additionally requires the matched pairwise factor-graph
comparison.

## Explicit Non-Claims

QESA is not LACA/OACH, a Hawkes model, an order predictor, a co-access model,
or a chain-of-thought reconstruction method. A future observed memory-tool
trace can add an independent LACA branch only after passing the separate data
eligibility audit.
