# Method 1 v21 Data Leakage Audit

## Fixed Partitions

The fixed split remains train = 1,355, calibration = 290, and formal test =
290. v21 development may read only the train query plan, train query
embeddings, common memory embeddings/IDs, and frozen train retrieval rows.
The guard in `src/v21/io.py` rejects paths containing formal-test markers.

## Permitted Training Signals

The projection adapter receives only train evidence-group memberships,
train-query embeddings, memory embeddings, and hard negatives mined from the
train-visible embedding space. UOT receives frozen adapter outputs and no
calibration/test labels. QESA may materialize only train rows of
`query_id -> evidence_group_id -> acceptable_memory_ids`, after checking that
every member is in the row's `eligible_memory_ids`; it may not derive or add a
temporal ordering. Selector training may use only train OOF evidence and
frozen train reranker results after Gate A.

## Prohibited Signals

The following are prohibited as inputs, targets, model-selection criteria, or
runtime feedback: calibration/test evidence labels; formal-test paths; stage
or workflow labels; future memory arrivals; future queries; future agent
access; gold evidence as runtime access; payload serialization order; and
manually weighted semantic/recency/stage scores.

## Calibration And Formal Test

Calibration remains locked until the semantic structure and selector are
frozen. It can only fit CRC/LTT policy state and an OOD fallback boundary.
Formal test remains unread and has zero runs until all v21 gates and the final
freeze manifest are complete. No v20/v20-r1 calibration or test artifact is
imported into v21.

## Version Integrity

v11-v20-r1 are immutable references. v21 writes only under
`versions/20260714_method1_fsa_dynamic_hypergraph_v21/`, with v21-prefixed
artifact and aggregate names. Reference files are SHA-256 recorded in the v21
manifest at initialization.
