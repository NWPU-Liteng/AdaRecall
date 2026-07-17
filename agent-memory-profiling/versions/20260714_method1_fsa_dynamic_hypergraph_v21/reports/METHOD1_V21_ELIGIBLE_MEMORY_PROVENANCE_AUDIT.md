# Method 1 v21 Eligible Memory Provenance Audit

## Result

The train-only audit passed for all 1,355 queries. Recomputing eligibility from memory metadata exactly reproduced every stored `eligible_memory_ids` list. No eligible memory crossed a persona boundary or came from a future session, and the train evidence incidence contained exactly the train query IDs. Calibration and formal test were not read.

| Check | Result |
| --- | ---: |
| train queries audited | 1355 |
| memory items | 42034 |
| eligible incidences | 1138785 |
| exact-list mismatches | 0 |
| cross-persona incidences | 0 |
| future-session incidences | 0 |
| current-session incidences | 27523 |
| calibration accesses | 0 |
| formal-test accesses | 0 |

## Generation Rule

The deployable membership rule uses only memory persona/session/type/order metadata and query persona/session metadata. The resolved v6 configuration includes noise and permits `memory.session_id <= query.query_session_id`; it never reads answers, gold memory IDs, evidence groups, or acceptable members. The same function therefore runs when no labels exist.

The source benchmark evaluates a query after the accumulated current session, so same-session memory is treated as already written current context. The source contains session IDs but no independent query-event timestamp; causality is therefore confirmed at the benchmark session contract, not at sub-session event-time resolution. Future sessions remain forbidden.

## Gold And Split Boundary

The original split builder used evidence to retain scoreable queries and to create gold fields. This is a query-pool selection dependency, not an eligible-membership dependency: for any included query, the eligible set is reproduced without loading labels. Later key-point mapping intersects labels with the pre-existing eligible set, and the session-blocked split moves complete rows without regenerating eligibility. The v21 train incidence is checked only against the train plan; calibration/test eligible sets do not enter its statistics.

## Reproducibility

The machine-readable audit records SHA-256 and byte size for the generator/helper scripts, inherited configurations, memory corpus, train plan, and v21 train incidence.
