# Memory Item and Evidence Audit

This report is initialized by memory construction and completed by `build_evidence_mapping.py`.

Granularity: `turn_pair`
Memory items: 42034
Personas: 20

## Memory Kinds

- `singleton_turn`: 1
- `turn_pair`: 42033

## Evidence Mapping

Queries: 1939
Eligible evidence-metric queries: 1935
Evidence Mapping Coverage: 0.9979

### Mapping Methods

- `session_id_exact`: 8133
- `unresolved`: 349

### Mapping Granularity

The current deterministic evidence contract maps each query source key point to all medical turn-pair memory items in the referenced `(persona_id, session_id)`. This is a session-level anchor, not a span-level key-point annotation. It is intentionally conservative and must be reported as such in evidence-metric interpretations.

### History Eligibility

The schema audit and smoke attempts showed that some official MedMemoryBench queries reference source key points from the query's own `session_id`. Therefore v3 configs set `allow_current_session_context: true` and use `session_id <= query_session_id`. Future sessions remain forbidden.
