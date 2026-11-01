"""Audit naturally co-ready decode requests from observed serving queues."""
from __future__ import annotations

from collections import defaultdict
from statistics import median


TRACE_SCHEMA = "pim-online-ready-queue-trace-v1"


def audit_microbatch_opportunity(trace: dict) -> dict:
    if trace.get("schema") != TRACE_SCHEMA:
        return {
            "schema": "pim-microbatch-opportunity-audit-v1",
            "qualified": False,
            "status": "missing_observed_online_ready_queue_trace",
            "source_schema": trace.get("schema"),
            "has_natural_microbatch_opportunity": None,
            "paired_smoke_allowed": False,
            "claim_boundary": "request order and service durations do not establish online concurrency",
        }

    provenance = trace.get("provenance", {})
    if provenance.get("kind") != "observed_online_runtime":
        raise ValueError("trace provenance must identify an observed online runtime")
    if trace.get("clock") != "monotonic_ns":
        raise ValueError("trace clock must be monotonic_ns")
    snapshots = trace.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("at least one observed ready-queue snapshot is required")

    snapshot_times = []
    max_ready_counts = []
    max_compatible_decode_counts = []
    compatible_pair_snapshot_count = 0
    decode_ready_events = 0
    observed_wait_ns = []
    unique_tokens = set()

    for snapshot in snapshots:
        snapshot_ns = snapshot.get("snapshot_ns")
        ready = snapshot.get("ready")
        if not isinstance(snapshot_ns, int) or snapshot_ns < 0:
            raise ValueError("snapshot_ns must be a non-negative integer")
        if not isinstance(ready, list):
            raise ValueError("snapshot ready queue must be a list")
        if snapshot_times and snapshot_ns < snapshot_times[-1]:
            raise ValueError("queue snapshots must be ordered by monotonic time")
        snapshot_times.append(snapshot_ns)

        seen_requests = set()
        compatible_decode = defaultdict(int)
        for event in ready:
            request_id = event.get("request_id")
            token_index = event.get("token_index")
            ready_since_ns = event.get("ready_since_ns")
            phase = event.get("phase")
            compatibility_key = event.get("compatibility_key")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("ready events require a request_id")
            if request_id in seen_requests:
                raise ValueError("a request may have only one ready token in a snapshot")
            seen_requests.add(request_id)
            if not isinstance(token_index, int) or token_index < 0:
                raise ValueError("token_index must be a non-negative integer")
            if not isinstance(ready_since_ns, int) or ready_since_ns > snapshot_ns:
                raise ValueError("ready_since_ns must not be later than snapshot_ns")
            if not isinstance(phase, str) or not phase:
                raise ValueError("ready events require a phase")
            if not isinstance(compatibility_key, str) or not compatibility_key:
                raise ValueError("ready events require an exact compatibility_key")

            if phase == "decode":
                decode_ready_events += 1
                observed_wait_ns.append(snapshot_ns - ready_since_ns)
                unique_tokens.add((request_id, token_index))
                compatible_decode[compatibility_key] += 1

        max_ready_counts.append(len(ready))
        max_compatible = max(compatible_decode.values(), default=0)
        max_compatible_decode_counts.append(max_compatible)
        if max_compatible >= 2:
            compatible_pair_snapshot_count += 1

    has_opportunity = compatible_pair_snapshot_count > 0
    return {
        "schema": "pim-microbatch-opportunity-audit-v1",
        "qualified": True,
        "status": "natural_compatible_decode_concurrency_observed" if has_opportunity else "no_compatible_decode_concurrency_observed",
        "source_schema": TRACE_SCHEMA,
        "provenance": provenance,
        "observed_snapshots": len(snapshots),
        "unique_decode_request_token_pairs": len(unique_tokens),
        "decode_ready_event_observations": decode_ready_events,
        "snapshots_with_multiple_compatible_decode_requests": compatible_pair_snapshot_count,
        "fraction_snapshots_with_multiple_compatible_decode_requests": compatible_pair_snapshot_count / len(snapshots),
        "max_ready_requests_in_snapshot": max(max_ready_counts, default=0),
        "max_compatible_decode_batch_opportunity": max(max_compatible_decode_counts, default=0),
        "observed_decode_queue_wait_ms": {
            "p50": median(observed_wait_ns) / 1_000_000 if observed_wait_ns else None,
            "max": max(observed_wait_ns) / 1_000_000 if observed_wait_ns else None,
        },
        "has_natural_microbatch_opportunity": has_opportunity,
        "paired_smoke_allowed": has_opportunity,
        "pim_speedup": None,
        "end_to_end_speedup": None,
        "claim_boundary": "observed queue co-readiness only; no PIM execution or latency/speedup inference",
    }
