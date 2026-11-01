"""Audit observed request-level memory traces before considering a PIM route."""
from __future__ import annotations

import math


TRACE_SCHEMA = "agent-memory-observed-memory-trace-v1"


def _number(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def audit_workload_trace(trace: dict, *, minimum_critical_wait_fraction: float = 0.05) -> dict:
    """Validate observed events and qualify only critical-path memory waits.

    A pass only authorizes candidate design work. It does not establish a PIM
    speedup; that requires a calibrated target model and paired end-to-end run.
    """
    if trace.get("schema") != TRACE_SCHEMA:
        return _reject("unsupported_or_missing_trace_schema", trace.get("schema"))
    provenance = trace.get("provenance", {})
    if provenance.get("kind") != "observed_target_runtime" or not provenance.get("runtime_id"):
        raise ValueError("trace must identify an observed target runtime")
    if trace.get("clock") != "monotonic_ns":
        raise ValueError("trace clock must be monotonic_ns")
    if trace.get("synthetic_timestamps") is not False:
        raise ValueError("synthetic timestamps are not admissible")
    request_id = trace.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("trace must identify a request")
    if not 0 < minimum_critical_wait_fraction <= 1:
        raise ValueError("minimum critical wait fraction must be in (0, 1]")

    events = trace.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("nonempty observed event list required")
    by_id = {}
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in by_id:
            raise ValueError("event IDs must be unique nonempty strings")
        start = _number(event.get("start_ns"), f"{event_id}.start_ns")
        end = _number(event.get("end_ns"), f"{event_id}.end_ns")
        if end <= start:
            raise ValueError(f"{event_id} must have positive duration")
        if event.get("request_id") != trace.get("request_id"):
            raise ValueError(f"{event_id} request identity mismatch")
        by_id[event_id] = {**event, "_duration_ns": end - start}

    for event_id, event in by_id.items():
        dependencies = event.get("dependencies")
        if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
            raise ValueError(f"{event_id}.dependencies must be a unique list")
        for dependency_id in dependencies:
            if dependency_id not in by_id or dependency_id == event_id:
                raise ValueError(f"{event_id} has unknown or cyclic dependency")
            dependency = by_id[dependency_id]
            if dependency["end_ns"] > event["start_ns"]:
                raise ValueError(f"{event_id} starts before dependency {dependency_id} completes")

    ordered = sorted(by_id.values(), key=lambda event: (event["start_ns"], event["end_ns"]))
    completion, parent = {}, {}
    for event in ordered:
        event_id = event["event_id"]
        dependencies = event["dependencies"]
        if dependencies:
            predecessor = max(dependencies, key=lambda item: completion[item])
            completion[event_id] = completion[predecessor] + event["_duration_ns"]
            parent[event_id] = predecessor
        else:
            completion[event_id] = event["_duration_ns"]
            parent[event_id] = None

    terminal = max(completion, key=completion.get)
    critical_ids = []
    current = terminal
    while current is not None:
        critical_ids.append(current)
        current = parent[current]
    critical_ids.reverse()
    request_duration = max(event["end_ns"] for event in events) - min(
        event["start_ns"] for event in events)
    if request_duration <= 0:
        raise ValueError("request duration must be positive")

    critical_wait_ns = 0.0
    eligible_critical_wait_ns = 0.0
    critical_memory_events = 0
    total_read_bytes = 0
    total_write_bytes = 0
    eligible_read_bytes = 0
    for event in by_id.values():
        kind = event.get("kind")
        if kind not in {"compute", "memory", "transfer", "sync"}:
            raise ValueError(f"{event['event_id']} has unsupported event kind")
        counters = event.get("memory_counters", {})
        read_bytes = int(_number(counters.get("read_bytes", 0), "read_bytes"))
        write_bytes = int(_number(counters.get("write_bytes", 0), "write_bytes"))
        wait_ns = _number(counters.get("memory_wait_ns", 0), "memory_wait_ns")
        if wait_ns > event["_duration_ns"]:
            raise ValueError(f"{event['event_id']} memory wait exceeds event duration")
        total_read_bytes += read_bytes
        total_write_bytes += write_bytes
        if event["event_id"] in critical_ids and event.get("pim_eligible") is True:
            eligible_read_bytes += read_bytes
            eligible_critical_wait_ns += wait_ns
        if event["event_id"] in critical_ids and wait_ns > 0:
            critical_wait_ns += wait_ns
            critical_memory_events += 1

    wait_fraction = critical_wait_ns / request_duration
    eligible_wait_fraction = eligible_critical_wait_ns / request_duration
    qualified = bool(eligible_read_bytes and
                     eligible_wait_fraction >= minimum_critical_wait_fraction)
    return {
        "schema": "pim-workload-trace-audit-v1",
        "qualified_for_candidate_design": qualified,
        "status": "critical_path_memory_wait_observed" if qualified else "no_qualifying_critical_path_memory_wait",
        "request_id": trace["request_id"],
        "observed_runtime_id": provenance["runtime_id"],
        "event_count": len(events),
        "request_duration_ns": request_duration,
        "critical_path_event_ids": critical_ids,
        "critical_path_memory_event_count": critical_memory_events,
        "critical_path_memory_wait_ns": critical_wait_ns,
        "critical_path_memory_wait_fraction": wait_fraction,
        "pim_eligible_critical_memory_wait_ns": eligible_critical_wait_ns,
        "pim_eligible_critical_memory_wait_fraction": eligible_wait_fraction,
        "memory_read_bytes": total_read_bytes,
        "memory_write_bytes": total_write_bytes,
        "pim_eligible_read_bytes": eligible_read_bytes,
        "minimum_critical_wait_fraction": minimum_critical_wait_fraction,
        "target_model_calibrated": trace.get("target_model_calibrated") is True,
        "paired_e2e_speedup": None,
        "pim_speedup_claim_allowed": False,
        "claim_boundary": "observed trace qualification only; no PIM execution, calibrated prediction, or E2E speedup",
    }


def _reject(reason, source_schema=None):
    return {
        "schema": "pim-workload-trace-audit-v1",
        "qualified_for_candidate_design": False,
        "status": reason,
        "source_schema": source_schema,
        "paired_e2e_speedup": None,
        "pim_speedup_claim_allowed": False,
        "claim_boundary": "trace qualification failure; no PIM execution or speedup inference",
    }
