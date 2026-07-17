from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter_ns
from typing import Iterator

import torch


LATENCY_FIELDS = (
    "query_encode_ms", "ann_seed_search_ms", "latent_aspect_predict_ms", "semantic_edge_lookup_ms",
    "evidence_bundle_build_ms", "bundle_hyperedge_score_ms", "submodular_select_ms", "candidate_union_ms",
    "payload_gather_ms", "rerank_ms", "selection_total_ms", "query_e2e_ms",
)


def synchronize_if_cuda(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


class OnlineLatencyRecorder:
    """Batch-1 online timing with GPU synchronization around every timed stage."""

    def __init__(self, device: str):
        self.device = str(device)
        self.values = {field: 0.0 for field in LATENCY_FIELDS}
        self._e2e_start: int | None = None

    def start_query(self) -> None:
        if self._e2e_start is not None:
            raise RuntimeError("query timing already started")
        synchronize_if_cuda(self.device)
        self._e2e_start = perf_counter_ns()

    @contextmanager
    def stage(self, field: str) -> Iterator[None]:
        if field not in self.values or field in {"selection_total_ms", "query_e2e_ms"}:
            raise ValueError(f"invalid explicit latency stage: {field}")
        synchronize_if_cuda(self.device)
        start = perf_counter_ns()
        try:
            yield
        finally:
            synchronize_if_cuda(self.device)
            self.values[field] += (perf_counter_ns() - start) / 1_000_000.0

    def finish_query(self, selection_start_ns: int) -> dict[str, float]:
        if self._e2e_start is None:
            raise RuntimeError("query timing was not started")
        synchronize_if_cuda(self.device)
        end = perf_counter_ns()
        self.values["selection_total_ms"] = (end - selection_start_ns) / 1_000_000.0
        self.values["query_e2e_ms"] = (end - self._e2e_start) / 1_000_000.0
        self._e2e_start = None
        return dict(self.values)
