from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    stages_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter_ns()
        yield
        self.stages_ms[name] = self.stages_ms.get(name, 0.0) + (time.perf_counter_ns() - started) / 1e6
