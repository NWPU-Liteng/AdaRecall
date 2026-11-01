"""Shape-only profiling of actual Reader linear calls; no weights or activations stored."""
from __future__ import annotations

from collections import Counter
import math


class ReaderOperatorTrace:
    def __init__(self, model):
        import torch

        self.records = {}
        self.phase = 'prefill'
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def capture(module, inputs, output):
            if not inputs:
                return
            rows = math.prod(inputs[0].shape[:-1])
            key = (name, self.phase, rows)
            if key not in self.records:
                self.records[key] = {
                    'name': name, 'phase': self.phase, 'rows': rows,
                    'in_features': module.in_features, 'out_features': module.out_features,
                    'element_bytes': module.weight.element_size(),
                    'weight_bytes': module.weight.numel() * module.weight.element_size(), 'calls': 0,
                }
            self.records[key]['calls'] += 1
        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def as_dict(self):
        records = list(self.records.values())
        counts = Counter()
        for record in records:
            counts[record['phase']] += record['calls']
        decode_gemv_records = [record for record in records
                               if record['phase'] == 'decode' and record['rows'] == 1]
        unique_decode_weights = {record['name']: record['weight_bytes']
                                 for record in decode_gemv_records}
        return {'operators': records, 'calls_by_phase': dict(counts),
                'decode_single_row_linear_calls': sum(record['calls']
                                                       for record in decode_gemv_records),
                'decode_single_row_unique_weight_bytes': sum(unique_decode_weights.values()),
                'decode_single_row_macs': sum(
                    record['calls'] * record['in_features'] * record['out_features']
                    for record in decode_gemv_records),
                'coverage': 'executed nn.Linear calls only; attention, recurrent kernels, normalization and transfers excluded',
                'measurement': 'actual shapes and invocation counts, not per-kernel wall-clock'}
