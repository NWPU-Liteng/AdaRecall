"""Functional bank-local emulation for the Reader output projection only."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .functional_decode import BankOwnedLinear


class LmHeadBankDispatcher:
    def __init__(self, reader, *, mode, banks=32, input_tile=256):
        if mode not in {"shadow", "replace"}:
            raise ValueError("lm_head functional mode must be shadow or replace")
        module = reader.get_submodule("lm_head")
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("Reader lm_head must be a torch.nn.Linear")
        if module.weight.dtype != torch.bfloat16:
            raise ValueError("lm_head functional route requires BF16 weights")
        self.reader = reader
        self.module = module
        self.mode = mode
        self.banks = banks
        self.input_tile = input_tile
        self.original = None
        self.layout = None
        self.active = False
        self.calls = 0
        self.replaced_calls = 0
        self.skipped_non_single_row_calls = 0
        self.records = []

    def __enter__(self):
        if self.original is not None:
            raise ValueError("lm_head dispatcher cannot be entered twice")
        self.original = self.module.forward
        self.module.forward = self._wrapper
        return self

    def _wrapper(self, activation):
        if not self.active:
            return self.original(activation)
        if activation.numel() != self.module.in_features:
            self.skipped_non_single_row_calls += 1
            return self.original(activation)
        if self.layout is None:
            self.layout = BankOwnedLinear(self.module.weight, banks=self.banks,
                                          input_tile=self.input_tile)
        bank_output = self.layout(activation, self.module.bias)
        bank_output = bank_output.to(activation.dtype).reshape(
            *activation.shape[:-1], self.module.out_features)
        self.calls += 1
        if self.mode == "shadow":
            host_output = self.original(activation)
            host_flat = host_output.reshape(-1)
            bank_flat = bank_output.reshape(-1)
            difference = (bank_flat.float() - host_flat.float()).abs()
            self.records.append({
                "call": self.calls,
                "activation_shape": list(activation.shape),
                "output_elements": int(bank_flat.numel()),
                "max_absolute_error": float(difference.max().item()),
                "bitwise_equal": bool(torch.equal(bank_output, host_output)),
                "argmax_equal": bool(torch.equal(bank_flat.argmax(-1), host_flat.argmax(-1))),
            })
            return host_output
        self.replaced_calls += 1
        return bank_output

    @contextmanager
    def decode(self):
        if self.original is None:
            raise ValueError("dispatcher must be entered before decode scope")
        if self.active:
            raise ValueError("nested lm_head decode scopes are not supported")
        self.active = True
        try:
            yield self
        finally:
            self.active = False

    def summary(self):
        return {
            "mode": self.mode,
            "calls": self.calls,
            "replaced_calls": self.replaced_calls,
            "skipped_non_single_row_calls": self.skipped_non_single_row_calls,
            "resident_bytes": (self.layout.resident_bytes if self.layout is not None else None),
            "records": self.records,
            "physical_pim_executed": False,
            "performance_comparable": False,
            "boundary": "GPU bank-owned functional emulation; not physical PIM latency",
        }

    def __exit__(self, *unused):
        if self.original is not None:
            self.module.forward = self.original
            self.original = None
        self.active = False
