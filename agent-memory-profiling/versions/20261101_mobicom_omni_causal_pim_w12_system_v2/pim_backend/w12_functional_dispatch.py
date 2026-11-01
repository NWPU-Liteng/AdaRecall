"""Decode-time dispatch into the functional W12 bank plane.

This module owns the integration boundary between the Reader and the
bit-sliced bank layout.  Single-row decode calls execute from packed bank
weights; prefill and multi-row calls use an explicit, counted Host fallback.
The fallback is intentionally visible in every summary.
"""
from __future__ import annotations

import hashlib

import torch

from .functional_decode import Int12WeightOnlyContract
from .w12_bank_layout import W12BankLayout


class W12FunctionalBankLinear(torch.nn.Module):
    """One Reader Linear backed by an immutable W12 bank layout."""

    def __init__(self, name, layout, *, mode, numeric_mode,
                 fail_on_decode_fallback):
        super().__init__()
        if mode not in {"shadow", "replace"}:
            raise ValueError("W12 dispatch mode must be shadow or replace")
        if numeric_mode not in {"fp32_bank", "bf16_reference"}:
            raise ValueError("W12 numeric mode must be fp32_bank or bf16_reference")
        if not isinstance(layout, W12BankLayout):
            raise ValueError("one W12 bank layout required")
        self.name = name
        self.layout = layout
        self.in_features = layout.columns
        self.out_features = layout.rows
        self.mode = mode
        self.numeric_mode = numeric_mode
        self.fail_on_decode_fallback = fail_on_decode_fallback
        self.active = True
        self.decode = False
        self.bank_calls = 0
        self.prefill_fallback_calls = 0
        self.decode_fallback_calls = 0
        self.fallback_rows = 0
        self.bank_external_bytes = 0
        self.fallback_external_bytes = 0
        self.shadow_records = []
        self.max_absolute_error = 0.0
        self.max_relative_l2_error = 0.0
        self.argmax_matches = 0
        self.argmax_observations = 0

    @property
    def resident_bytes(self):
        return self.layout.resident_bytes

    @property
    def weight_bytes(self):
        return self.layout.weight_bytes

    @property
    def element_bytes(self):
        return 1.5

    def _is_single_decode_row(self, activation):
        return (self.decode and isinstance(activation, torch.Tensor)
                and activation.shape[-1] == self.in_features
                and activation.numel() == self.in_features)

    def _host_fallback(self, activation):
        rows = int(activation.numel() // self.in_features)
        output = self.layout.linear(activation)
        self.fallback_external_bytes += (
            self.layout.weight_bytes + self.layout.scale_bytes
            + self.layout.bias_bytes
            + rows * self.in_features * 2 + rows * self.out_features * 4)
        self.fallback_rows += rows
        return output

    def forward(self, activation):
        if not self.active:
            raise RuntimeError("inactive W12 bank module was called")
        if not self._is_single_decode_row(activation):
            if self.decode:
                self.decode_fallback_calls += 1
                if self.fail_on_decode_fallback:
                    raise RuntimeError(
                        "decode call did not match the single-row bank route: "
                        + self.name)
            else:
                self.prefill_fallback_calls += 1
            return self._host_fallback(activation)

        flat = activation.reshape(self.in_features)
        bank_output = self.layout.gemv(flat, numeric_mode=self.numeric_mode)
        self.bank_calls += 1
        self.bank_external_bytes += self.layout.external_transfer_bytes_per_token
        if self.mode == "shadow":
            host_output = self._host_fallback(activation)
            bank_float = bank_output.float()
            host_float = host_output.reshape(-1).float()
            error = (bank_float - host_float).abs()
            relative = ((bank_float - host_float).norm()
                        / host_float.norm().clamp_min(1e-30))
            maximum = float(error.max().item())
            relative_value = float(relative.item())
            self.max_absolute_error = max(self.max_absolute_error, maximum)
            self.max_relative_l2_error = max(self.max_relative_l2_error,
                                             relative_value)
            self.argmax_observations += 1
            self.argmax_matches += int(
                bank_float.argmax().item() == host_float.argmax().item())
            if len(self.shadow_records) < 16:
                self.shadow_records.append({
                    "name": self.name,
                    "max_absolute_error": maximum,
                    "relative_l2_error": relative_value,
                    "argmax_equal": bool(
                        bank_float.argmax().item() == host_float.argmax().item()),
                })
            return host_output
        return bank_output.reshape(*activation.shape[:-1], self.out_features)

    def summary(self):
        return {
            "name": self.name,
            "shape": [self.out_features, self.in_features],
            "mode": self.mode,
            "numeric_mode": self.numeric_mode,
            "bank_calls": self.bank_calls,
            "prefill_fallback_calls": self.prefill_fallback_calls,
            "decode_fallback_calls": self.decode_fallback_calls,
            "fallback_rows": self.fallback_rows,
            "bank_external_bytes": self.bank_external_bytes,
            "fallback_external_bytes": self.fallback_external_bytes,
            "resident_bytes": self.resident_bytes,
            "max_absolute_error_vs_host_contract": self.max_absolute_error,
            "max_relative_l2_error_vs_host_contract": self.max_relative_l2_error,
            "argmax_observations": self.argmax_observations,
            "argmax_matches": self.argmax_matches,
            "shadow_records": self.shadow_records,
        }


class W12FunctionalDispatch:
    """Replace Reader language Linears with packed W12 bank modules."""

    def __init__(self, reader, *, mode="replace", bank_count=256,
                 tile_rows=16, group_size=128, numeric_mode="fp32_bank",
                 release_source=False, fail_on_decode_fallback=True):
        if mode not in {"shadow", "replace"}:
            raise ValueError("W12 dispatch mode must be shadow or replace")
        if (not isinstance(bank_count, int) or bank_count <= 0
                or not isinstance(tile_rows, int) or tile_rows <= 0):
            raise ValueError("positive bank count and row tile required")
        if group_size not in (8, 16, 32, 64, 128, 256):
            raise ValueError("supported 12-bit group size required")
        if numeric_mode not in {"fp32_bank", "bf16_reference"}:
            raise ValueError("W12 numeric mode must be fp32_bank or bf16_reference")
        self.reader = reader
        self.mode = mode
        self.bank_count = bank_count
        self.tile_rows = tile_rows
        self.group_size = group_size
        self.numeric_mode = numeric_mode
        self.release_source = release_source
        self.fail_on_decode_fallback = fail_on_decode_fallback
        self.modules = {}
        self.originals = []
        self.hook = None
        self.decode = False
        self.entered = False
        self.source_released = False
        self.alias_modules = 0
        self.released_source_bytes = 0
        self.conversion_seconds = 0.0
        self._layout_instance_sha256 = None

    def _targets(self):
        for name, module in self.reader.named_modules():
            if (isinstance(module, (torch.nn.Linear, Int12WeightOnlyContract))
                    and (name.startswith("model.language_model.")
                         or name == "lm_head")):
                yield name, module

    def _replace(self, name, module, layout):
        parent_name, separator, attribute = name.rpartition(".")
        parent = self.reader.get_submodule(parent_name) if separator else self.reader
        wrapper = W12FunctionalBankLinear(
            name, layout, mode=self.mode, numeric_mode=self.numeric_mode,
            fail_on_decode_fallback=self.fail_on_decode_fallback)
        setattr(parent, attribute, wrapper)
        self.modules[name] = wrapper
        if not self.release_source:
            self.originals.append((parent, attribute, module))

    def __enter__(self):
        if self.entered:
            raise ValueError("W12 dispatcher cannot be entered twice")
        if self.reader.training:
            raise ValueError("immutable eval Reader required")
        import time
        started = time.perf_counter()
        layouts = {}
        identities = {}
        released_identities = set()
        try:
            names = [name for name, _ in self._targets()]
            if not names:
                raise ValueError("no Reader language Linear modules found")
            for name in names:
                module = self.reader.get_submodule(name)
                weight = module.weight
                identity = id(weight)
                if identity in layouts:
                    self.alias_modules += 1
                    layout = layouts[identity]
                else:
                    scales = (module.group_scales
                              if isinstance(module, Int12WeightOnlyContract)
                              and hasattr(module, "group_scales") else None)
                    layout = W12BankLayout(
                        weight, group_size=self.group_size,
                        bank_count=self.bank_count, tile_rows=self.tile_rows,
                        bias=module.bias, group_scales=scales)
                    layouts[identity] = layout
                identities[name] = identity
                self._replace(name, module, layout)
                if identity not in released_identities:
                    source = (module.weight if hasattr(module, "weight")
                              and module.weight is not None else None)
                    if source is not None:
                        self.released_source_bytes += (
                            source.numel() * source.element_size())
                    released_identities.add(identity)
                del module
            if not self.modules:
                raise ValueError("no W12 bank modules were installed")
            self.hook = self.reader.register_forward_pre_hook(
                self._set_phase, with_kwargs=True)
            self.decode = False
            self.entered = True
            self.conversion_seconds = time.perf_counter() - started
            self.source_released = bool(self.release_source)
            self._layout_instance_sha256 = self._compute_layout_instance_sha256()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _set_phase(self, module, inputs, kwargs):
        self.decode = kwargs.get("past_key_values") is not None
        for wrapper in self.modules.values():
            wrapper.decode = self.decode

    def _compute_layout_instance_sha256(self):
        digest = hashlib.sha256()
        unique_layouts = {}
        for name in sorted(self.modules):
            wrapper = self.modules[name]
            layout = wrapper.layout
            unique_layouts[id(layout)] = layout
            values = (
                name,
                str(id(layout)),
                str(id(layout.storage)),
                str(id(layout.group_scales)),
                str(id(layout.bias)),
                str(layout.rows),
                str(layout.columns),
                str(layout.bank_count),
                str(layout.tile_rows),
                str(layout.group_size),
            )
            digest.update(("\0".join(values) + "\n").encode("ascii"))
        digest.update(("unique_layouts=" + str(len(unique_layouts))).encode("ascii"))
        return digest.hexdigest()

    def counter_snapshot(self):
        """Return cumulative counters with a process-local layout identity."""
        if self._layout_instance_sha256 is None:
            raise RuntimeError("W12 dispatcher must be entered before taking counters")
        wrappers = list(self.modules.values())
        unique_layouts = {id(item.layout) for item in wrappers}
        return {
            "selected_modules": len(wrappers),
            "unique_layouts": len(unique_layouts),
            "bank_calls": sum(item.bank_calls for item in wrappers),
            "decode_fallback_calls": sum(item.decode_fallback_calls
                                         for item in wrappers),
            "prefill_fallback_calls": sum(item.prefill_fallback_calls
                                          for item in wrappers),
            "fallback_rows": sum(item.fallback_rows for item in wrappers),
            "bank_external_bytes": sum(item.bank_external_bytes
                                       for item in wrappers),
            "fallback_external_bytes": sum(item.fallback_external_bytes
                                           for item in wrappers),
            "layout_instance_sha256": self._layout_instance_sha256,
        }

    @staticmethod
    def counter_delta(before, after):
        """Subtract snapshots and fail if the dispatcher was rematerialized."""
        if before["layout_instance_sha256"] != after["layout_instance_sha256"]:
            raise RuntimeError("W12 bank layout identity changed across a stage")
        fields = (
            "bank_calls",
            "decode_fallback_calls",
            "prefill_fallback_calls",
            "fallback_rows",
            "bank_external_bytes",
            "fallback_external_bytes",
        )
        result = {name: after[name] - before[name] for name in fields}
        result.update({
            "selected_modules": after["selected_modules"],
            "unique_layouts": after["unique_layouts"],
            "layout_instance_sha256": after["layout_instance_sha256"],
            "layout_identity_unchanged": True,
        })
        return result

    def materialization_summary(self):
        unique_layouts = {}
        for wrapper in self.modules.values():
            unique_layouts[id(wrapper.layout)] = wrapper.layout
        return {
            "backend": "w12_bit_sliced_bank_functional",
            "modules": len(self.modules),
            "unique_layouts": len(unique_layouts),
            "aliased_modules": self.alias_modules,
            "group_size": self.group_size,
            "weight_bits": 12,
            "bank_count": self.bank_count,
            "tile_rows": self.tile_rows,
            "contract_row_chunk": self.bank_count * self.tile_rows,
            "resident_bytes": sum(layout.resident_bytes
                                  for layout in unique_layouts.values()),
            "logical_bytes": sum(layout.logical_bytes
                                 for layout in unique_layouts.values()),
            "released_source_parameter_bytes": self.released_source_bytes,
            "source_release": self.source_released,
            "conversion_seconds": self.conversion_seconds,
            "functional_materialization": (
                "one packed 12-plane weight copy plus bank-local FP32 group "
                "scales; no dequantized BF16 language-weight copy; Host fallback "
                "and control share the bank-tile-aligned BF16 row contract"),
            "boundary": (
                "functional bank reference; Torch executes the layout, not "
                "silicon PIM or a cycle-accurate device"),
        }

    def summary(self):
        wrappers = list(self.modules.values())
        unique_layouts = {id(item.layout): item.layout for item in wrappers}
        bank_calls = sum(item.bank_calls for item in wrappers)
        decode_fallback = sum(item.decode_fallback_calls for item in wrappers)
        prefill_fallback = sum(item.prefill_fallback_calls for item in wrappers)
        unobserved = sorted(name for name, item in self.modules.items()
                            if item.bank_calls == 0)
        relative_errors = [item.max_relative_l2_error for item in wrappers
                           if item.argmax_observations]
        absolute_errors = [item.max_absolute_error for item in wrappers
                           if item.argmax_observations]
        return {
            "backend": "w12_bit_sliced_bank_functional",
            "mode": self.mode,
            "numeric_mode": self.numeric_mode,
            "selected_modules": len(wrappers),
            "bank_calls": bank_calls,
            "decode_fallback_calls": decode_fallback,
            "prefill_fallback_calls": prefill_fallback,
            "fallback_rows": sum(item.fallback_rows for item in wrappers),
            "bank_external_bytes": sum(item.bank_external_bytes for item in wrappers),
            "fallback_external_bytes": sum(item.fallback_external_bytes
                                           for item in wrappers),
            "resident_bytes": sum(layout.resident_bytes
                                  for layout in unique_layouts.values()),
            "released_source_parameter_bytes": self.released_source_bytes,
            "source_released": self.source_released,
            "unobserved_selected_modules": unobserved,
            "all_selected_modules_executed": not unobserved,
            "silent_fallback": False,
            "shadow": {
                "observations": sum(item.argmax_observations for item in wrappers),
                "argmax_matches": sum(item.argmax_matches for item in wrappers),
                "max_absolute_error": (max(absolute_errors)
                                       if absolute_errors else None),
                "max_relative_l2_error": (max(relative_errors)
                                          if relative_errors else None),
            },
            "calls_per_module": {
                name: {
                    "bank_calls": item.bank_calls,
                    "decode_fallback_calls": item.decode_fallback_calls,
                    "prefill_fallback_calls": item.prefill_fallback_calls,
                }
                for name, item in self.modules.items()
            },
            "physical_pim_executed": False,
            "performance_comparable": False,
            "boundary": (
                "packed bank layout and numerical dispatch evidence only; "
                "Torch wall-clock is not a PIM or end-to-end latency result"),
        }

    def __exit__(self, *unused):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        for wrapper in self.modules.values():
            wrapper.active = False
        if self.originals:
            for parent, attribute, original in reversed(self.originals):
                setattr(parent, attribute, original)
            self.originals.clear()
        self.entered = False
