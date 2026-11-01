"""Synchronous bank-owned GEMV emulation; no hardware latency interpretation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math

import torch

_pending_finite_checks = ContextVar('int8_pending_finite_checks', default=None)


def require_finite(value, message):
    condition = torch.isfinite(value).all()
    pending = _pending_finite_checks.get()
    if pending is None:
        if not condition:
            raise ValueError(message)
    else:
        pending.append(condition)


@contextmanager
def deferred_int8_finite_checks():
    if _pending_finite_checks.get() is not None:
        raise ValueError('finite-check scopes must not nest')
    predicates = []
    token = _pending_finite_checks.set(predicates)
    stats = {'checks': 0, 'host_confirmations': 0, 'predicate_payload_bytes': 0}
    try:
        yield stats
        stats['checks'] = len(predicates)
        stats['predicate_payload_bytes'] = sum(value.numel() * value.element_size() for value in predicates)
        if predicates:
            if len({value.device for value in predicates}) != 1:
                raise ValueError('finite-check aggregation requires one device')
            stats['host_confirmations'] = 1
            if not torch.stack(predicates).all():
                raise ValueError('nonfinite INT8 activation or scaled output before Reader forward completion')
    finally:
        _pending_finite_checks.reset(token)


class DeferredInt8Validation:
    def __init__(self, reader):
        self.reader = reader
        self.previous = None
        self.original = None
        self.calls = 0
        self.checks = 0
        self.host_confirmations = 0
        self.maximum_predicate_payload_bytes = 0

    def __enter__(self):
        if self.reader.training or self.original is not None:
            raise ValueError('immutable eval Reader and one validation dispatch required')
        self.previous = self.reader.__dict__.get('forward')
        self.original = self.reader.forward
        original = self.original

        def forward(*args, **kwargs):
            if kwargs.get('past_key_values') is None:
                return original(*args, **kwargs)
            with deferred_int8_finite_checks() as stats:
                output = original(*args, **kwargs)
            self.calls += 1
            self.checks += stats['checks']
            self.host_confirmations += stats['host_confirmations']
            self.maximum_predicate_payload_bytes = max(
                self.maximum_predicate_payload_bytes, stats['predicate_payload_bytes'])
            return output

        self.reader.forward = forward
        return self

    def summary(self):
        return {'decode_forward_calls': self.calls, 'finite_predicates_checked': self.checks,
                'host_confirmations': self.host_confirmations,
                'maximum_predicate_payload_bytes': self.maximum_predicate_payload_bytes,
                'arithmetic_changed': False, 'checks_removed': False,
                'boundary': 'all existing predicates evaluated; one Host confirmation before each cached Reader forward returns; prefill remains synchronous; no skipped validation or repeated producer arithmetic'}

    def __exit__(self, *unused):
        if self.original is not None:
            if self.previous is None:
                del self.reader.forward
            else:
                self.reader.forward = self.previous
            self.original = None


def quantize_int8_rows(value):
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError('finite floating-point rows required')
    require_finite(value, 'finite floating-point rows required')
    floating = value.float()
    scales = floating.abs().amax(dim=1, keepdim=True).div(127).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = floating.div(scales).round().clamp(-127, 127).to(torch.int8)
    return quantized, scales


def quantize_int8_groups(value, group_size=128):
    if (value.ndim != 2 or not value.is_floating_point()
            or not isinstance(group_size, int) or group_size < 1
            or value.shape[1] % group_size):
        raise ValueError('finite floating matrix and aligned positive group size required')
    require_finite(value, 'finite floating matrix required')
    floating = value.float()
    groups = value.shape[1] // group_size
    grouped = floating.reshape(value.shape[0], groups, group_size)
    scales = grouped.abs().amax(dim=2, keepdim=True).div(127).clamp_min(
        torch.finfo(torch.float32).tiny)
    grouped_quantized = grouped.div(scales).round().clamp(-127, 127).to(torch.int8)
    dequantized = (grouped_quantized.float() * scales).reshape(value.shape)
    return grouped_quantized.reshape(value.shape), scales.squeeze(2), dequantized


def quantize_signed_groups(value, group_size=128, weight_bits=12):
    """Symmetric group-wise signed quantization for bank-resident bit planes.

    The code is stored in ``int16`` for functional replay; the logical storage
    contract charges only ``weight_bits`` per element.  No residual plane or
    residual scale is introduced, so the bank format has one scale plane and
    one bit-sliced weight plane.
    """
    if weight_bits not in (8, 10, 12):
        raise ValueError('weight bits must be eight, ten or twelve')
    if (value.ndim != 2 or not value.is_floating_point()
            or not isinstance(group_size, int) or group_size < 1
            or value.shape[1] % group_size):
        raise ValueError('finite floating matrix and aligned positive group size required')
    require_finite(value, 'finite floating matrix required')
    levels = 2 ** (weight_bits - 1) - 1
    groups = value.shape[1] // group_size
    dtype = torch.int8 if weight_bits == 8 else torch.int16
    quantized = torch.empty_like(value, dtype=dtype)
    scales = torch.empty((value.shape[0], groups), dtype=torch.float32,
                         device=value.device)
    dequantized = torch.empty_like(value)
    for begin in range(0, value.shape[0], 512):
        end = min(begin + 512, value.shape[0])
        floating = value[begin:end].float().reshape(end - begin, groups, group_size)
        row_scales = floating.abs().amax(dim=2, keepdim=True).div(levels).clamp_min(
            torch.finfo(torch.float32).tiny)
        row_quantized = floating.div(row_scales).round().clamp(-levels, levels)
        quantized[begin:end] = row_quantized.reshape(end - begin, -1).to(dtype)
        scales[begin:end] = row_scales.squeeze(2)
        dequantized[begin:end] = (
            row_quantized * row_scales).reshape(end - begin, -1).to(value.dtype)
    return quantized, scales, dequantized


def quantize_int8_residual_groups(value, group_size=128, residual_bits=0):
    """Refine group-wise INT8 weights with a packed low-bit residual.

    The base weight remains one INT8 value per element.  A second, much
    smaller signed code stores the rounding residual in the same group.  The
    returned BF16 matrix is only a functional replay artifact; the target PIM
    layout keeps the base code, base scales, residual code and residual scales
    bank-resident.
    """
    if residual_bits not in (0, 2, 4):
        raise ValueError('residual bits must be zero, two or four')
    quantized, scales, dequantized = quantize_int8_groups(value, group_size)
    if residual_bits == 0:
        return quantized, scales, dequantized, None, None
    groups = value.shape[1] // group_size
    levels = 2 ** (residual_bits - 1) - 1
    residual_quantized = torch.empty_like(quantized)
    residual_scales = torch.empty((value.shape[0], groups), dtype=torch.float32,
                                  device=value.device)
    refined = torch.empty_like(value)
    for begin in range(0, value.shape[0], 512):
        end = min(begin + 512, value.shape[0])
        floating = value[begin:end].float().reshape(end - begin, groups, group_size)
        base = dequantized[begin:end].reshape(end - begin, groups, group_size)
        residual = floating - base
        row_scales = residual.abs().amax(dim=2, keepdim=True).div(levels).clamp_min(
            torch.finfo(torch.float32).tiny)
        row_quantized = residual.div(row_scales).round().clamp(-levels, levels)
        residual_quantized[begin:end] = row_quantized.reshape(end - begin, -1).to(
            torch.int8)
        residual_scales[begin:end] = row_scales.squeeze(2)
        refined[begin:end] = (base + row_quantized * row_scales).reshape(
            end - begin, -1).to(value.dtype)
    return (quantized, scales, refined, residual_quantized, residual_scales)


class Int12WeightOnlyContract(torch.nn.Module):
    """Uniform 12-bit group-quantized weight reference with BF16 activations.

    This is the functional replay contract for one bit-sliced bank plane.  The
    logical footprint is bit-packed; the Torch module materializes BF16 weights
    only so the quality path can run on the existing Reader.
    """

    def __init__(self, weight, bias=None, *, group_size=128, row_chunk=None):
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] % group_size:
            raise ValueError('aligned 12-bit weight matrix required')
        if row_chunk is not None and (isinstance(row_chunk, bool)
                                      or not isinstance(row_chunk, int)
                                      or row_chunk <= 0):
            raise ValueError('positive integer contract row chunk required')
        if bias is not None and (bias.shape != (weight.shape[0],)
                                 or not torch.isfinite(bias).all()):
            raise ValueError('finite output bias required')
        quantized, scales, dequantized = quantize_signed_groups(
            weight.detach(), group_size, weight_bits=12)
        self.register_buffer('weight', dequantized.to(weight.dtype))
        self.register_buffer('group_scales', scales)
        self.register_buffer('bias', bias.detach().clone() if bias is not None else None)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.group_size = group_size
        self.row_chunk = row_chunk
        self.weight_bits = 12
        self.logical_weight_bytes = (
            self.out_features * math.ceil(self.in_features * 12 / 8))
        self.scale_storage_bytes = scales.numel() * scales.element_size()
        del quantized

    @property
    def logical_storage_bytes(self):
        bias_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        return self.logical_weight_bytes + self.scale_storage_bytes + bias_bytes

    @property
    def functional_dequantized_weight_bytes(self):
        return self.weight.numel() * self.weight.element_size()

    def forward(self, activation):
        if (not activation.is_floating_point() or activation.shape[-1] != self.in_features
                or activation.device != self.weight.device):
            raise ValueError('floating activations on the pinned dequantized weight device required')
        flat = activation.reshape(-1, self.in_features).to(self.weight.dtype)
        if self.row_chunk is None:
            output = torch.nn.functional.linear(flat, self.weight, self.bias)
        else:
            chunks = []
            for begin in range(0, self.out_features, self.row_chunk):
                end = min(begin + self.row_chunk, self.out_features)
                bias = None if self.bias is None else self.bias[begin:end]
                chunks.append(torch.nn.functional.linear(
                    flat, self.weight[begin:end], bias))
            output = torch.cat(chunks, dim=-1)
        return output.reshape(*activation.shape[:-1], self.out_features)


class Int8WeightOnlyContract(torch.nn.Module):
    def __init__(self, weight, bias=None, *, group_size=128, residual_bits=0):
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] % group_size:
            raise ValueError('aligned weight-only INT8 matrix required')
        if bias is not None and (bias.shape != (weight.shape[0],) or not torch.isfinite(bias).all()):
            raise ValueError('finite output bias required')
        quantized, scales, dequantized, residual, residual_scales = (
            quantize_int8_residual_groups(
                weight.detach(), group_size, residual_bits=residual_bits))
        self.register_buffer('weight', dequantized.to(weight.dtype))
        self.register_buffer('group_scales', scales)
        self.register_buffer('bias', bias.detach().clone() if bias is not None else None)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.group_size = group_size
        self.residual_bits = residual_bits
        self.logical_int8_weight_bytes = quantized.numel()
        self.scale_storage_bytes = scales.numel() * scales.element_size()
        self.residual_payload_bytes = (
            quantized.shape[0] * (quantized.shape[1] // group_size)
            * math.ceil(group_size * residual_bits / 8) if residual_bits else 0)
        self.residual_scale_bytes = (
            residual_scales.numel() * residual_scales.element_size()
            if residual_scales is not None else 0)
        del quantized
        del residual
        del residual_scales

    @property
    def logical_storage_bytes(self):
        bias_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        return (self.logical_int8_weight_bytes + self.scale_storage_bytes
                + self.residual_payload_bytes + self.residual_scale_bytes
                + bias_bytes)

    @property
    def functional_dequantized_weight_bytes(self):
        return self.weight.numel() * self.weight.element_size()

    def forward(self, activation):
        if (not activation.is_floating_point() or activation.shape[-1] != self.in_features
                or activation.device != self.weight.device):
            raise ValueError('floating activations on the pinned dequantized weight device required')
        return torch.nn.functional.linear(activation.to(self.weight.dtype), self.weight, self.bias)


class Int8LinearContract(torch.nn.Module):
    def __init__(self, weight, bias=None, *, output_partitions=1):
        super().__init__()
        if (weight.ndim != 2 or weight.shape[1] % 8 or min(weight.shape) < 1
                or weight.shape[1] * 128 ** 2 > torch.iinfo(torch.int32).max
                or output_partitions < 1):
            raise ValueError('aligned matrix, positive partitions and overflow-safe INT32 reduction required')
        if bias is not None and (bias.shape != (weight.shape[0],) or not torch.isfinite(bias).all()):
            raise ValueError('finite output bias required')
        quantized = torch.empty_like(weight, dtype=torch.int8)
        scales = torch.empty((weight.shape[0], 1), dtype=torch.float32, device=weight.device)
        for begin in range(0, weight.shape[0], 1024):
            block, block_scales = quantize_int8_rows(weight.detach()[begin:begin + 1024])
            quantized[begin:begin + 1024] = block
            scales[begin:begin + 1024] = block_scales
        self.register_buffer('weight', quantized)
        self.register_buffer('scales', scales)
        self.register_buffer('bias', bias.detach().float().clone() if bias is not None else None)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.output_partitions = output_partitions

    @property
    def storage_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.buffers())

    def integer_product(self, activations):
        if (activations.dtype != torch.int8 or activations.ndim != 2
                or activations.shape[1] != self.in_features or len(activations) < 1):
            raise ValueError('nonempty aligned INT8 activation rows required')
        if self.weight.device.type != 'cuda':
            return activations.to(torch.int32) @ self.weight.t().to(torch.int32)
        blocks = []
        chunk_size = math.ceil(self.out_features / self.output_partitions / 8) * 8
        padded_activations = torch.nn.functional.pad(activations, (0, 0, 0, max(0, 32 - len(activations))))
        for begin in range(0, self.out_features, chunk_size):
            weights = self.weight[begin:begin + chunk_size]
            padded_weights = torch.nn.functional.pad(weights, (0, 0, 0, (-len(weights)) % 8))
            product = torch._int_mm(padded_activations.contiguous(), padded_weights.t())
            blocks.append(product[:len(activations), :len(weights)])
        return torch.cat(blocks, dim=1)

    def forward(self, activation):
        if (not activation.is_floating_point() or activation.shape[-1] != self.in_features
                or activation.device != self.weight.device):
            raise ValueError('floating activations on the pinned weight device required')
        flat = activation.reshape(-1, self.in_features)
        chunks = []
        for begin in range(0, len(flat), 128):
            quantized, scales = quantize_int8_rows(flat[begin:begin + 128])
            product = self.integer_product(quantized).float()
            output = product * (scales * self.scales.t())
            if self.bias is not None:
                output = output + self.bias
            converted = output.to(activation.dtype)
            require_finite(converted, 'nonfinite scaled integer output violates the functional contract')
            chunks.append(converted)
        return torch.cat(chunks).reshape(*activation.shape[:-1], self.out_features)


class Int8ReaderDispatch:
    def __init__(self, reader):
        self.reader = reader
        self.originals = []
        self.layouts = {}
        self.output_partitions = 1
        self.calls = 0
        self.quantization_seconds = 0.0

    def __enter__(self):
        if self.reader.training:
            raise ValueError('immutable eval Reader required')
        device = next(self.reader.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        import time
        started = time.perf_counter()
        try:
            for name, module in self.reader.named_modules():
                if (not isinstance(module, torch.nn.Linear)
                        or not (name.startswith('model.language_model.') or name == 'lm_head')):
                    continue
                layout = Int8LinearContract(module.weight, module.bias)
                self.layouts[name] = layout
                original = module.forward
                self.originals.append((module, original))
                module.forward = self._wrapper(layout)
            if not self.layouts:
                raise ValueError('no Reader language Linear modules found')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            self.quantization_seconds = time.perf_counter() - started
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def _wrapper(self, layout):
        def forward(activation):
            layout.output_partitions = self.output_partitions
            self.calls += 1
            return layout(activation)
        return forward

    @property
    def storage_bytes(self):
        return sum(layout.storage_bytes for layout in self.layouts.values())

    def __exit__(self, *unused):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
        self.layouts.clear()


def materialize_int8_primary(reader):
    if reader.training:
        raise ValueError('immutable eval Reader required')
    modules = dict(reader.named_modules())
    selected = {name: module for name, module in modules.items()
                if isinstance(module, torch.nn.Linear)
                and (name.startswith('model.language_model.') or name == 'lm_head')}
    if not selected:
        raise ValueError('no Reader language Linear modules found')
    aliases = {}
    for name, parameter in reader.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    selected_parameters = {name + '.weight' for name in selected}
    retained = {}
    for module in selected.values():
        external_names = set(aliases[id(module.weight)]) - selected_parameters
        for external_name in external_names:
            parent_name, _, attribute = external_name.rpartition('.')
            if not isinstance(modules[parent_name], torch.nn.Embedding) or attribute != 'weight':
                raise ValueError('only pinned BF16 embedding aliases may remain outside integer Linears')
        if external_names:
            retained[id(module.weight)] = module.weight
    bias_identities = {}
    for module in selected.values():
        bias_identities.setdefault(id(module.weight), set()).add(id(module.bias))
    if any(len(identities) > 1 for identities in bias_identities.values()):
        raise ValueError('shared weights with distinct biases require an explicitly shared scale layout')
    source_bytes = sum(parameter.numel() * parameter.element_size()
                       for parameter in {id(parameter): parameter for module in selected.values()
                                         for parameter in module.parameters()}.values())
    descriptors = [(name, (id(module.weight), id(module.bias))) for name, module in selected.items()]
    del selected, modules, module
    layouts = {}
    for name, identity in descriptors:
        module = reader.get_submodule(name)
        if identity not in layouts:
            layouts[identity] = Int8LinearContract(module.weight, module.bias).eval()
        parent_name, separator, attribute = name.rpartition('.')
        parent = reader.get_submodule(parent_name) if separator else reader
        setattr(parent, attribute, layouts[identity])
        del module
    integer_bytes = sum(layout.storage_bytes for layout in layouts.values())
    retained_bytes = sum(parameter.numel() * parameter.element_size() for parameter in retained.values())
    return {'modules': len(descriptors), 'unique_integer_layouts': len(layouts),
            'replaced_source_parameter_bytes': source_bytes, 'integer_primary_bytes': integer_bytes,
            'retained_bf16_embedding_alias_bytes': retained_bytes,
            'released_bf16_parameter_bytes': source_bytes - retained_bytes,
            'source_weight_copies_retained_by_converter': False,
            'quantization_rows_per_block': 1024,
            'replacement_lifetime': 'one module converted and detached before the next; externally tied BF16 embedding remains charged',
            'boundary': 'bounded-scratch in-memory primary conversion; tied input embedding BF16 storage explicitly retained and charged; initial BF16 checkpoint and transient conversion still charged; not a streamed quantized checkpoint load or phone capacity qualification'}


def _w8a16_fallback_modules(module_names, scope):
    if scope == 'none':
        return set()
    if scope == 'lm_head':
        return {name for name in module_names if name == 'lm_head'}
    if scope != 'output_sensitive':
        raise ValueError('known W8A16 fallback scope required')
    layer_indices = set()
    for name in module_names:
        marker = '.layers.'
        if marker in name:
            suffix = name.split(marker, 1)[1].split('.', 1)[0]
            if suffix.isdigit():
                layer_indices.add(int(suffix))
    if not layer_indices:
        raise ValueError('layer indices required for output-sensitive fallback')
    selected_layers = {min(layer_indices), max(layer_indices)}
    if len(layer_indices) > 2:
        selected_layers |= {sorted(layer_indices)[1], sorted(layer_indices)[-2]}
    fallback = {name for name in module_names if name == 'lm_head'}
    fallback |= {name for name in module_names
                 if any(f'.layers.{index}.' in name or name.endswith(f'.layers.{index}')
                        for index in selected_layers)}
    return fallback


def materialize_w8a16_primary(reader, *, group_size=128, fallback_scope='none',
                              residual_bits=0):
    if reader.training:
        raise ValueError('immutable eval Reader required')
    modules = dict(reader.named_modules())
    selected = {name: module for name, module in modules.items()
                if isinstance(module, torch.nn.Linear)
                and (name.startswith('model.language_model.') or name == 'lm_head')}
    if not selected:
        raise ValueError('no Reader language Linear modules found')
    fallback_names = _w8a16_fallback_modules(selected, fallback_scope)
    quantized_selected = {name: module for name, module in selected.items()
                          if name not in fallback_names}
    if not quantized_selected:
        raise ValueError('at least one W8A16-quantized language Linear required')
    aliases = {}
    for name, parameter in reader.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    selected_parameters = {name + '.weight' for name in selected}
    retained = {}
    for module in quantized_selected.values():
        external_names = set(aliases[id(module.weight)]) - selected_parameters
        for external_name in external_names:
            parent_name, _, attribute = external_name.rpartition('.')
            if not isinstance(modules[parent_name], torch.nn.Embedding) or attribute != 'weight':
                raise ValueError('only pinned BF16 embedding aliases may remain outside weight-only Linears')
        if external_names:
            retained[id(module.weight)] = module.weight
    source_bytes = sum(parameter.numel() * parameter.element_size()
                       for parameter in {id(parameter): parameter for module in quantized_selected.values()
                                         for parameter in module.parameters()}.values())
    fallback_bytes = sum(parameter.numel() * parameter.element_size()
                         for parameter in {id(parameter): parameter for name, module in selected.items()
                                           if name in fallback_names
                                           for parameter in module.parameters()}.values())
    descriptors = [(name, (id(module.weight), id(module.bias)))
                   for name, module in quantized_selected.items()]
    del selected, quantized_selected, modules, module
    layouts = {}
    for name, identity in descriptors:
        module = reader.get_submodule(name)
        if identity not in layouts:
            layouts[identity] = Int8WeightOnlyContract(
                module.weight, module.bias, group_size=group_size,
                residual_bits=residual_bits).eval()
        parent_name, separator, attribute = name.rpartition('.')
        parent = reader.get_submodule(parent_name) if separator else reader
        setattr(parent, attribute, layouts[identity])
        del module
    logical_bytes = sum(layout.logical_storage_bytes for layout in layouts.values())
    materialized_bytes = sum(layout.functional_dequantized_weight_bytes for layout in layouts.values())
    retained_bytes = sum(parameter.numel() * parameter.element_size() for parameter in retained.values())
    return {'modules': len(descriptors), 'unique_weight_only_layouts': len(layouts),
            'group_size': group_size, 'fallback_scope': fallback_scope,
            'residual_bits': residual_bits,
            'residual_payload_bytes': sum(layout.residual_payload_bytes
                                          for layout in layouts.values()),
            'residual_scale_bytes': sum(layout.residual_scale_bytes
                                        for layout in layouts.values()),
            'fallback_modules': sorted(fallback_names),
            'fallback_module_count': len(fallback_names),
            'retained_bf16_fallback_bytes': fallback_bytes,
            'replaced_source_parameter_bytes': source_bytes,
            'logical_int8_weight_and_scale_bytes': logical_bytes,
            'functional_dequantized_weight_bytes': materialized_bytes,
            'retained_bf16_embedding_alias_bytes': retained_bytes,
            'source_weight_copies_retained_by_converter': False,
            'functional_materialization': 'dequantized BF16 weights for exact quality replay; target PIM stores INT8 weights plus FP32 group scales',
            'boundary': 'weight-only INT8 quality reference with BF16 activations; functional replay materializes dequantized BF16 weights, so its process memory is not the target PIM footprint; initial BF16 checkpoint and transient conversion remain charged'}


def materialize_w12a16_primary(reader, *, group_size=128, row_chunk=None):
    """Replace Reader language Linears with uniform 12-bit bank-plane contracts.

    Tied BF16 embedding storage is retained during quality replay, as in the
    W8A16 path.  The target capacity design may fold that alias into the same
    bank plane; this function intentionally does not claim that saving.
    """
    if reader.training:
        raise ValueError('immutable eval Reader required')
    modules = dict(reader.named_modules())
    selected = {name: module for name, module in modules.items()
                if isinstance(module, torch.nn.Linear)
                and (name.startswith('model.language_model.') or name == 'lm_head')}
    if not selected:
        raise ValueError('no Reader language Linear modules found')
    aliases = {}
    for name, parameter in reader.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    selected_parameters = {name + '.weight' for name in selected}
    retained = {}
    for module in selected.values():
        external_names = set(aliases[id(module.weight)]) - selected_parameters
        for external_name in external_names:
            parent_name, _, attribute = external_name.rpartition('.')
            if not isinstance(modules[parent_name], torch.nn.Embedding) or attribute != 'weight':
                raise ValueError('only pinned BF16 embedding aliases may remain outside weight-only Linears')
        if external_names:
            retained[id(module.weight)] = module.weight
    source_bytes = sum(parameter.numel() * parameter.element_size()
                       for parameter in {id(parameter): parameter
                                         for module in selected.values()
                                         for parameter in module.parameters()}.values())
    descriptors = [(name, (id(module.weight), id(module.bias)))
                   for name, module in selected.items()]
    del selected, modules, module
    layouts = {}
    for name, identity in descriptors:
        module = reader.get_submodule(name)
        if identity not in layouts:
            layouts[identity] = Int12WeightOnlyContract(
                module.weight, module.bias, group_size=group_size,
                row_chunk=row_chunk).eval()
        parent_name, separator, attribute = name.rpartition('.')
        parent = reader.get_submodule(parent_name) if separator else reader
        setattr(parent, attribute, layouts[identity])
        del module
    logical_bytes = sum(layout.logical_storage_bytes for layout in layouts.values())
    materialized_bytes = sum(layout.functional_dequantized_weight_bytes
                             for layout in layouts.values())
    retained_bytes = sum(parameter.numel() * parameter.element_size()
                         for parameter in retained.values())
    return {'modules': len(descriptors), 'unique_weight_only_layouts': len(layouts),
            'group_size': group_size, 'weight_bits': 12,
            'contract_row_chunk': row_chunk,
            'replaced_source_parameter_bytes': source_bytes,
            'logical_int12_weight_and_scale_bytes': logical_bytes,
            'functional_dequantized_weight_bytes': materialized_bytes,
            'retained_bf16_embedding_alias_bytes': retained_bytes,
            'source_weight_copies_retained_by_converter': False,
            'functional_materialization': (
                'dequantized BF16 weights for exact quality replay; target PIM stores '
                'packed 12-bit weights plus FP32 group scales; row chunks align the '
                'Host control with bank-tile accumulation'),
            'capacity_boundary': (
                'retains the tied BF16 embedding alias; folding that alias into the '
                'bank plane is a separate, unexecuted capacity optimization'),
            'boundary': (
                'weight-only 12-bit quality reference with BF16 activations; functional '
                'replay materializes dequantized BF16 weights, so process memory is not '
                'the target PIM footprint')}


def tensor_signature(value):
    return (value.data_ptr(), value._version, tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device)


class BankOwnedLinear:
    def __init__(self, weight, *, banks=32, input_tile=256):
        if (weight.ndim != 2 or weight.dtype != torch.bfloat16
                or min(*weight.shape, banks, input_tile) <= 0
                or not torch.isfinite(weight).all().item()):
            raise ValueError('finite full BF16 matrix and positive layout dimensions required')
        self.source = weight
        self.signature = tensor_signature(weight)
        self.banks = banks
        self.input_tile = input_tile
        self.outputs, self.inputs = weight.shape
        self.padded_inputs = math.ceil(self.inputs / input_tile) * input_tile
        self.order = torch.cat([torch.arange(bank, self.outputs, banks, device=weight.device)
                                for bank in range(min(banks, self.outputs))])
        self.packed = torch.zeros((self.outputs, self.padded_inputs),
                                  dtype=weight.dtype, device=weight.device)
        self.packed[:, :self.inputs] = weight.detach().index_select(0, self.order)

    @property
    def resident_bytes(self):
        return self.packed.numel() * self.packed.element_size()

    def validate(self):
        if tensor_signature(self.source) != self.signature:
            raise ValueError('immutable weight identity changed after bank placement')

    def __call__(self, activation, bias=None):
        self.validate()
        if (activation.numel() != self.inputs or activation.shape[-1] != self.inputs
                or activation.dtype != torch.bfloat16 or activation.device != self.packed.device
                or not torch.isfinite(activation).all().item()):
            raise ValueError('one finite BF16 decode row on the weight device required')
        if bias is not None and (bias.shape != (self.outputs,) or bias.dtype != torch.bfloat16
                                 or not torch.isfinite(bias).all().item()):
            raise ValueError('finite BF16 output bias required')
        activation = activation.reshape(-1).float()
        accumulators = torch.zeros(self.outputs, device=activation.device, dtype=torch.float32)
        for start in range(0, self.inputs, self.input_tile):
            end = min(start + self.input_tile, self.inputs)
            accumulators.add_(torch.mv(self.packed[:, start:end].float(), activation[start:end]))
        output = torch.empty_like(accumulators)
        output.index_copy_(0, self.order, accumulators)
        if bias is not None:
            output.add_(bias.float())
        return output


@torch.no_grad()
def full_matrix_numerics(weight, activation, bank_output, host_output, *, bias=None, row_chunk=4096):
    if row_chunk < 1:
        raise ValueError('positive qualification row chunk required')
    activation = activation.reshape(-1).float()
    outputs, inputs = weight.shape
    if (activation.shape != (inputs,) or bank_output.shape != (outputs,)
            or host_output.numel() != outputs):
        raise ValueError('complete matrix and complete output required for qualification')
    reference_parts, magnitude_parts = [], []
    for start in range(0, outputs, row_chunk):
        block = weight[start:start + row_chunk].float()
        reference_parts.append(torch.mv(block, activation))
        magnitude_parts.append(torch.mv(block.abs(), activation.abs()))
    reference = torch.cat(reference_parts)
    magnitude = torch.cat(magnitude_parts)
    if bias is not None:
        reference.add_(bias.float())
        magnitude.add_(bias.float().abs())
    steps = inputs + math.ceil(inputs / 256) + 2
    unit = torch.finfo(torch.float32).eps / 2
    gamma = steps * unit / (1 - steps * unit)
    bound = 2 * gamma * magnitude + torch.finfo(torch.float32).tiny
    error = (bank_output - reference).abs()
    rounded = bank_output.to(torch.bfloat16).reshape(-1)
    host = host_output.reshape(-1)
    host_error = (rounded.float() - host.float()).abs()
    host_bound = bound + (reference.abs() + bound) * (2 * torch.finfo(torch.bfloat16).eps / 2)
    host_bound += 2 * torch.finfo(torch.bfloat16).tiny
    finite = bool(torch.isfinite(bank_output).all() and torch.isfinite(reference).all()
                  and torch.isfinite(host).all())
    return {'output_elements': outputs, 'finite': finite,
        'fp32_max_absolute_error': float(error.max()),
        'fp32_relative_l2_error': float(torch.linalg.vector_norm(error) /
                                       torch.linalg.vector_norm(reference).clamp_min(1e-30)),
        'fp32_allclose_rtol_1e4_atol_3e5': bool(torch.allclose(bank_output, reference, rtol=1e-4, atol=3e-5)),
        'forward_error_bound_pass': finite and bool((error <= bound).all()),
        'host_bf16_forward_error_bound_pass': finite and bool((host_error <= host_bound).all()),
        'host_bf16_equal_elements': int((rounded == host).sum()),
        'host_bf16_bitwise_equal': bool(torch.equal(rounded, host)),
        'host_bf16_max_absolute_error': float(host_error.max()),
        'bound_contract': 'two FP32 reduction roundoff envelopes plus BF16 output rounding; not bitwise equivalence, task quality, or physical PIM certification'}


class DecodeBankDispatcher:
    def __init__(self, reader, residency, *, mode, banks=32, input_tile=256):
        if mode not in {'shadow', 'replace'} or input_tile != 256:
            raise ValueError('shadow/replace mode and frozen input tile 256 required')
        self.reader = reader
        self.residency = residency
        self.mode = mode
        self.banks = banks
        self.input_tile = input_tile
        self.layouts = {}
        self.originals = []
        self.active = False
        self.observe = False
        self.records = []
        self.calls = 0
        self.pim_calls = 0
        self.host_calls = 0
        self.replaced_host_calls = 0

    def __enter__(self):
        aliases = {}
        for name, parameter in self.reader.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append('parameter:' + name)
        found = set()
        for name, module in self.reader.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            identity = '||'.join(sorted(aliases[id(module.weight)]))
            if identity not in self.residency.pool:
                continue
            if identity in found:
                raise ValueError('duplicate executing modules for one candidate identity unsupported')
            found.add(identity)
            original = module.forward
            self.originals.append((module, original))
            module.forward = self._wrapper(module, original, identity, name)
        if found != self.residency.pool:
            self.__exit__(None, None, None)
            raise ValueError('all frozen physical candidate matrices must have executing Linear modules')
        return self

    def _wrapper(self, module, original, identity, name):
        def forward(activation):
            if not self.active:
                return original(activation)
            if activation.numel() != module.in_features:
                raise ValueError('PIM dispatch cannot intercept prefill or multirow work')
            decision = self.residency.dispatch(identity)
            self.calls += 1
            if decision['route'] == 'host':
                self.host_calls += 1
                return original(activation)
            self.pim_calls += 1
            if decision['placement_now']:
                self.layouts[identity] = BankOwnedLinear(module.weight, banks=self.banks,
                                                         input_tile=self.input_tile)
                if self.layouts[identity].resident_bytes != self.residency.costs[identity]['resident_bytes']:
                    raise ValueError('actual bank layout bytes differ from frozen route allocation')
            layout = self.layouts[identity]
            layout.validate()
            if self.mode == 'shadow' and not self.observe:
                return original(activation)
            bank_output = layout(activation, module.bias)
            if self.observe:
                host_output = original(activation)
                numerics = full_matrix_numerics(module.weight, activation, bank_output,
                                                host_output, bias=module.bias)
                self.records.append({'module': name, 'parameter_identity': identity,
                    'shape': list(module.weight.shape), 'activation_shape': list(activation.shape),
                    'resident_bytes': layout.resident_bytes, 'numerics': numerics,
                    'decision': decision})
            if self.mode == 'shadow':
                return host_output
            self.replaced_host_calls += 1
            return bank_output.to(activation.dtype).reshape(*activation.shape[:-1], module.out_features)
        return forward

    @contextmanager
    def decode(self, *, observe=False):
        if self.active:
            raise ValueError('nested decode dispatch is unsupported')
        self.active, self.observe = True, observe
        try:
            yield
        finally:
            self.active, self.observe = False, False

    def __exit__(self, *unused):
        for module, original in self.originals:
            module.forward = original
        self.originals.clear()
        self.layouts.clear()
