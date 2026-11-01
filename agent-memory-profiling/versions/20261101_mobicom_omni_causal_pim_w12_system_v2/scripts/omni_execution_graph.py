"""Observe actual ATen data and alias hazards without changing tensor values."""
from __future__ import annotations

from collections import Counter
from functools import wraps
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def tensors_in(value):
    return [item for item in tree_flatten(value)[0] if isinstance(item, torch.Tensor)]


class ExecutionGraph(TorchDispatchMode):
    def __init__(self, *, context_identity, max_events=20000):
        super().__init__()
        if not context_identity or max_events < 1:
            raise ValueError('immutable execution context and positive event limit required')
        self.context_identity = context_identity
        self.max_events = max_events
        self.events, self.values, self.allocations = [], [], []
        self.opaque_regions = []
        self._objects, self._storage = {}, {}
        self._writers, self._readers = {}, {}
        self._linear_stack, self._hooks = [], []
        self._opaque_restore = []

    def _allocation(self, tensor):
        if tensor.layout != torch.strided:
            raise ValueError('only strided tensors are qualified for this observer')
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata)
        previous = self._storage.get(key)
        if previous is None or not any(reference() is not None for reference in previous[1]):
            allocation_id = len(self.allocations)
            self.allocations.append({'allocation_id': allocation_id, 'device': str(tensor.device),
                                     'storage_bytes': storage.nbytes(), 'external_labels': []})
            previous = (allocation_id, [])
            self._storage[key] = previous
        previous[1].append(weakref.ref(tensor))
        return previous[0]

    def _value(self, tensor, *, producer=None, force=False):
        previous = self._objects.get(id(tensor))
        if not force and previous is not None and previous[0]() is tensor:
            return previous[1]
        value_id = len(self.values)
        self.values.append({'value_id': value_id, 'allocation_id': self._allocation(tensor),
            'producer_event': producer, 'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'storage_offset': tensor.storage_offset(), 'dtype': str(tensor.dtype),
            'logical_bytes': tensor.numel() * tensor.element_size()})
        self._objects[id(tensor)] = (weakref.ref(tensor), value_id)
        return value_id

    def register_external(self, tensor, label):
        value_id = self._value(tensor)
        allocation = self.allocations[self.values[value_id]['allocation_id']]
        if label not in allocation['external_labels']:
            allocation['external_labels'].append(label)

    def attach_linears(self, model):
        if self._hooks:
            raise ValueError('linear observation already attached')
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            def enter(current, arguments, module_name=name):
                activation = arguments[0]
                self._linear_stack.append({'name': module_name,
                    'input_value': self._value(activation), 'weight_value': self._value(current.weight),
                    'in_features': current.in_features, 'out_features': current.out_features,
                    'rows': activation.numel() // activation.shape[-1],
                    'weight_element_bytes': current.weight.element_size()})

            def leave(unused_module, unused_arguments, unused_output):
                self._linear_stack.pop()

            self._hooks.append(module.register_forward_pre_hook(enter))
            self._hooks.append(module.register_forward_hook(leave, always_call=True))

    def close(self):
        for module, name, original in reversed(self._opaque_restore):
            setattr(module, name, original)
        self._opaque_restore.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self._linear_stack:
            raise ValueError('unbalanced module execution scope')

    def attach_opaque_regions(self, model):
        if self._opaque_restore:
            raise ValueError('opaque region observation already attached')
        for module_name, module in model.named_modules():
            for name in ('causal_conv1d_update', 'recurrent_gated_delta_rule', 'chunk_gated_delta_rule'):
                if not hasattr(module, name):
                    continue
                original = getattr(module, name)
                if not callable(original):
                    continue
                region_name = (module_name or module.__class__.__name__) + '.' + name
                mutation_index = 1 if name == 'causal_conv1d_update' else None

                @wraps(original)
                def wrapper(*args, _original=original, _region_name=region_name,
                            _mutation_index=mutation_index, **kwargs):
                    input_values = [self._value(tensor) for tensor in tensors_in((args, kwargs))]
                    mutations = []
                    if _mutation_index is not None:
                        state = args[_mutation_index] if len(args) > _mutation_index else kwargs.get('conv_state')
                        if state is None:
                            raise ValueError('causal convolution state mutation contract missing')
                        mutations = tensors_in(state)
                    start_event = len(self.events)
                    result = _original(*args, **kwargs)
                    self.finish_opaque_region(_region_name, input_values, result,
                        start_event=start_event, mutated_tensors=mutations,
                        implementation=getattr(_original, '__module__', '') + '.' +
                                       getattr(_original, '__qualname__', type(_original).__name__))
                    return result

                setattr(module, name, wrapper)
                self._opaque_restore.append((module, name, original))

    def finish_opaque_region(self, name, input_values, outputs, *, start_event,
                             mutated_tensors, implementation):
        if len(self.events) >= self.max_events:
            raise ValueError('execution graph event bound exceeded; no silent truncation')
        event_id = len(self.events)
        child_events = list(range(start_event, event_id))
        read_allocations = {self.values[value]['allocation_id'] for value in input_values}
        dependencies = {self.values[value]['producer_event'] for value in input_values}
        dependencies.update(child_events)
        dependencies.update(self._writers.get(allocation) for allocation in read_allocations)
        output_tensors = tensors_in(outputs)
        by_object = {}
        for tensor in output_tensors + mutated_tensors:
            if id(tensor) not in by_object:
                by_object[id(tensor)] = self._value(tensor, producer=event_id, force=True)
        output_values = [by_object[id(tensor)] for tensor in output_tensors]
        mutated_values = [by_object[id(tensor)] for tensor in mutated_tensors]
        write_allocations = {self.values[value]['allocation_id'] for value in by_object.values()}
        for allocation in write_allocations:
            dependencies.add(self._writers.get(allocation))
            dependencies.update(self._readers.get(allocation, set()))
        dependencies.discard(None)
        for allocation in read_allocations - write_allocations:
            self._readers.setdefault(allocation, set()).add(event_id)
        for allocation in write_allocations:
            self._writers[allocation], self._readers[allocation] = event_id, set()
        self.events.append({'event_id': event_id, 'operator': 'opaque::' + name,
            'input_values': input_values, 'output_values': output_values,
            'mutated_values': mutated_values, 'dependencies': sorted(dependencies),
            'write_allocations': sorted(write_allocations), 'linear_scope': None,
            'nondeterministic_seeded': False, 'opaque_region_completion': True})
        self.opaque_regions.append({'name': name, 'start_event': start_event,
            'end_event': event_id, 'completion_event': event_id,
            'input_values': input_values, 'output_values': output_values,
            'mutated_values': mutated_values, 'child_event_ids': child_events,
            'implementation': implementation, 'materialized_event': True,
            'duration_seconds': None,
            'boundary': 'inclusive callable completion, not an additional device kernel/write; child events retained, not additive timing; internal FLA/Triton traffic remains unknown'})

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if len(self.events) >= self.max_events:
            raise ValueError('execution graph event bound exceeded; no silent truncation')
        input_tensors = tensors_in((args, kwargs))
        input_values = [self._value(tensor) for tensor in input_tensors]
        read_allocations = {self.values[value]['allocation_id'] for value in input_values}
        mutated = []
        for index, argument in enumerate(function._schema.arguments):
            if argument.alias_info is not None and argument.alias_info.is_write:
                actual = args[index] if index < len(args) else kwargs.get(argument.name)
                mutated.extend(tensors_in(actual))
        write_allocations = {self.values[self._value(tensor)]['allocation_id'] for tensor in mutated}
        dependencies = {self.values[value]['producer_event'] for value in input_values}
        dependencies.update(self._writers.get(allocation) for allocation in read_allocations)
        for allocation in write_allocations:
            dependencies.update(self._readers.get(allocation, set()))
        dependencies.discard(None)
        result = function(*args, **kwargs)
        event_id = len(self.events)
        result_tensors = tensors_in(result)
        outputs, by_object = [], {}
        for tensor in result_tensors + mutated:
            if id(tensor) not in by_object:
                by_object[id(tensor)] = self._value(tensor, producer=event_id, force=True)
            if len(outputs) < len(result_tensors):
                outputs.append(by_object[id(tensor)])
        new_allocations = {self.values[value]['allocation_id'] for value in by_object.values()}
        write_allocations.update(new_allocations - read_allocations)
        for allocation in read_allocations:
            self._readers.setdefault(allocation, set()).add(event_id)
        for allocation in write_allocations:
            self._writers[allocation] = event_id
            self._readers[allocation] = set()
        self.events.append({'event_id': event_id, 'operator': str(function),
            'input_values': input_values, 'output_values': outputs,
            'mutated_values': [by_object[id(tensor)] for tensor in mutated],
            'dependencies': sorted(dependencies), 'write_allocations': sorted(write_allocations),
            'linear_scope': dict(self._linear_stack[-1]) if self._linear_stack else None,
            'nondeterministic_seeded': torch.Tag.nondeterministic_seeded in function.tags})
        return result

    def summary(self):
        if any(dependency >= event['event_id'] for event in self.events
               for dependency in event['dependencies']):
            raise ValueError('execution dependencies must precede their consumer')
        linears = []
        for event in self.events:
            if event['operator'] in {'aten.linear.default', 'aten.mm.default',
                                     'aten.addmm.default', 'aten.bmm.default'}:
                scope = event['linear_scope']
                if scope is not None:
                    linears.append({'event_id': event['event_id'], **scope})
        shared_inputs = []
        groups = {}
        for linear in linears:
            groups.setdefault(linear['input_value'], []).append(linear)
        for value, group in groups.items():
            if len(group) > 1:
                shared_inputs.append({'input_value': value, 'linear_events': group})
        return {'schema_version': 'omni-actual-execution-graph-with-opaque-boundaries-v2',
            'context_identity': self.context_identity, 'events': self.events, 'values': self.values,
            'allocations': self.allocations, 'operator_counts': dict(Counter(
                event['operator'] for event in self.events)), 'linear_events': linears,
            'opaque_regions': self.opaque_regions,
            'shared_linear_input_values': shared_inputs,
            'boundary': 'actual observed ATen dataflow and conservative allocation read/write hazards; logical tensor bytes are not DRAM traffic; no per-kernel timing, simulator cycles, state-reuse authorization, full-request timeline or hardware speedup'}
