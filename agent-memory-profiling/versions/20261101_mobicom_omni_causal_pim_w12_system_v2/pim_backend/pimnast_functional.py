"""Fixed PIMnast CR/CRO-1 layout and integer semantics, independent of timing."""
from __future__ import annotations

import math
import numpy as np


def frozen_route(projection, user):
    from pathlib import Path
    if (projection.get('operand_bits') != 8 or not projection.get('complete')
            or str(user) not in {'0', '1', '2'}):
        raise ValueError('completed INT8 architecture projection for authorized development users required')
    profiles = [profile for profile in projection['per_user']
                if Path(profile['run']).stem.endswith('_user_' + str(user))]
    if len(profiles) != 1:
        raise ValueError('one frozen user-specific placement required')
    profile = profiles[0]
    if profile['group_residency_policy']['future_call_count_used_by_objective'] is not False:
        raise ValueError('future call counts cannot determine online residency')
    candidates = {item['name']: {key: item[key] for key in
                  ('shape', 'model_route', 'candidate_resident_bytes')} for item in profile['matrices']}
    if (not candidates or len(candidates) != len(profile['matrices'])
            or any(item['model_route'] not in {'host', 'pim'} for item in candidates.values())):
        raise ValueError('unique explicit Host/PIM matrix placements required')
    return candidates, profile['declared_pim_pool_bytes']


class PimnastInt8Layout:
    banks = 128
    banks_per_channel = 16
    tile_rows = 16
    tile_columns = 16
    interleave_bytes = 256
    dram_row_bytes = 2048
    simd_bytes = 32
    output_register_bytes = 64

    def __init__(self, weights, *, allow_row_padding=False):
        if (not isinstance(weights, np.ndarray) or weights.dtype != np.int8
                or weights.ndim != 2 or min(weights.shape) < 1
                or weights.shape[0] % self.tile_rows
                or weights.shape[1] % self.tile_columns):
            raise ValueError('nonempty signed INT8 matrix with exact 16x16 tiles required')
        if weights.shape[1] * 128 ** 2 > np.iinfo(np.int32).max:
            raise ValueError('worst-case signed INT8 accumulation may overflow INT32')
        self.rows, self.columns = weights.shape
        self.row_blocks = self.rows // self.tile_rows
        self.active_banks = min(self.row_blocks, self.banks)
        self.row_rounds = math.ceil(self.row_blocks / self.banks)
        self.padded_rows = (self.row_blocks +
                            (-self.row_blocks % self.banks)) * self.tile_rows
        if self.padded_rows != self.rows and not allow_row_padding:
            raise ValueError('row padding must be explicitly authorized')
        self.column_tiles = self.columns // self.tile_columns
        self.used_bytes_per_bank = self.row_rounds * self.column_tiles * self.interleave_bytes
        bank_bytes = math.ceil(self.used_bytes_per_bank / self.dram_row_bytes) * self.dram_row_bytes
        self.storage = np.zeros((self.banks, bank_bytes), dtype=np.int8)
        for row_block in range(self.row_blocks):
            bank = row_block % self.banks
            row_round = row_block // self.banks
            offset = row_round * self.column_tiles * self.interleave_bytes
            block = weights[row_block * self.tile_rows:(row_block + 1) * self.tile_rows]
            tiled = block.reshape(self.tile_rows, self.column_tiles, self.tile_columns)
            self.storage[bank, offset:offset + self.column_tiles * self.interleave_bytes] = (
                tiled.transpose(1, 2, 0).reshape(-1))
        self.storage.flags.writeable = False

    def address(self, row, column):
        if (not isinstance(row, (int, np.integer)) or not isinstance(column, (int, np.integer))
                or not 0 <= row < self.rows or not 0 <= column < self.columns):
            raise ValueError('logical matrix coordinate out of range')
        row_block, row_lane = divmod(int(row), self.tile_rows)
        column_tile, column_lane = divmod(int(column), self.tile_columns)
        bank = row_block % self.banks
        tile = row_block // self.banks * self.column_tiles + column_tile
        bank_offset = tile * self.interleave_bytes + column_lane * self.tile_rows + row_lane
        physical_address = ((bank_offset // self.interleave_bytes * self.banks + bank)
                            * self.interleave_bytes + bank_offset % self.interleave_bytes)
        return {'channel': bank // self.banks_per_channel,
                'bank_in_channel': bank % self.banks_per_channel,
                'bank': bank, 'bank_offset': bank_offset,
                'dram_row': bank_offset // self.dram_row_bytes,
                'dram_column_byte': bank_offset % self.dram_row_bytes,
                'physical_byte_address': physical_address}

    def inventory(self):
        return {'layout': 'CR, column-major 16x16 tile, CRO-1, SPLIT_K=1',
                'rows': self.rows, 'columns': self.columns, 'padded_rows': self.padded_rows,
                'bank_count': self.banks, 'active_banks': self.active_banks,
                'channels': 8, 'banks_per_channel': 16,
                'interleave_bytes': self.interleave_bytes, 'simd_operand_bytes': self.simd_bytes,
                'orf_accumulator_bytes_per_active_row_block': self.output_register_bytes,
                'upstream_required_output_orf_registers': 2,
                'upstream_required_input_irf_registers': 1,
                'upstream_orf_register_bytes': 64,
                'upstream_irf_register_bytes': 32,
                'instruction_or_register_allocation_verified': False,
                'input_tile_bytes': self.tile_columns,
                'dense_weight_bytes': self.rows * self.columns,
                'resident_layout_bytes': self.storage.nbytes,
                'row_padding_bytes': (self.padded_rows - self.rows) * self.columns,
                'dram_alignment_padding_bytes': self.storage.nbytes - self.padded_rows * self.columns,
                'used_bytes_per_bank': self.used_bytes_per_bank,
                'simd_fetches_per_gemv': self.row_rounds * self.column_tiles * self.banks * 8,
                'timing_ns': None,
                'boundary': 'numerical layout reference, not PIM instructions, timing, energy or physical hardware'}

    def gemv(self, vector):
        if (not isinstance(vector, np.ndarray) or vector.dtype != np.int8
                or vector.shape != (self.columns,)):
            raise ValueError('one complete signed INT8 vector required')
        tiles = self.storage[:, :self.used_bytes_per_bank].reshape(
            self.banks, self.row_rounds, self.column_tiles,
            self.tile_columns, self.tile_rows)
        output = np.zeros(self.padded_rows, dtype=np.int32)
        for row_round in range(self.row_rounds):
            accumulator = np.zeros((self.banks, self.tile_rows), dtype=np.int64)
            for column_tile in range(self.column_tiles):
                for column_pair in range(0, self.tile_columns, 2):
                    fetched = tiles[:, row_round, column_tile, column_pair:column_pair + 2]
                    scalar_offset = column_tile * self.tile_columns + column_pair
                    scalars = vector[scalar_offset:scalar_offset + 2].astype(np.int64)
                    products = fetched.astype(np.int64) * scalars[None, :, None]
                    accumulator += products[:, 0, :]
                    accumulator += products[:, 1, :]
            begin = row_round * self.banks * self.tile_rows
            output[begin:begin + self.banks * self.tile_rows] = accumulator.reshape(-1)
        if np.any(output[self.rows:]):
            raise AssertionError('padded rows must have zero output')
        return output[:self.rows].copy()


class TorchPimnastInt8Layout:
    def __init__(self, weights, *, workspace_bytes=256 * 1024 ** 2):
        import torch
        if (not isinstance(weights, torch.Tensor) or weights.dtype != torch.int8
                or weights.ndim != 2 or not isinstance(workspace_bytes, int)
                or workspace_bytes < 1):
            raise ValueError('INT8 matrix and positive bounded functional workspace required')
        reference = PimnastInt8Layout(weights.detach().cpu().numpy(), allow_row_padding=True)
        self.inventory_data = reference.inventory()
        self.rows, self.columns = weights.shape
        self.row_rounds = reference.row_rounds
        self.used_bytes_per_bank = reference.used_bytes_per_bank
        self.storage = torch.tensor(reference.storage, dtype=torch.int8, device=weights.device)
        round_temporary_bytes = 128 * 16 * self.columns * 8
        if workspace_bytes < round_temporary_bytes:
            raise ValueError('functional workspace cannot hold one bank-owned row round')
        self.round_chunk = min(self.row_rounds, workspace_bytes // round_temporary_bytes)
        self.workspace_bound_bytes = self.round_chunk * round_temporary_bytes

    def gemv(self, vector):
        import torch
        if (not isinstance(vector, torch.Tensor) or vector.dtype != torch.int8
                or vector.shape != (self.columns,) or vector.device != self.storage.device):
            raise ValueError('one complete INT8 vector on the layout device required')
        tiles = self.storage[:, :self.used_bytes_per_bank].reshape(
            128, self.row_rounds, self.columns // 16, 16, 16)
        inputs = vector.to(torch.int32).reshape(1, 1, self.columns // 16, 16, 1)
        outputs = []
        for begin in range(0, self.row_rounds, self.round_chunk):
            products = tiles[:, begin:begin + self.round_chunk].to(torch.int32) * inputs
            outputs.append(products.sum(dim=(2, 3), dtype=torch.int32).permute(1, 0, 2).reshape(-1))
            del products
        return torch.cat(outputs)[:self.rows].contiguous()


class PimnastReaderDispatch:
    def __init__(self, reader, candidates):
        self.reader = reader
        self.candidates = candidates
        self.layouts = {}
        self.originals = []
        self.hook = None
        self.decode = False
        self.calls = {}
        self.host_fallback_calls = 0

    def __enter__(self):
        from .functional_decode import Int8LinearContract
        if self.reader.training:
            raise ValueError('immutable eval Reader required')
        modules = dict(self.reader.named_modules())
        try:
            for name, candidate in sorted(self.candidates.items()):
                if candidate['model_route'] != 'pim':
                    continue
                module = modules.get(name)
                if not isinstance(module, Int8LinearContract):
                    raise ValueError('selected PIM operator is not a shared INT8 primary: ' + name)
                if 'shape' in candidate and candidate['shape'] != f'{module.out_features}x{module.in_features}':
                    raise ValueError('frozen matrix shape differs from primary: ' + name)
                layout = TorchPimnastInt8Layout(module.weight)
                if layout.inventory_data['resident_layout_bytes'] != candidate['candidate_resident_bytes']:
                    raise ValueError('functional layout and timing placement differ: ' + name)
                self.layouts[name] = layout
                self.calls[name] = 0
                original = module.integer_product
                previous = module.__dict__.get('integer_product')
                self.originals.append((module, previous))
                module.integer_product = self.make_product(name, layout, original)
            self.hook = self.reader.register_forward_pre_hook(self.set_phase, with_kwargs=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def set_phase(self, module, inputs, kwargs):
        self.decode = kwargs.get('past_key_values') is not None

    def make_product(self, name, layout, original):
        def product(activations):
            if self.decode and activations.shape == (1, layout.columns):
                self.calls[name] += 1
                return layout.gemv(activations[0]).reshape(1, -1)
            self.host_fallback_calls += 1
            return original(activations)
        return product

    def summary(self):
        return {'selected_layouts': len(self.layouts),
                'resident_layout_bytes': sum(layout.storage.numel() for layout in self.layouts.values()),
                'functional_max_workspace_bound_bytes': max(
                    (layout.workspace_bound_bytes for layout in self.layouts.values()), default=0),
                'calls_per_module': dict(self.calls), 'prefill_or_multirow_host_fallback_calls': self.host_fallback_calls,
                'unobserved_selected_modules': sorted(name for name, count in self.calls.items() if not count),
                'boundary': 'Torch executes bank-layout integer reference; not physical PIM, a timing simulator or GPU/PIM latency comparison; workspace describes this functional implementation only'}

    def __exit__(self, *args):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        for module, previous in reversed(self.originals):
            if previous is None:
                del module.integer_product
            else:
                module.integer_product = previous
        self.originals.clear()
