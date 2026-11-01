"""Functional W8A16 bank-resident layout with on-bank group scales."""
from __future__ import annotations

import math

import torch

from .functional_decode import quantize_int8_residual_groups, require_finite


class W8A16BankLayout:
    """Place complete output-row blocks and FP32 group scales in fixed banks.

    The class is a correctness reference for the bank-resident operand layout.
    Its Torch forwards are not a PIM timing simulator and must never be used as
    an end-to-end latency measurement.
    """

    def __init__(self, weight, *, group_size=128, bank_count=256, tile_rows=16,
                 residual_bits=0):
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                or weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all().item()
                or weight.shape[1] % group_size):
            raise ValueError('finite BF16 matrix aligned to the W8A16 group size required')
        if bank_count <= 0 or tile_rows <= 0:
            raise ValueError('positive bank count and row tile required')
        if residual_bits not in (0, 2, 4):
            raise ValueError('residual bits must be zero, two or four')
        quantized, scales, _, residual, residual_scales = (
            quantize_int8_residual_groups(
                weight.detach(), group_size, residual_bits=residual_bits))
        self.rows, self.columns = weight.shape
        self.group_size = group_size
        self.residual_bits = residual_bits
        self.groups = self.columns // group_size
        self.bank_count = bank_count
        self.tile_rows = tile_rows
        self.row_blocks = math.ceil(self.rows / tile_rows)
        self.active_banks = min(self.row_blocks, bank_count)
        self.rounds = math.ceil(self.row_blocks / bank_count)
        self.rows_per_bank = self.rounds * tile_rows
        self.padded_rows = self.row_blocks * tile_rows
        self.storage = torch.zeros(
            (bank_count, self.rows_per_bank, self.columns),
            dtype=torch.int8, device=weight.device)
        self.group_scales = torch.zeros(
            (bank_count, self.rows_per_bank, self.groups),
            dtype=torch.float32, device=weight.device)
        self.residual = torch.zeros(
            (bank_count, self.rows_per_bank, self.columns),
            dtype=torch.int8, device=weight.device)
        self.residual_scales = torch.zeros(
            (bank_count, self.rows_per_bank, self.groups),
            dtype=torch.float32, device=weight.device)
        for block in range(self.row_blocks):
            bank = block % bank_count
            row_round = block // bank_count
            begin = block * tile_rows
            end = min(begin + tile_rows, self.rows)
            count = end - begin
            destination = row_round * tile_rows
            self.storage[bank, destination:destination + count].copy_(quantized[begin:end])
            self.group_scales[bank, destination:destination + count].copy_(scales[begin:end])
            if residual_bits:
                self.residual[bank, destination:destination + count].copy_(
                    residual[begin:end])
                self.residual_scales[bank, destination:destination + count].copy_(
                    residual_scales[begin:end])
        self.storage.requires_grad_(False)
        self.group_scales.requires_grad_(False)
        self.residual.requires_grad_(False)
        self.residual_scales.requires_grad_(False)

    @property
    def resident_bytes(self):
        return (self.storage.numel() * self.storage.element_size()
                + self.group_scales.numel() * self.group_scales.element_size()
                + (self.residual.numel() * self.residual.element_size()
                   + self.residual_scales.numel() * self.residual_scales.element_size()
                   if self.residual_bits else 0))

    @property
    def logical_bytes(self):
        residual_bytes = (self.rows * math.ceil(self.columns * self.residual_bits / 8)
                          + self.rows * self.groups * 4
                          if self.residual_bits else 0)
        return self.rows * self.columns + self.rows * self.groups * 4 + residual_bytes

    def inventory(self):
        return {
            'layout': 'row-block-major within bank; INT8 weights and FP32 group scales bank-resident',
            'rows': self.rows, 'columns': self.columns, 'padded_rows': self.padded_rows,
            'group_size': self.group_size, 'groups_per_row': self.groups,
            'residual_bits': self.residual_bits,
            'bank_count': self.bank_count, 'active_banks': self.active_banks,
            'row_blocks': self.row_blocks, 'rounds': self.rounds,
            'rows_per_bank': self.rows_per_bank,
            'weight_bytes': self.storage.numel() * self.storage.element_size(),
            'group_scale_bytes': (self.group_scales.numel()
                                  * self.group_scales.element_size()),
            'resident_bytes': self.resident_bytes,
            'logical_bytes': self.logical_bytes,
            'weight_copies': 1,
            'boundary': ('functional Torch layout only; residual codes are stored as '
                         'int8 for replay, while logical_bytes charges packed bits'),
        }

    def gemv(self, activation):
        if (not isinstance(activation, torch.Tensor) or activation.dtype != torch.bfloat16
                or activation.shape != (self.columns,)
                or activation.device != self.storage.device
                or not torch.isfinite(activation).all().item()):
            raise ValueError('one finite BF16 activation row on the layout device required')
        output = torch.zeros(self.padded_rows, dtype=torch.bfloat16,
                             device=activation.device)
        for row_round in range(self.rounds):
            for bank in range(self.active_banks):
                begin = row_round * self.tile_rows
                quantized = self.storage[bank, begin:begin + self.tile_rows]
                scales = self.group_scales[bank, begin:begin + self.tile_rows]
                dequantized = (quantized.float().reshape(
                    self.tile_rows, self.groups, self.group_size)
                    * scales[:, :, None]).reshape(self.tile_rows, self.columns)
                if self.residual_bits:
                    residual = self.residual[bank, begin:begin + self.tile_rows]
                    residual_scales = self.residual_scales[
                        bank, begin:begin + self.tile_rows]
                    dequantized = (dequantized + residual.float().reshape(
                        self.tile_rows, self.groups, self.group_size).mul(
                            residual_scales[:, :, None]).reshape(
                                self.tile_rows, self.columns))
                dequantized = dequantized.to(torch.bfloat16)
                values = torch.nn.functional.linear(
                    activation.unsqueeze(0), dequantized).squeeze(0)
                first = (row_round * self.bank_count + bank) * self.tile_rows
                valid = min(self.tile_rows, self.rows - first)
                if valid > 0:
                    output[first:first + valid].copy_(values[:valid])
        result = output[:self.rows].contiguous()
        require_finite(result, 'nonfinite W8A16 bank output')
        return result
