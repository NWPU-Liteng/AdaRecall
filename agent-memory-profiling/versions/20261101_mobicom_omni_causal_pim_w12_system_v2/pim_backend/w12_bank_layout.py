"""Functional 12-bit bit-sliced bank layout for the HyperPIM-SC decoder.

The layout stores twelve binary weight planes and one FP32 group-scale plane
inside bank-owned row blocks.  It is a numerical and capacity reference for a
target bank-PIM memory; Torch executes the arithmetic on the host device and
the measured wall clock is therefore not a PIM latency result.
"""
from __future__ import annotations

import math

import torch

from .functional_decode import quantize_signed_groups, require_finite


_BIT_SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


def _pack_plane_bits(bits):
    """Pack ``[rows, padded_columns]`` 0/1 bytes into big-endian bytes."""
    if bits.ndim != 2 or bits.dtype != torch.uint8:
        raise ValueError("a two-dimensional uint8 bit matrix is required")
    rows, columns = bits.shape
    if columns % 8:
        raise ValueError("bit-matrix columns must be eight-aligned")
    shifts = _BIT_SHIFTS.to(bits.device)
    return ((bits.reshape(rows, columns // 8, 8) << shifts).sum(dim=2)
            .to(torch.uint8))


def _unpack_plane_bits(packed, columns):
    """Unpack big-endian bytes into ``[rows, columns]`` 0/1 bytes."""
    if packed.ndim != 2 or packed.dtype != torch.uint8 or columns <= 0:
        raise ValueError("packed uint8 rows and a positive column count are required")
    shifts = _BIT_SHIFTS.to(packed.device)
    bits = ((packed.unsqueeze(2) >> shifts) & 1).to(torch.uint8)
    bits = bits.reshape(packed.shape[0], -1)
    return bits[:, :columns]


class W12BankLayout:
    """Twelve-bit signed weights in bank-owned bit planes and group scales.

    Each logical output row occupies ``ceil(12 * columns / 8)`` bytes, rounded
    up to a physical DRAM row.  The twelve planes are packed contiguously in
    that row.  A row block of ``tile_rows`` output rows is assigned to one bank
    and repeated across banks in fixed rounds.  This is the same row ownership
    used by the capacity audit and timeline projection.
    """

    weight_bits = 12
    levels = 2 ** 11 - 1

    def __init__(self, weight, *, group_size=128, bank_count=256, tile_rows=16,
                 row_bytes=2048, bias=None, group_scales=None):
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
                or weight.dtype != torch.bfloat16
                or not torch.isfinite(weight).all().item()):
            raise ValueError("finite BF16 weight matrix required")
        if (not isinstance(group_size, int) or isinstance(group_size, bool)
                or group_size <= 0 or weight.shape[1] % group_size):
            raise ValueError("columns must align to a positive group size")
        if (not isinstance(bank_count, int) or bank_count <= 0
                or not isinstance(tile_rows, int) or tile_rows <= 0
                or not isinstance(row_bytes, int) or row_bytes <= 0):
            raise ValueError("positive bank, tile and physical-row dimensions required")
        if bias is not None and (bias.shape != (weight.shape[0],)
                                 or bias.dtype != torch.bfloat16
                                 or not torch.isfinite(bias).all().item()):
            raise ValueError("finite BF16 output bias with one value per row required")

        self.rows, self.columns = (int(value) for value in weight.shape)
        self.has_bias = bias is not None
        self.group_size = group_size
        self.groups = self.columns // group_size
        self.bank_count = bank_count
        self.tile_rows = tile_rows
        self.row_bytes = row_bytes
        self.plane_bytes = math.ceil(self.columns / 8)
        self.bitplane_row_bytes = self.weight_bits * self.plane_bytes
        self.physical_row_bytes = (math.ceil(self.bitplane_row_bytes / row_bytes)
                                   * row_bytes)
        self.row_blocks = math.ceil(self.rows / tile_rows)
        self.active_banks = min(self.row_blocks, bank_count)
        self.rounds = math.ceil(self.row_blocks / bank_count)
        self.rows_per_bank = self.rounds * tile_rows
        self.padded_rows = self.active_banks * self.rows_per_bank

        if group_scales is None:
            quantized, scales, _ = quantize_signed_groups(
                weight.detach(), group_size, weight_bits=self.weight_bits)
        else:
            if (not isinstance(group_scales, torch.Tensor)
                    or group_scales.shape != (self.rows, self.groups)
                    or group_scales.dtype != torch.float32
                    or not torch.isfinite(group_scales).all().item()
                    or not (group_scales > 0).all().item()):
                raise ValueError("finite positive FP32 group-scale matrix required")
            scales = group_scales.detach()
            floating = weight.detach().float().reshape(
                self.rows, self.groups, group_size)
            quantized = (floating / scales[:, :, None]).round().clamp(
                -self.levels, self.levels).to(torch.int16).reshape(
                    self.rows, self.columns)

        self.storage = torch.zeros(
            (self.active_banks, self.rows_per_bank, self.physical_row_bytes),
            dtype=torch.uint8, device=weight.device)
        self.group_scales = torch.zeros(
            (self.active_banks, self.rows_per_bank, self.groups),
            dtype=torch.float32, device=weight.device)
        self.bias = torch.zeros(
            (self.active_banks, self.rows_per_bank), dtype=torch.bfloat16,
            device=weight.device)

        with torch.no_grad():
            chunk_rows = 512
            for begin in range(0, self.rows, chunk_rows):
                end = min(begin + chunk_rows, self.rows)
                codes = quantized[begin:end]
                packed = torch.empty(
                    (end - begin, self.weight_bits, self.plane_bytes),
                    dtype=torch.uint8, device=weight.device)
                padded_columns = self.plane_bytes * 8
                for plane in range(self.weight_bits):
                    bits = ((codes >> plane) & 1).to(torch.uint8)
                    if padded_columns != self.columns:
                        padding = torch.zeros(
                            (end - begin, padded_columns - self.columns),
                            dtype=torch.uint8, device=weight.device)
                        bits = torch.cat((bits, padding), dim=1)
                    packed[:, plane] = _pack_plane_bits(bits)
                packed = packed.reshape(end - begin, self.bitplane_row_bytes)

                logical = torch.arange(begin, end, device=weight.device)
                block = torch.div(logical, tile_rows, rounding_mode="floor")
                bank = block % bank_count
                row_round = torch.div(block, bank_count, rounding_mode="floor")
                slot = row_round * tile_rows + (logical % tile_rows)
                if bool((bank >= self.active_banks).any().item()):
                    raise AssertionError("row mapping exceeded active banks")
                self.storage[bank, slot, :self.bitplane_row_bytes] = packed
                self.group_scales[bank, slot] = scales[begin:end]
                if bias is not None:
                    self.bias[bank, slot] = bias[begin:end]

        self.storage.requires_grad_(False)
        self.group_scales.requires_grad_(False)
        self.bias.requires_grad_(False)

    @property
    def weight_bytes(self):
        return self.storage.numel() * self.storage.element_size()

    @property
    def scale_bytes(self):
        return self.group_scales.numel() * self.group_scales.element_size()

    @property
    def bias_bytes(self):
        return self.bias.numel() * self.bias.element_size()

    @property
    def resident_bytes(self):
        return self.weight_bytes + self.scale_bytes + self.bias_bytes

    @property
    def logical_weight_bytes(self):
        return self.rows * math.ceil(self.columns * self.weight_bits / 8)

    @property
    def logical_bytes(self):
        return (self.logical_weight_bytes
                + self.rows * self.groups * 4
                + (self.rows * 2 if self.has_bias else 0))

    @property
    def external_transfer_bytes_per_token(self):
        return self.columns * 2 + self.rows * 4

    def inventory(self):
        return {
            "layout": "12 bit planes, row-packed, bank-owned group scales",
            "weight_bits": self.weight_bits,
            "rows": self.rows,
            "columns": self.columns,
            "padded_rows": self.padded_rows,
            "group_size": self.group_size,
            "groups_per_row": self.groups,
            "bank_count": self.bank_count,
            "active_banks": self.active_banks,
            "row_blocks": self.row_blocks,
            "rounds": self.rounds,
            "rows_per_bank": self.rows_per_bank,
            "tile_rows": self.tile_rows,
            "logical_row_bytes": math.ceil(self.columns * self.weight_bits / 8),
            "bitplane_row_bytes": self.bitplane_row_bytes,
            "physical_row_bytes": self.physical_row_bytes,
            "physical_rows_per_output_row": self.physical_row_bytes // self.row_bytes,
            "weight_bytes": self.weight_bytes,
            "scale_bytes": self.scale_bytes,
            "bias_bytes": self.bias_bytes,
            "has_bias": self.has_bias,
            "resident_bytes": self.resident_bytes,
            "logical_bytes": self.logical_bytes,
            "weight_copies": 1,
            "boundary": ("functional bit-plane reference; Torch executes the "
                         "reconstruction and MAC, not bank hardware"),
        }

    def _decode_tile_codes(self, slot_begin, bank_begin, bank_end):
        planes = self.storage[
            bank_begin:bank_end, slot_begin:slot_begin + self.tile_rows,
            :self.bitplane_row_bytes]
        planes = planes.reshape(
            bank_end - bank_begin, self.tile_rows, self.weight_bits,
            self.plane_bytes)
        codes = torch.zeros(
            (bank_end - bank_begin, self.tile_rows, self.columns),
            dtype=torch.int16, device=self.storage.device)
        for plane in range(self.weight_bits):
            bits = _unpack_plane_bits(
                planes[:, :, plane].reshape(-1, self.plane_bytes), self.columns)
            bits = bits.reshape(bank_end - bank_begin, self.tile_rows,
                                self.columns)
            codes |= bits.to(torch.int16) << plane
        return torch.where(codes >= 2 ** (self.weight_bits - 1),
                           codes - 2 ** self.weight_bits, codes)

    def _dequantize_tile(self, slot_begin, bank_begin, bank_end):
        codes = self._decode_tile_codes(slot_begin, bank_begin, bank_end)
        scales = self.group_scales[
            bank_begin:bank_end, slot_begin:slot_begin + self.tile_rows]
        values = (codes.float().reshape(
            bank_end - bank_begin, self.tile_rows, self.groups, self.group_size)
            * scales[:, :, :, None]).reshape(
                (bank_end - bank_begin) * self.tile_rows, self.columns)
        return values.to(torch.bfloat16)

    def _bank_fp32_tile(self, slot_begin, bank_begin, bank_end, activation):
        codes = self._decode_tile_codes(slot_begin, bank_begin, bank_end)
        activation = activation.float().reshape(1, self.groups, self.group_size)
        products = codes.float().reshape(
            bank_end - bank_begin, self.tile_rows, self.groups,
            self.group_size) * activation
        summed = products.sum(dim=3)
        scales = self.group_scales[
            bank_begin:bank_end, slot_begin:slot_begin + self.tile_rows]
        output = (summed * scales).sum(dim=2)
        bias = self.bias[
            bank_begin:bank_end, slot_begin:slot_begin + self.tile_rows]
        output = output + bias.float()
        return output.reshape(-1).to(torch.bfloat16)

    def gemv(self, activation, *, numeric_mode="fp32_bank"):
        """Execute one bank-resident row against a single BF16 activation."""
        if (not isinstance(activation, torch.Tensor)
                or activation.dtype != torch.bfloat16
                or activation.shape != (self.columns,)
                or activation.device != self.storage.device
                or not torch.isfinite(activation).all().item()):
            raise ValueError("one finite BF16 activation row on the layout device required")
        if numeric_mode not in {"fp32_bank", "bf16_reference"}:
            raise ValueError("numeric mode must be fp32_bank or bf16_reference")
        output = torch.zeros(self.padded_rows, dtype=torch.bfloat16,
                             device=activation.device)
        for row_round in range(self.rounds):
            slot_begin = row_round * self.tile_rows
            first = row_round * self.bank_count * self.tile_rows
            valid_tile_rows = min(
                self.active_banks * self.tile_rows, self.rows - first)
            if valid_tile_rows <= 0:
                break
            if numeric_mode == "fp32_bank":
                values = self._bank_fp32_tile(
                    slot_begin, 0, self.active_banks, activation)
            else:
                dequantized = self._dequantize_tile(
                    slot_begin, 0, self.active_banks)[:valid_tile_rows]
                bias = (None if not self.has_bias else
                        self.bias[:, slot_begin:slot_begin + self.tile_rows]
                        .reshape(-1)[:valid_tile_rows])
                values = torch.nn.functional.linear(
                    activation.unsqueeze(0), dequantized, bias).squeeze(0)
            valid = min(values.numel(), self.rows - first)
            if valid > 0:
                output[first:first + valid].copy_(values[:valid])
        result = output[:self.rows].contiguous()
        require_finite(result, "nonfinite W12 bank output")
        return result

    def linear(self, activation):
        """Explicit Host fallback used for prefill and multi-row calls.

        The fallback reads the packed bank rows, dequantizes bounded bank
        batches, and runs a BF16 matrix multiply.  It is counted by the
        dispatcher and must never be described as a silent PIM hit.
        """
        if (not isinstance(activation, torch.Tensor) or not activation.is_floating_point()
                or activation.shape[-1] != self.columns
                or activation.device != self.storage.device):
            raise ValueError("floating activation rows on the layout device required")
        original_shape = activation.shape
        rows = int(activation.numel() // self.columns)
        flat = activation.reshape(rows, self.columns).to(torch.bfloat16)
        local_bank_chunk = self.active_banks
        output = torch.empty((rows, self.rows), dtype=torch.bfloat16,
                             device=activation.device)
        for row_round in range(self.rounds):
            slot_begin = row_round * self.tile_rows
            for bank_begin in range(0, self.active_banks, local_bank_chunk):
                bank_end = min(bank_begin + local_bank_chunk, self.active_banks)
                dequantized = self._dequantize_tile(
                    slot_begin, bank_begin, bank_end)
                bias = (None if not self.has_bias else
                        self.bias[bank_begin:bank_end,
                                  slot_begin:slot_begin + self.tile_rows]
                        .reshape(-1))
                first = ((row_round * self.active_banks + bank_begin)
                         * self.tile_rows)
                valid = min(dequantized.shape[0], self.rows - first)
                if valid > 0:
                    values = torch.nn.functional.linear(
                        flat, dequantized[:valid],
                        None if bias is None else bias[:valid])
                    output[:, first:first + valid] = values[:, :valid]
        require_finite(output, "nonfinite W12 Host fallback output")
        return output.reshape(*original_shape[:-1], self.rows)
