"""Shadow-only progressive top-1 validation for the Reader lm_head."""
from __future__ import annotations

import copy
import resource
import time

import numpy as np
import torch

from .progressive_topk import (from_codes, progressive_topk_torch,
                               reference_scores, to_torch_matrix)
from .w12_bank_layout import W12BankLayout


PHASES = ('contract_parse', 'witness_construct', 'answer')


def _empty_phase_stats():
    return {
        'checked_tokens': 0,
        'forced_tokens': 0,
        'progressive_calls': 0,
        'weight_bytes_read': 0,
        'weight_bytes_full': 0,
        'rounds_total': 0,
        'agree_bf16': 0,
        'disagree_bf16': 0,
    }


def _finish_stats(stats):
    result = dict(stats)
    calls = result['progressive_calls']
    observations = result['agree_bf16'] + result['disagree_bf16']
    result['average_rounds'] = (
        result.pop('rounds_total') / calls if calls else None)
    result['bf16_agreement_rate'] = (
        result['agree_bf16'] / observations if observations else None)
    result['weight_read_fraction'] = (
        result['weight_bytes_read'] / result['weight_bytes_full']
        if result['weight_bytes_full'] else None)
    return result


def _host_peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class ProgressiveLmHeadShadow:
    """Capture the lm_head input and validate scheduled decode decisions."""

    def __init__(self, lm_head, *, vocab_rows, valid_vocab_size,
                 shadow_every=8, plane_schedule=None, eps_rel=1.0e-9,
                 device=None, chunk_rows=4096):
        layout = getattr(lm_head, 'layout', None)
        if not isinstance(layout, W12BankLayout):
            raise ValueError('lm_head shadow requires a W12 bank layout')
        if (isinstance(vocab_rows, bool) or not isinstance(vocab_rows, int)
                or vocab_rows < 1 or vocab_rows != layout.rows):
            raise ValueError('vocab_rows must equal the lm_head layout rows')
        if (isinstance(valid_vocab_size, bool)
                or not isinstance(valid_vocab_size, int)
                or not 0 < valid_vocab_size <= vocab_rows):
            raise ValueError('valid_vocab_size must be within the vocab rows')
        if (isinstance(shadow_every, bool) or not isinstance(shadow_every, int)
                or shadow_every < 1):
            raise ValueError('positive lm_head shadow interval required')
        if (isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int)
                or chunk_rows < 1):
            raise ValueError('positive row chunk required')
        if (isinstance(eps_rel, bool) or not isinstance(eps_rel, (int, float))
                or not np.isfinite(float(eps_rel)) or eps_rel < 0):
            raise ValueError('nonnegative finite eps_rel required')

        self.lm_head = lm_head
        self.layout = layout
        self.vocab_rows = vocab_rows
        self.valid_vocab_size = valid_vocab_size
        self.shadow_every = shadow_every
        self.plane_schedule = (tuple(plane_schedule) if plane_schedule is not None
                               else (1,) * layout.weight_bits)
        self.eps_rel = float(eps_rel)
        self.device = torch.device(device or layout.storage.device)
        self.chunk_rows = chunk_rows
        self.matrix = None
        self.torch_matrix = None
        self._hook = None
        self._last_hidden = None
        self._full_allowed = None
        self._stats = {phase: _empty_phase_stats() for phase in PHASES}
        self.build = None

    @property
    def enabled(self):
        return self._hook is not None

    def __enter__(self):
        if self._hook is not None:
            raise ValueError('lm_head shadow cannot be entered twice')
        started = time.perf_counter()
        cuda_device = (self.device.type == 'cuda')
        if cuda_device:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        codes, scales = self.layout.export_codes_and_scales()
        codes_cpu = codes.detach().to('cpu')
        scales_cpu = scales.detach().to('cpu')
        del codes, scales
        self.matrix = from_codes(
            codes_cpu.numpy(), scales_cpu.numpy(), self.layout.weight_bits,
            self.layout.group_size)
        del codes_cpu, scales_cpu
        self.torch_matrix = to_torch_matrix(self.matrix, device=self.device)
        self._full_allowed = torch.ones(
            self.vocab_rows, dtype=torch.bool, device=self.device)
        self._full_allowed[self.valid_vocab_size:] = False
        self._hook = self.lm_head.register_forward_pre_hook(self._capture)
        if cuda_device:
            torch.cuda.synchronize(self.device)
            cuda_peak = int(torch.cuda.max_memory_allocated(self.device))
        else:
            cuda_peak = None
        self.build = {
            'seconds': time.perf_counter() - started,
            'host_peak_rss_bytes': _host_peak_rss_bytes(),
            'cuda_peak_allocated_bytes': cuda_peak,
            'plane_bytes': int(self.matrix.planes.nbytes),
            'scale_bytes': int(self.matrix.scales.nbytes),
            'codes_sha256': self.matrix.codes_sha256,
            'device': str(self.device),
            'chunk_rows': self.chunk_rows,
        }
        return self

    def _capture(self, module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError('lm_head shadow did not receive a tensor input')
        value = inputs[0]
        if value.shape[-1] != self.layout.columns:
            raise RuntimeError('lm_head shadow input width changed')
        self._last_hidden = value.detach()

    def full_allowed(self):
        if self._full_allowed is None:
            raise RuntimeError('lm_head shadow must be entered first')
        return self._full_allowed

    def allowed_from_bitmask(self, bitmask):
        """Unpack one xgrammar mask and mask tokenizer padding rows."""
        if torch.is_tensor(bitmask):
            values = bitmask.detach().cpu().numpy()
        else:
            values = np.asarray(bitmask)
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError('one xgrammar token-bitmask row required')
        words = np.ascontiguousarray(values[0], dtype=np.uint32)
        bits = np.unpackbits(words.view(np.uint8), bitorder='little')
        if bits.size < self.vocab_rows:
            raise ValueError('xgrammar token bitmask is shorter than the vocabulary')
        allowed = np.zeros(self.vocab_rows, dtype=bool)
        allowed[:self.valid_vocab_size] = bits[:self.valid_vocab_size]
        return torch.from_numpy(allowed).to(self.device)

    def should_validate(self, position, forced):
        return bool(forced or position % self.shadow_every == 0)

    @staticmethod
    def _deterministic_argmax(scores, allowed):
        indices = torch.nonzero(allowed, as_tuple=False).flatten()
        order = torch.argsort(scores, descending=True, stable=True)
        return indices[order[0]]

    def observe(self, *, phase, actual_token, allowed, forced):
        if not self.enabled:
            raise RuntimeError('lm_head shadow must be entered first')
        if phase not in PHASES:
            raise ValueError(f'unknown lm_head shadow phase: {phase}')
        if isinstance(actual_token, bool) or not isinstance(actual_token, int):
            raise ValueError('actual token must be an integer')
        if not torch.is_tensor(allowed):
            allowed = torch.as_tensor(allowed, device=self.device)
        allowed = allowed.to(device=self.device, dtype=torch.bool)
        if allowed.shape != (self.vocab_rows,):
            raise ValueError('allowed mask must match the lm_head vocabulary rows')
        allowed_count = int(allowed.sum().item())
        if allowed_count < 1:
            raise ValueError('at least one token must be allowed')
        if bool(forced) != (allowed_count == 1):
            raise ValueError('forced flag must match a singleton allowed set')

        stats = self._stats[phase]
        stats['checked_tokens'] += 1
        if forced:
            stats['forced_tokens'] += 1
            return
        if self._last_hidden is None:
            raise RuntimeError('lm_head shadow has no captured hidden state')

        activation = self._last_hidden[..., -1, :].reshape(-1)
        indices, scores, trial = progressive_topk_torch(
            self.torch_matrix, activation, 1, row_mask=allowed,
            plane_schedule=self.plane_schedule, eps_rel=self.eps_rel,
            chunk_rows=self.chunk_rows)
        reference = reference_scores(
            self.torch_matrix, activation, rows=allowed)
        reference_index = self._deterministic_argmax(reference, allowed)
        if int(indices[0].item()) != int(reference_index.item()):
            raise AssertionError(
                'progressive top-1 disagrees with the exact reference')
        if not torch.equal(scores[0], reference.max()):
            raise AssertionError(
                'progressive top-1 score disagrees with the exact reference')

        progressive_token = int(indices[0].item())
        stats['progressive_calls'] += 1
        stats['weight_bytes_read'] += int(trial['weight_bytes_read'])
        stats['weight_bytes_full'] += int(trial['weight_bytes_full'])
        stats['rounds_total'] += len(trial['rounds'])
        if progressive_token == actual_token:
            stats['agree_bf16'] += 1
        else:
            stats['disagree_bf16'] += 1

    def summary(self):
        phases = {
            phase: _finish_stats(copy.deepcopy(self._stats[phase]))
            for phase in PHASES
        }
        raw_totals = _empty_phase_stats()
        for phase in PHASES:
            for key in raw_totals:
                raw_totals[key] += self._stats[phase][key]
        totals = _finish_stats(raw_totals)
        return {
            'enabled': self.enabled,
            'mode': 'shadow',
            'shadow_every': self.shadow_every,
            'vocab_rows': self.vocab_rows,
            'valid_vocab_size': self.valid_vocab_size,
            'bits': self.layout.weight_bits,
            'group_size': self.layout.group_size,
            'plane_schedule': list(self.plane_schedule),
            'codes_sha256': (self.matrix.codes_sha256
                             if self.matrix is not None else None),
            'build': self.build,
            'phases': phases,
            'totals': totals,
            'physical_pim_executed': False,
            'performance_comparable': False,
            'boundary': ('shadow comparison only; the measured logits and answer '
                         'remain on the normal Reader path'),
        }

    @classmethod
    def disabled_summary(cls):
        phases = {
            phase: _finish_stats(_empty_phase_stats()) for phase in PHASES
        }
        return {
            'enabled': False,
            'mode': 'off',
            'phases': phases,
            'totals': _finish_stats(_empty_phase_stats()),
            'physical_pim_executed': False,
            'performance_comparable': None,
        }

    def __exit__(self, *unused):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._last_hidden = None
        self._full_allowed = None
        self.torch_matrix = None
        self.matrix = None
