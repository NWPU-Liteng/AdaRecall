"""Match the host accumulation contract without changing prefill or opaque work."""
from __future__ import annotations

import torch

from pim_backend.functional_decode import DecodeBankDispatcher


class MatchedDecodeDispatcher(DecodeBankDispatcher):
    def _wrapper(self, module, original, identity, name):
        def matched_host(activation):
            setting = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            try:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                return original(activation)
            finally:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = setting

        dispatch = super()._wrapper(module, matched_host, identity, name)

        def forward(activation):
            if not self.active:
                return original(activation)
            return dispatch(activation)

        return forward
