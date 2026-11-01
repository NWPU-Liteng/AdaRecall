"""Mechanism-level accounting for safe prefix/state reuse."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict


class PrefixStateStore:
    """Model-contract-scoped immutable checkpoints, never cropped recurrent state."""

    def __init__(self, contract: str, *, capacity_bytes: int, minimum_tokens: int = 128):
        if not contract or capacity_bytes < 0 or minimum_tokens < 1:
            raise ValueError('invalid prefix store contract or budget')
        self.contract = contract
        self.capacity_bytes = capacity_bytes
        self.minimum_tokens = minimum_tokens
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0

    def lookup(self, input_ids, *, contract, maximum_tokens=None):
        if contract != self.contract:
            raise ValueError('prefix model/tokenizer/numeric contract mismatch')
        candidates = [(identity, snapshot) for identity, snapshot in self.entries.items()
                      if snapshot.compatible(input_ids) and snapshot.tokens < input_ids.shape[1]
                      and (maximum_tokens is None or snapshot.tokens <= maximum_tokens)]
        if not candidates:
            self.misses += 1
            return None
        identity, snapshot = max(candidates, key=lambda item: item[1].tokens)
        self.entries.move_to_end(identity)
        self.hits += 1
        return snapshot

    def admit(self, snapshot, *, contract):
        if contract != self.contract:
            raise ValueError('prefix admission contract mismatch')
        if snapshot.tokens < self.minimum_tokens or snapshot.bytes > self.capacity_bytes:
            return False
        identity = snapshot.prefix_fingerprint
        if identity in self.entries:
            self.bytes -= self.entries.pop(identity).bytes
        while self.entries and self.bytes + snapshot.bytes > self.capacity_bytes:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.bytes
            self.evictions += 1
        self.entries[identity] = snapshot
        self.bytes += snapshot.bytes
        return True

    def invalidate(self):
        self.entries.clear()
        self.bytes = 0

    def discard(self, fingerprint):
        removed = self.entries.pop(fingerprint, None)
        if removed is not None:
            self.bytes -= removed.bytes

    def metrics(self):
        return {'entries': len(self.entries), 'tensor_bytes': self.bytes,
                'capacity_bytes': self.capacity_bytes, 'hits': self.hits,
                'misses': self.misses, 'evictions': self.evictions}


@dataclass(frozen=True)
class ReuseDecision:
    status: str
    safe: bool
    reused_tokens: int
    snapshot_seconds: float
    restore_seconds: float
    reused_prefill_seconds: float
    full_prefill_seconds: float | None
    net_prefill_seconds: float | None
    reason: str

    def as_dict(self) -> dict:
        return {
            'status': self.status, 'safe': self.safe, 'reused_tokens': self.reused_tokens,
            'snapshot_seconds': self.snapshot_seconds, 'restore_seconds': self.restore_seconds,
            'reused_prefill_seconds': self.reused_prefill_seconds,
            'full_prefill_seconds': self.full_prefill_seconds,
            'net_prefill_seconds': self.net_prefill_seconds, 'reason': self.reason,
            'accounting': 'full_prefill - stage_prefill - snapshot_if_created_outside_stage; restore is already in stage',
            'verification_scope': 'full greedy token and stop-reason equivalence when audit protocol is enabled',
        }


def account_prefix_reuse(snapshot: dict, validation: dict, stages: dict) -> ReuseDecision:
    status = snapshot.get('status', 'missing')
    reused = int(snapshot.get('reused_tokens', 0) or 0)
    snapshot_seconds = float(snapshot.get('snapshot_seconds', 0.0) or 0.0)
    restore_seconds = float(snapshot.get('restore_seconds', 0.0) or 0.0)
    single_pass = stages.get('proof', {}).get('stop_reason') == 'not_executed'
    stage_name = 'first' if single_pass else 'proof'
    reused_prefill = float(stages.get(stage_name, {}).get('prefill_seconds', 0.0) or 0.0)
    full_prefill = validation.get('reference_full_prefill_seconds')
    full_prefill = float(full_prefill) if full_prefill is not None else None
    safe = status == 'exact_token_prefix_restored' and validation.get('status') == 'pass'
    net = None
    if full_prefill is not None and validation.get('status') != 'fail':
        net = full_prefill - reused_prefill - (0.0 if single_pass else snapshot_seconds)
    if not safe:
        reason = 'unverified_or_fallback'
    elif net is None:
        reason = 'no_full_prefill_reference'
    elif net <= 0:
        reason = 'reuse_overhead_exceeds_prefill_saving'
    else:
        reason = 'safe_and_net_positive'
    return ReuseDecision(status, safe, reused, snapshot_seconds, restore_seconds,
                         reused_prefill, full_prefill, net, reason)
