"""Query-local exact token costs for a pinned newline-separated rendering contract."""
from __future__ import annotations

import json
import sys


QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
              r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
QWEN_LEGACY_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
                     r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def supports_fragment_costs(tokenizer):
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None or type(backend.model).__name__ != 'BPE' or backend.model.dropout is not None:
        return False
    if backend.normalizer is None or json.loads(backend.normalizer.__getstate__()) != {'type': 'NFC'}:
        return False
    expected = {'type': 'Sequence', 'pretokenizers': [
        {'type': 'Split', 'pattern': {'Regex': QWEN_SPLIT}, 'behavior': 'Isolated', 'invert': False},
        {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': True, 'use_regex': False}]}
    if backend.pre_tokenizer is None:
        return False
    observed = json.loads(backend.pre_tokenizer.__getstate__())
    if observed.get('type') == 'Sequence' and len(observed.get('pretokenizers', [])) == 2:
        pattern = observed['pretokenizers'][0].get('pattern', {}).get('Regex')
        if pattern not in (QWEN_SPLIT, QWEN_LEGACY_SPLIT):
            return False
        expected['pretokenizers'][0]['pattern']['Regex'] = pattern
        byte_level = observed['pretokenizers'][1]
        if type(byte_level.get('trim_offsets')) is bool:
            byte_level['trim_offsets'] = True
    if observed != expected:
        return False
    if backend.post_processor is not None:
        state = json.loads(backend.post_processor.__getstate__())
        if state.get('type') != 'ByteLevel':
            return False
    for added in backend.get_added_tokens_decoder().values():
        if '\n' in added.content or '\r' in added.content or added.lstrip or added.rstrip:
            return False
    return True


class AtomUnionTokenCosts:
    def __init__(self, tokenizer, atoms_by_id):
        self.tokenizer = tokenizer
        self.supported = supports_fragment_costs(tokenizer)
        self.atoms = {identity: dict(atom) for identity, atom in atoms_by_id.items()}
        self.costs = {}
        self.requests = 0
        self.fragment_producer_calls = 0
        self.fragment_hits = 0
        self.fallback_calls = 0

    def count(self, identities):
        from omni_evidence_planner import render_atoms
        self.requests += 1
        ordered = sorted((self.atoms[identity] for identity in identities
                          if self.atoms[identity].get('content_type') == 'text'
                          and self.atoms[identity].get('content')), key=lambda atom: atom['ordinal'])
        if not self.supported:
            self.fallback_calls += 1
            return len(self.tokenizer.encode(render_atoms(ordered), add_special_tokens=False))
        total = 0
        for index, atom in enumerate(ordered):
            followed = index + 1 < len(ordered)
            key = (atom['atom_id'], followed)
            if key not in self.costs:
                text = render_atoms([atom]) + ('\n' if followed else '')
                self.costs[key] = len(self.tokenizer.encode(text, add_special_tokens=False))
                self.fragment_producer_calls += 1
            else:
                self.fragment_hits += 1
            total += self.costs[key]
        return total

    def summary(self):
        return {'supported_tokenizer_contract': self.supported, 'union_cost_requests': self.requests,
                'fragment_producer_calls': self.fragment_producer_calls,
                'fragment_state_hits': self.fragment_hits, 'fallback_calls': self.fallback_calls,
                'resident_cost_entries': len(self.costs),
                'cost_dictionary_python_bytes': sys.getsizeof(self.costs) + sum(
                    sys.getsizeof(key) + sys.getsizeof(value) for key, value in self.costs.items()),
                'atom_snapshot_dictionary_python_bytes': sys.getsizeof(self.atoms) + sum(
                    sys.getsizeof(atom) for atom in self.atoms.values()),
                'scope': 'query-local immutable atom rendering, plain/followed-by-newline costs; not Reader KV or a cross-query cache',
                'boundary': 'NFC and the pinned Split+ByteLevel tokenizer only; newline followed by the next atom opening bracket is a pretokenization boundary; no token prefix/suffix substitution; unsupported tokenizers use the original full rendering'}
