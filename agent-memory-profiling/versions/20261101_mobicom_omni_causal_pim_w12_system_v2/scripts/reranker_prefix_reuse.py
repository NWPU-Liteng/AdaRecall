"""Fork immutable model-native reranker prefix state across document branches."""
from __future__ import annotations

import copy
import hashlib
import time

import torch

from intact_bundle_scorer import token_batches
from joint_bundle_scorer import scorer_inputs
from source_closed_scorer import PROBABILITY_TOLERANCE, SourceClosedScorer


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def shared_prefix_scores(model, encoded, *, device, batch_size, pad_token_id, yes_id, no_id):
    if not encoded or any(not tokens for tokens in encoded):
        raise ValueError('nonempty scoring token sequences required')
    common = 0
    for values in zip(*encoded):
        if len(set(values)) != 1:
            break
        common += 1
    prefix_tokens = min(common, min(map(len, encoded)) - 1)
    if prefix_tokens < 1:
        raise ValueError('no reusable exact token prefix')
    lengths = [len(tokens) for tokens in encoded]
    batches = token_batches(lengths, batch_size)
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prefix = torch.tensor([encoded[0][:prefix_tokens]], dtype=torch.long, device=device)
        output = model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                       use_cache=True, logits_to_keep=1)
        snapshot = output.past_key_values
        del output
        snapshot_bytes = sum(tensor.numel() * tensor.element_size()
                             for layer in snapshot.layers for name in ('keys', 'values')
                             if torch.is_tensor(tensor := getattr(layer, name, None)))
        synchronize(device)
        prefill_seconds = time.perf_counter() - started
        fork_seconds, padded_suffix_tokens = 0.0, 0
        probabilities = [0.0] * len(encoded)
        for indices in batches:
            width = max(lengths[index] - prefix_tokens for index in indices)
            suffix = torch.full((len(indices), width), pad_token_id, dtype=torch.long, device=device)
            suffix_mask = torch.zeros_like(suffix)
            for batch_index, document_index in enumerate(indices):
                tokens = encoded[document_index][prefix_tokens:]
                suffix[batch_index, -len(tokens):] = torch.tensor(tokens, device=device)
                suffix_mask[batch_index, -len(tokens):] = 1
            mask = torch.cat((torch.ones((len(indices), prefix_tokens), dtype=torch.long, device=device),
                              suffix_mask), dim=1)
            positions = mask.cumsum(-1)[:, -width:] - 1
            synchronize(device)
            fork_started = time.perf_counter()
            branch = copy.deepcopy(snapshot)
            branch.batch_repeat_interleave(len(indices))
            synchronize(device)
            fork_seconds += time.perf_counter() - fork_started
            output = model(input_ids=suffix, attention_mask=mask, position_ids=positions,
                           past_key_values=branch, use_cache=True, logits_to_keep=1)
            logits = output.logits[:, -1, :]
            pairs = torch.stack((logits[:, no_id], logits[:, yes_id]), dim=-1)
            for index, probability in zip(indices, pairs.softmax(-1)[:, 1].tolist()):
                probabilities[index] = float(probability)
            padded_suffix_tokens += len(indices) * width
            del output, branch
        if snapshot.get_seq_length() != prefix_tokens:
            raise ValueError('branch mutated immutable prefix snapshot')
        del snapshot
    synchronize(device)
    return probabilities, {
        'prefix_tokens': prefix_tokens, 'prefix_state_bytes': snapshot_bytes,
        'prefix_prefill_seconds': prefill_seconds, 'fork_seconds': fork_seconds,
        'model_seconds': time.perf_counter() - started,
        'avoided_prefix_token_evaluations': (len(encoded) - 1) * prefix_tokens,
        'reference_padded_tokens': sum(len(indices) * max(lengths[index] for index in indices)
                                       for indices in batches),
        'shared_path_padded_tokens': prefix_tokens + padded_suffix_tokens,
        'branches': len(encoded), 'batches': len(batches),
        'identity': 'exact token longest-common-prefix; same live model/dtype; no cross-query state retention',
        'boundary': 'mathematically compatible prefix forks; numerical and latency behavior require reference probe'}


class SharedPrefixScorer(SourceClosedScorer):
    def score(self, question, documents):
        if len(documents) < 2:
            scores, trace = super().score(question, documents)
            trace['state_reuse'] = {'status': 'single_branch_full_compute'}
            return scores, trace
        synchronize(self.device)
        started = time.perf_counter()
        inputs = scorer_inputs(question, documents)
        encoded = self.tokenizer(inputs, add_special_tokens=False, truncation=False, padding=False)['input_ids']
        lengths = list(map(len, encoded))
        if max(lengths) > self.max_length:
            raise ValueError('shared-prefix scorer refuses truncation')
        scores, reuse = shared_prefix_scores(
            self.model, encoded, device=self.device, batch_size=self.batch_size,
            pad_token_id=self.tokenizer.pad_token_id, yes_id=self.yes_id, no_id=self.no_id)
        synchronize(self.device)
        candidate_seconds = time.perf_counter() - started
        reference_started = time.perf_counter()
        reference, _ = super().score(question, documents)
        synchronize(self.device)
        reference_seconds = time.perf_counter() - reference_started
        max_error = max(abs(actual - expected) for actual, expected in zip(scores, reference))
        candidate_order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        reference_order = sorted(range(len(reference)), key=lambda index: (-reference[index], index))
        compatible = max_error <= PROBABILITY_TOLERANCE and candidate_order == reference_order
        reuse['candidate_status'] = reuse.get('status')
        reuse['numeric_validation'] = {
            'status': 'pass' if compatible else 'fail', 'max_probability_error': max_error,
            'tolerance': PROBABILITY_TOLERANCE, 'full_ranking_equal': candidate_order == reference_order,
            'reference_seconds': reference_seconds, 'candidate_seconds': candidate_seconds,
            'fallback_to_full_scores': not compatible,
        }
        if not compatible:
            scores = reference
            reuse['status'] = 'numeric_contract_full_recompute'
            reuse['submitted_reused_prefix_tokens'] = 0
        else:
            reuse['submitted_reused_prefix_tokens'] = reuse.get('prefix_tokens', 0)
        return scores, {'seconds': time.perf_counter() - started,
                        'input_tokens': lengths, 'max_input_tokens': max(lengths),
                        'inputs_sha256': [hashlib.sha256(text.encode()).hexdigest() for text in inputs],
                        'documents_sha256': [hashlib.sha256(text.encode()).hexdigest() for text in documents],
                        'truncated_documents': 0, 'provenance': self.provenance,
                        'state_reuse': reuse}
