"""Past-only exact input-prefix opportunities, not authorization for KV reuse."""
from __future__ import annotations


class PrefixOpportunityIndex:
    def __init__(self):
        self.root = {}
        self.identities = set()

    def observe(self, request_id, symbols):
        if not symbols or request_id in self.identities:
            raise ValueError('nonempty unique natural request required')
        node, matched, prior = self.root, 0, None
        for symbol in symbols:
            if symbol not in node:
                break
            node, prior = node[symbol]
            matched += 1
        result = {'matched_prefix_tokens': matched, 'prior_request_id': prior,
                  'prefix_state_reuse_authorized': False}
        node = self.root
        for symbol in symbols:
            if symbol not in node:
                node[symbol] = ({}, request_id)
            node = node[symbol][0]
        self.identities.add(request_id)
        return result


def multimodal_prefix_symbols(token_ids, image_token_id, image_token_counts, batch_identity):
    if not batch_identity or any(count <= 0 for count in image_token_counts):
        raise ValueError('pinned ordered batch identity and positive image token counts required')
    symbols, image_index, offset = [], 0, 0
    for token in token_ids:
        if token != image_token_id:
            symbols.append(token)
            continue
        if image_index >= len(image_token_counts):
            raise ValueError('image placeholders exceed actual processed image sizes')
        symbols.append(('image', token, batch_identity, image_index, offset))
        offset += 1
        if offset == image_token_counts[image_index]:
            image_index += 1
            offset = 0
    if image_index != len(image_token_counts) or offset:
        raise ValueError('image placeholders differ from actual processed image sizes')
    return symbols
