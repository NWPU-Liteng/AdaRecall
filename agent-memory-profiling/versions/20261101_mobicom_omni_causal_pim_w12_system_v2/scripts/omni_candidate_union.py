"""Deterministic union of independently retrieved modalities, without score fusion."""


def candidate_union(visual_edges, visual_scores, text_edges, text_scores, text_limit):
    if (len(visual_edges) != len(visual_scores) or len(text_edges) != len(text_scores)
            or text_limit < 1):
        raise ValueError('matching candidates/scores and positive text limit required')
    import math

    if any(not math.isfinite(float(score)) for score in [*visual_scores, *text_scores]):
        raise ValueError('finite model scores required')
    if any(len({edge['edge_id'] for edge in edges}) != len(edges)
           for edges in (visual_edges, text_edges)):
        raise ValueError('duplicate candidates within retrieval branch')
    order = sorted(range(len(text_edges)), key=lambda index: (
        -float(text_scores[index]), text_edges[index]['edge_id']))[:text_limit]
    unique, records = {}, []
    for branch, edges, scores, indices, semantics in (
            ('visual', visual_edges, visual_scores, range(len(visual_edges)), 'joint_embedding_cosine'),
            ('text', text_edges, text_scores, order, 'text_reranker_logit_difference')):
        for index in indices:
            edge = edges[index]
            if edge['edge_id'] in unique and unique[edge['edge_id']]['atom_ids'] != edge['atom_ids']:
                raise ValueError('one producer identity refers to different provenance')
            unique.setdefault(edge['edge_id'], edge)
            records.append({'edge_id': edge['edge_id'], 'branch': branch,
                            'score': float(scores[index]), 'score_semantics': semantics})
    return list(unique.values()), records
