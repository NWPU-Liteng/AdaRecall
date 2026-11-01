"""Query-scored local dialogue hyperedges with indivisible image/context unions."""
from __future__ import annotations

import math


def build_local_hyperedges(ranked_session_ids, atoms_by_session, radius=2):
    if radius < 1 or len(set(ranked_session_ids)) != len(ranked_session_ids):
        raise ValueError('positive locality radius and unique sessions required')
    edges, seen, identities = [], set(), set()
    for session_id in ranked_session_ids:
        turns = atoms_by_session.get(session_id, [])
        for atom in turns:
            if atom['session_id'] != session_id or atom['atom_id'] in identities:
                raise ValueError('invalid or duplicate observable atom identity')
            identities.add(atom['atom_id'])
        for center in range(len(turns)):
            members = turns[max(0, center - radius):center + radius + 1]
            identity = tuple(atom['atom_id'] for atom in members)
            if identity in seen:
                continue
            seen.add(identity)
            text = '\n'.join(f"[{atom['role']}] {atom['content']}" for atom in members
                             if atom.get('content_type') == 'text' and atom.get('content'))
            if text:
                edges.append({'edge_id': f'{session_id}:local:{center}',
                              'atom_ids': list(identity), 'embedding_text': text})
    return edges


def render_atoms(atoms):
    return '\n'.join(f"[{atom['session_id']}][{atom['atom_id']}][{atom['role']}] {atom['content']}"
                     for atom in sorted(atoms, key=lambda item: item['ordinal'])
                     if atom.get('content_type') == 'text' and atom.get('content'))


def images_for_atoms(atoms):
    refs, seen = [], set()
    for atom in sorted(atoms, key=lambda item: item['ordinal']):
        for ref in atom.get('image_refs', []):
            if ref.get('status') not in {'exact', 'unique_user_basename'}:
                raise ValueError('unresolved image in indivisible dialogue hyperedge')
            member = ref.get('resolved', ref['normalized'])
            if member not in seen:
                refs.append({**ref, 'source_session_id': atom['session_id'],
                             'source_atom_id': atom['atom_id'],
                             'selection_source': 'observable_local_hyperedge'})
                seen.add(member)
    return refs


def bound_hyperedges(edges, similarities, atoms_by_id, limit, image_edges_per_session=0):
    if limit < 1 or image_edges_per_session < 0 or len(edges) != len(similarities):
        raise ValueError('positive candidate limit, nonnegative image quota and matching scores required')
    if len({edge['edge_id'] for edge in edges}) != len(edges):
        raise ValueError('duplicate candidate hyperedge identity')
    for edge, score in zip(edges, similarities):
        if not math.isfinite(float(score)) or any(atom_id not in atoms_by_id for atom_id in edge['atom_ids']):
            raise ValueError('finite score and known observable members required')
    ranked = sorted(zip(edges, similarities), key=lambda item: item[1], reverse=True)

    def has_image(edge):
        return any(ref.get('status') in {'exact', 'unique_user_basename'}
                   for atom_id in edge['atom_ids'] for ref in atoms_by_id[atom_id].get('image_refs', []))

    bounded = ranked[:limit]
    if image_edges_per_session:
        session_images, required = {}, []
        for edge, score in ranked:
            sessions = {atoms_by_id[atom_id]['session_id'] for atom_id in edge['atom_ids']}
            if len(sessions) != 1:
                raise ValueError('reserved image quota requires session-local hyperedges')
            session_id = next(iter(sessions))
            known_images = session_images.setdefault(session_id, set())
            members = {ref.get('resolved', ref['normalized']) for atom_id in edge['atom_ids']
                       for ref in atoms_by_id[atom_id].get('image_refs', [])
                       if ref.get('status') in {'exact', 'unique_user_basename'}}
            if len(known_images) < image_edges_per_session and members - known_images:
                required.append((edge, score))
                known_images.update(members)
        required = sorted(required, key=lambda item: item[1], reverse=True)[:limit]
        required_ids = {edge['edge_id'] for edge, _ in required}
        bounded = (required + [item for item in ranked if item[0]['edge_id'] not in required_ids])[:limit]
        order = {edge['edge_id']: rank for rank, (edge, _) in enumerate(ranked)}
        bounded.sort(key=lambda item: order[item[0]['edge_id']])
    if bounded and not any(has_image(edge) for edge, _ in bounded):
        image_edge = next((item for item in ranked if has_image(item[0])), None)
        if image_edge is not None:
            bounded = bounded[:-1] + [image_edge]
    retained_ids = {edge['edge_id'] for edge, _ in bounded}
    return ([edge for edge, _ in bounded], [score for _, score in bounded],
            [edge['edge_id'] for edge, _ in ranked if edge['edge_id'] not in retained_ids])


def select_hyperedges(edges, similarities, atoms_by_id, tokenizer, *, text_budget,
                      max_images, require_image=True, image_novelty_weight=0.0,
                      atom_token_costs=None):
    if (text_budget <= 0 or max_images < 1 or len(edges) != len(similarities)
            or not math.isfinite(image_novelty_weight) or image_novelty_weight < 0):
        raise ValueError('positive budgets and one score per edge required')
    if len({edge['edge_id'] for edge in edges}) != len(edges):
        raise ValueError('duplicate hyperedge identity')
    candidates = []
    for edge, similarity in zip(edges, similarities):
        if not math.isfinite(float(similarity)):
            raise ValueError('finite query similarity required')
        if not edge['atom_ids'] or len(set(edge['atom_ids'])) != len(edge['atom_ids']):
            raise ValueError('nonempty unique hyperedge membership required')
        for atom_id in edge['atom_ids']:
            if atom_id not in atoms_by_id:
                raise ValueError('hyperedge references an unknown observable atom')
        image_ids = {ref.get('resolved', ref['normalized']) for atom_id in edge['atom_ids']
                     for ref in atoms_by_id[atom_id].get('image_refs', [])
                     if ref.get('status') in {'exact', 'unique_user_basename'}}
        candidates.append({**edge, 'score': max(0.0, float(similarity)) ** 2,
                           'image_ids': image_ids})
    selected, selected_ids, coverage, selected_edges = set(), set(), {}, []
    image_coverage = {}
    rejected = {}
    text_cost = 0
    image_refs = []
    union_states = {}
    union_evaluations = 0
    union_hits = 0

    def evaluate(edge):
        nonlocal union_evaluations, union_hits
        gain = sum(max(0.0, edge['score'] - coverage.get(atom_id, 0.0))
                   for atom_id in edge['atom_ids'])
        gain += image_novelty_weight * sum(max(0.0, edge['score'] - image_coverage.get(identity, 0.0))
                                          for identity in edge['image_ids'])
        if gain <= 0:
            return None, 'zero_marginal_gain'
        union = frozenset(selected | set(edge['atom_ids']))
        union_evaluations += 1
        if union in union_states:
            union_hits += 1
            images, cost = union_states[union]
        else:
            atoms = [atoms_by_id[atom_id] for atom_id in union]
            try:
                images = images_for_atoms(atoms)
            except ValueError:
                return None, 'unresolved_image'
            cost = None
        if len(images) > max_images:
            return None, 'image_budget'
        if require_image and not selected_edges and not images:
            return None, 'raw_image_seed_required'
        if cost is None:
            if atom_token_costs is None:
                rendered = render_atoms([atoms_by_id[atom_id] for atom_id in union])
                cost = len(tokenizer.encode(rendered, add_special_tokens=False))
            else:
                cost = atom_token_costs.count(union)
            union_states[union] = (images, cost)
        if cost > text_budget:
            return None, 'text_budget'
        incremental_cost = max(1, cost - text_cost) + 256 * max(0, len(images) - len(image_refs))
        return (gain / incremental_cost, gain, cost, images, union), None

    while True:
        options = []
        for rank, edge in enumerate(candidates):
            if edge['edge_id'] in selected_ids:
                continue
            evaluated, reason = evaluate(edge)
            if evaluated is None:
                rejected[edge['edge_id']] = reason
                continue
            density, gain, cost, images, union = evaluated
            if gain <= 0 or (require_image and not selected_edges and not images):
                continue
            options.append((density, gain, -rank, edge, cost, images, union))
        if not options:
            break
        _, gain, _, edge, text_cost, image_refs, selected = max(options, key=lambda item: item[:3])
        selected_ids.add(edge['edge_id'])
        rejected.pop(edge['edge_id'], None)
        selected_edges.append({'edge_id': edge['edge_id'], 'atom_ids': edge['atom_ids'],
                               'query_similarity_squared': edge['score'], 'marginal_gain': gain})
        for atom_id in edge['atom_ids']:
            coverage[atom_id] = max(edge['score'], coverage.get(atom_id, 0.0))
        for identity in edge['image_ids']:
            image_coverage[identity] = max(edge['score'], image_coverage.get(identity, 0.0))
    if require_image and not image_refs:
        raise ValueError('no feasible positive-score raw-image hyperedge')
    ordered = sorted((atoms_by_id[atom_id] for atom_id in selected), key=lambda atom: atom['ordinal'])
    return {'text': render_atoms(ordered), 'images': image_refs,
            'accepted_atom_ids': [atom['atom_id'] for atom in ordered],
            'selected_hyperedges': selected_edges, 'rejected_hyperedges': rejected,
            'text_tokens': text_cost, 'text_budget': text_budget, 'max_images': max_images,
            'objective': sum(coverage.values()) + image_novelty_weight * sum(image_coverage.values()),
            'atom_coverage_objective': sum(coverage.values()),
            'image_coverage_objective': sum(image_coverage.values()),
            'image_novelty_weight': image_novelty_weight,
            'query_local_union_state_reuse': {'evaluations': union_evaluations, 'hits': union_hits,
                'unique_unions': len(union_states),
                'scope': 'same query, immutable tokenizer, atom contents and exact atom-id union; no cross-query retention'},
            'policy': ('greedy marginal query-weighted atom coverage / incremental union cost; image seed required'
                       if image_novelty_weight == 0 else
                       'greedy marginal query-weighted atom and unique-image max coverage / incremental union cost; image seed required'),
            'boundary': 'local dialogue adjacency only, not factual causality; uncalibrated scores; no optimality guarantee; 256-token/image ranking charge is a heuristic, actual prompt is checked separately'}
