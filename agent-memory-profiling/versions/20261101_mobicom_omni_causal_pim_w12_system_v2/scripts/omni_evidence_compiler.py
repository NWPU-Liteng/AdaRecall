"""Exact budgeted semantic coverage over frozen multimodal exchange vectors."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from omni_evidence_planner import images_for_atoms, render_atoms


def compile_evidence(edges, vectors, query, atoms_by_id, tokenizer, image_token_costs, *,
                     text_budget, max_images, input_budget, fixed_prompt_tokens,
                     time_limit=5.0):
    started = time.perf_counter()
    vectors = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    count = len(edges)
    if (not count or vectors.ndim != 2 or vectors.shape[0] != count
            or query.shape != (vectors.shape[1],) or not np.isfinite(vectors).all()
            or not np.isfinite(query).all() or min(text_budget, input_budget, time_limit) <= 0
            or max_images < 0 or fixed_prompt_tokens < 0):
        raise ValueError('finite aligned vectors and positive evidence budgets required')
    if (len({edge['edge_id'] for edge in edges}) != count
            or any(not edge['atom_ids'] or len(set(edge['atom_ids'])) != len(edge['atom_ids'])
                   or any(identity not in atoms_by_id for identity in edge['atom_ids']) for edge in edges)):
        raise ValueError('unique nonempty observable exchanges required')
    if (not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4)
            or not np.isclose(np.linalg.norm(query), 1., atol=1e-4)):
        raise ValueError('one normalized shared embedding space required')
    relevance = np.maximum(vectors @ query, 0.) ** 2
    affinity = np.clip(vectors @ vectors.T, 0., 1.)
    np.fill_diagonal(affinity, 1.)
    atom_ids = sorted({identity for edge in edges for identity in edge['atom_ids']},
                      key=lambda identity: atoms_by_id[identity]['ordinal'])
    refs = images_for_atoms([atoms_by_id[identity] for identity in atom_ids])
    image_ids = [ref.get('resolved', ref['normalized']) for ref in refs]
    if any(identity not in image_token_costs or image_token_costs[identity] < 1 for identity in image_ids):
        raise ValueError('positive actual processor-grid costs required')
    atom_index = {identity: count + offset for offset, identity in enumerate(atom_ids)}
    image_offset = count + len(atom_ids)
    image_index = {identity: image_offset + offset for offset, identity in enumerate(image_ids)}
    assignment_offset = image_offset + len(image_ids)
    variable_count = assignment_offset + count * count
    constraints = []

    def constrain(coefficients, lower=-np.inf, upper=np.inf):
        constraints.append((coefficients, lower, upper))

    for identity in atom_ids:
        parents = [index for index, edge in enumerate(edges) if identity in edge['atom_ids']]
        for parent in parents:
            constrain({parent: 1., atom_index[identity]: -1.}, upper=0.)
        constrain({atom_index[identity]: 1., **{parent: -1. for parent in parents}}, upper=0.)
    for identity in image_ids:
        parents = [atom_index[atom_id] for atom_id in atom_ids
                   if any(ref.get('resolved', ref['normalized']) == identity
                          for ref in atoms_by_id[atom_id].get('image_refs', []))]
        for parent in parents:
            constrain({parent: 1., image_index[identity]: -1.}, upper=0.)
        constrain({image_index[identity]: 1., **{parent: -1. for parent in parents}}, upper=0.)
    text_costs = {atom_index[identity]: len(tokenizer.encode(
        render_atoms([atoms_by_id[identity]]) + '\n', add_special_tokens=False)) for identity in atom_ids}
    constrain(text_costs, upper=text_budget)
    constrain({position: 1. for position in image_index.values()}, upper=max_images)
    constrain({**text_costs, **{image_index[identity]: image_token_costs[identity] for identity in image_ids}},
              upper=input_budget - fixed_prompt_tokens)
    objective = np.zeros(variable_count)
    for demand in range(count):
        assignments = {}
        for supplier in range(count):
            position = assignment_offset + demand * count + supplier
            objective[position] = -relevance[demand] * affinity[demand, supplier]
            constrain({position: 1., supplier: -1.}, upper=0.)
            assignments[position] = 1.
        constrain(assignments, upper=1.)
    integrality = np.zeros(variable_count)
    integrality[:count] = 1
    cuts = []
    while True:
        remaining = time_limit - (time.perf_counter() - started)
        if remaining <= 0:
            raise RuntimeError('evidence compilation timed out before a feasible rendered input')
        matrix = lil_matrix((len(constraints), variable_count), dtype=np.float64)
        for row_index, (coefficients, _, _) in enumerate(constraints):
            for column_index, value in coefficients.items():
                matrix[row_index, column_index] = value
        solution = milp(objective, integrality=integrality,
                        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
                        constraints=LinearConstraint(matrix.tocsc(),
                            [item[1] for item in constraints], [item[2] for item in constraints]),
                        options={'time_limit': remaining, 'mip_rel_gap': 0.0})
        if solution.x is None:
            raise RuntimeError('no evidence-plan incumbent: ' + solution.message)
        if np.max(np.abs(solution.x[:count] - np.round(solution.x[:count]))) > 1e-5:
            raise RuntimeError('nonintegral evidence selection')
        chosen = [index for index in range(count) if solution.x[index] > .5]
        accepted = {identity for index in chosen for identity in edges[index]['atom_ids']}
        selected = [atoms_by_id[identity] for identity in atom_ids if identity in accepted]
        text = render_atoms(selected)
        text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        selected_refs = images_for_atoms(selected)
        visual_tokens = sum(image_token_costs[ref.get('resolved', ref['normalized'])] for ref in selected_refs)
        if text_tokens <= text_budget and text_tokens + visual_tokens + fixed_prompt_tokens <= input_budget:
            break
        if not chosen:
            raise RuntimeError('fixed prompt exceeds Reader budget')
        cuts.append([edges[index]['edge_id'] for index in chosen])
        constrain({index: 1. for index in chosen}, upper=len(chosen) - 1)
    coverage = affinity[:, chosen].max(axis=1) if chosen else np.zeros(count)
    return {'text': text, 'images': selected_refs,
            'accepted_atom_ids': [atom['atom_id'] for atom in selected],
            'selected_hyperedges': [{'edge_id': edges[index]['edge_id'], 'atom_ids': edges[index]['atom_ids'],
                                    'query_similarity_squared': float(relevance[index])} for index in chosen],
            'text_tokens': text_tokens, 'visual_tokens_from_processor_grid': visual_tokens,
            'text_budget': text_budget, 'max_images': max_images,
            'objective': float(relevance @ coverage),
            'solver_status': int(solution.status), 'mip_gap': float(solution.mip_gap),
            'solver_message': str(solution.message), 'render_feasibility_cuts': cuts,
            'planning_seconds': time.perf_counter() - started,
            'candidate_hyperedge_count': count,
            'candidate_replay': [{'edge_id': edge['edge_id'], 'atom_ids': edge['atom_ids'],
                                  'similarity': float(vectors[index] @ query)} for index, edge in enumerate(edges)],
            'planner': 'query-weighted facility-location MILP with exact provenance union and processor budget',
            'policy': 'evidence_compiler_v1',
            'boundary': 'semantic coverage surrogate, not factual sufficiency or a new solver; exact only at status=0; no labels, image quota, fitted threshold or hand-written query rule'}
