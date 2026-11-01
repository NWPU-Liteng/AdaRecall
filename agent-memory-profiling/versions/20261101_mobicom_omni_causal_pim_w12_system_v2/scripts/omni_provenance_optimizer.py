"""Label-free provenance closure with frozen relevance odds and exact MILP budgets."""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from omni_evidence_planner import images_for_atoms, render_atoms


def build_provenance_hyperedges(session_ids, atoms_by_session):
    edges = []
    for session_id in session_ids:
        turns = sorted(atoms_by_session[session_id], key=lambda atom: atom['ordinal'])
        exchanges = []
        current = []
        response_seen = False
        for atom in turns:
            if atom['role'] == 'user' and response_seen:
                exchanges.append(current)
                current = []
                response_seen = False
            current.append(atom)
            response_seen = response_seen or atom['role'] == 'assistant'
        if current:
            exchanges.append(current)
        for exchange in exchanges:
            text = render_atoms(exchange)
            if not text.strip() and not any(atom.get('image_refs') for atom in exchange):
                continue
            identity = hashlib.sha256('\0'.join(atom['atom_id'] for atom in exchange).encode()).hexdigest()
            edges.append({'edge_id': 'exchange:' + identity,
                          'atom_ids': [atom['atom_id'] for atom in exchange],
                          'embedding_text': text,
                          'session_id': session_id,
                          'event_ids': sorted({str(atom['event_id']) for atom in exchange}),
                          'provenance_relation': 'complete_observed_user_assistant_exchange'})
    return edges


def optimize_provenance(edges, log_odds, atoms_by_id, tokenizer, image_token_costs, *,
                        text_budget, max_images, input_budget, fixed_prompt_tokens,
                        time_limit=5.0):
    if (len(edges) != len(log_odds) or not edges or text_budget < 1 or max_images < 0
            or input_budget < 1 or fixed_prompt_tokens < 0 or time_limit <= 0):
        raise ValueError('nonempty candidates, matching scores and positive budgets required')
    if len({edge['edge_id'] for edge in edges}) != len(edges):
        raise ValueError('duplicate evidence identity')
    if any(not math.isfinite(float(value)) for value in log_odds):
        raise ValueError('finite frozen-model relevance log odds required')
    if any(identity not in atoms_by_id for edge in edges for identity in edge['atom_ids']):
        raise ValueError('unknown observable atom identity')
    atom_ids = sorted({identity for edge in edges for identity in edge['atom_ids']},
                      key=lambda identity: atoms_by_id[identity]['ordinal'])
    if any(not edge['atom_ids'] or len(set(edge['atom_ids'])) != len(edge['atom_ids']) for edge in edges):
        raise ValueError('nonempty, unique provenance members required')
    refs = images_for_atoms([atoms_by_id[identity] for identity in atom_ids])
    images = [ref.get('resolved', ref['normalized']) for ref in refs]
    if any(identity not in image_token_costs or image_token_costs[identity] < 1 for identity in images):
        raise ValueError('actual processor grid token cost required for every candidate image')
    edge_count = len(edges)
    atom_index = {identity: edge_count + offset for offset, identity in enumerate(atom_ids)}
    image_index = {identity: edge_count + len(atom_ids) + offset for offset, identity in enumerate(images)}
    variable_count = edge_count + len(atom_ids) + len(images)
    costs = {identity: len(tokenizer.encode(render_atoms([atoms_by_id[identity]]) + '\n',
                                           add_special_tokens=False)) for identity in atom_ids}
    memberships = {identity: [] for identity in atom_ids}
    for edge_index, edge in enumerate(edges):
        for identity in edge['atom_ids']:
            memberships[identity].append(edge_index)
    constraints = []

    def constrain(coefficients, lower=-np.inf, upper=np.inf):
        constraints.append((coefficients, lower, upper))

    for identity, parents in memberships.items():
        for parent in parents:
            constrain({parent: 1, atom_index[identity]: -1}, upper=0)
        constrain({atom_index[identity]: 1, **{parent: -1 for parent in parents}}, upper=0)
    image_parents = {identity: [] for identity in images}
    for identity in atom_ids:
        for ref in images_for_atoms([atoms_by_id[identity]]):
            image_id = ref.get('resolved', ref['normalized'])
            image_parents[image_id].append(atom_index[identity])
    for identity, parents in image_parents.items():
        for parent in parents:
            constrain({parent: 1, image_index[identity]: -1}, upper=0)
        constrain({image_index[identity]: 1, **{parent: -1 for parent in parents}}, upper=0)
    text_coefficients = {atom_index[identity]: cost for identity, cost in costs.items()}
    constrain(text_coefficients, upper=text_budget)
    constrain({position: 1 for position in image_index.values()}, upper=max_images)
    constrain({**text_coefficients,
               **{image_index[identity]: image_token_costs[identity] for identity in images}},
              upper=input_budget - fixed_prompt_tokens)
    objective = np.zeros(variable_count)
    objective[:edge_count] = -np.asarray(log_odds, dtype=np.float64)
    started = time.perf_counter()
    rejected = []
    while True:
        matrix = lil_matrix((len(constraints), variable_count), dtype=np.float64)
        for row_index, (coefficients, lower, upper) in enumerate(constraints):
            for column_index, value in coefficients.items():
                matrix[row_index, column_index] = value
        remaining = time_limit - (time.perf_counter() - started)
        if remaining <= 0:
            raise RuntimeError('MILP budget exhausted before exact rendered-token feasibility')
        solution = milp(objective, integrality=np.ones(variable_count),
                        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
                        constraints=LinearConstraint(matrix.tocsc(),
                            np.asarray([item[1] for item in constraints]),
                            np.asarray([item[2] for item in constraints])),
                        options={'time_limit': remaining, 'mip_rel_gap': 0.0})
        if solution.x is None:
            raise RuntimeError(f'no MILP incumbent: {solution.message}')
        if np.max(np.abs(solution.x - np.round(solution.x))) > 1e-5:
            raise RuntimeError('nonintegral evidence plan')
        chosen = [index for index in range(edge_count) if solution.x[index] > 0.5]
        selected_ids = {identity for index in chosen for identity in edges[index]['atom_ids']}
        selected = [atoms_by_id[identity] for identity in atom_ids if identity in selected_ids]
        text = render_atoms(selected)
        text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        selected_refs = images_for_atoms(selected)
        visual_tokens = sum(image_token_costs[ref.get('resolved', ref['normalized'])] for ref in selected_refs)
        if text_tokens <= text_budget and text_tokens + visual_tokens + fixed_prompt_tokens <= input_budget:
            break
        if not chosen:
            raise RuntimeError('fixed prompt cannot fit the registered input budget')
        rejected.append([edges[index]['edge_id'] for index in chosen])
        constrain({index: 1 for index in chosen}, upper=len(chosen) - 1)
    if len(selected_refs) > max_images:
        raise RuntimeError('solver violated image budget')
    return {'text': text, 'images': selected_refs,
            'accepted_atom_ids': [atom['atom_id'] for atom in selected],
            'selected_hyperedges': [{'edge_id': edges[index]['edge_id'],
                                    'atom_ids': edges[index]['atom_ids'],
                                    'relevance_log_odds': float(log_odds[index])} for index in chosen],
            'text_tokens': text_tokens, 'visual_tokens_from_processor_grid': visual_tokens,
            'image_token_costs': image_token_costs, 'text_budget': text_budget,
            'max_images': max_images, 'objective': -float(solution.fun),
            'solver_status': int(solution.status), 'solver_message': str(solution.message),
            'mip_gap': float(solution.mip_gap), 'mip_dual_bound': float(solution.mip_dual_bound),
            'mip_node_count': int(solution.mip_node_count),
            'planning_seconds': time.perf_counter() - started,
            'render_feasibility_cuts': rejected,
            'policy': 'frozen pointwise relevance log-odds MAP subset; exact union/closure MILP',
            'boundary': 'source exchange closure, not factual causality; externally supplied frozen relevance is not a sufficient-evidence guarantee; no task labels, keyword rules, fitted thresholds, or trained policy; optimality only when solver status=0'}


class FrozenRelevanceModel:
    def __init__(self, path, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval()
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.yes_id = self.tokenizer.convert_tokens_to_ids('yes')
        self.no_id = self.tokenizer.convert_tokens_to_ids('no')

    def score(self, query, documents, batch_size=8):
        values = []
        for begin in range(0, len(documents), batch_size):
            prompts = [self.prefix + '<Instruct>: Given a question, retrieve relevant passages that answer the question\n<Query>: '
                       + query + '\n<Document>: ' + document + self.suffix
                       for document in documents[begin:begin + batch_size]]
            encoded = self.tokenizer(prompts, padding=True, truncation=False, return_tensors='pt').to(self.device)
            if encoded.input_ids.shape[1] > 4096:
                raise ValueError('complete evidence exceeds registered reranker budget; no silent truncation')
            with self.torch.inference_mode():
                logits = self.model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            values.extend((logits[:, self.yes_id] - logits[:, self.no_id]).cpu().tolist())
        return values
