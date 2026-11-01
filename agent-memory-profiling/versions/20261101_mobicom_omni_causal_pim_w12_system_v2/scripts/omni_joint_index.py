"""Frozen joint image/text producers, immutable indices, and exact cosine search."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as functional

from omni_evidence_planner import images_for_atoms
from omni_service_measurement import measure_vision
from omni_patch_projection import configure_patch_projection


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            hasher.update(block)
    return hasher.hexdigest()


def model_files(path):
    return {str(item.relative_to(path)): digest(item) for item in sorted(Path(path).rglob('*'))
            if item.is_file() and item.suffix in ('.json', '.safetensors', '.txt', '.py', '.jinja')}


def import_official(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JointEncoder:
    def __init__(self, path, device, image_max_pixels, max_input_tokens=8192):
        from transformers import AutoProcessor

        self.path = Path(path)
        self.device = device
        self.max_input_tokens = max_input_tokens
        source = self.path / 'scripts/qwen3_vl_embedding.py'
        official = import_official(source, 'frozen_qwen3_joint_embedding')
        self.model = official.Qwen3VLForEmbedding.from_pretrained(
            str(path), local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval()
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.processor.tokenizer.padding_side = 'right'
        self.processor.image_processor.size['longest_edge'] = image_max_pixels

    def encode(self, documents):
        prompts, all_images = [], []
        for document in documents:
            content = [{'type': 'image', 'image': image} for image in document['images']]
            if document['text']:
                content.append({'type': 'text', 'text': document['text']})
            if not content:
                raise ValueError('empty observable embedding producer')
            messages = [{'role': 'system', 'content': [{'type': 'text', 'text': "Represent the user's input."}]},
                        {'role': 'user', 'content': content}]
            prompts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            all_images.extend(document['images'])
        kwargs = {'text': prompts, 'padding': True, 'truncation': False,
                  'return_tensors': 'pt'}
        if all_images:
            kwargs['images'] = all_images
        inputs = self.processor(**kwargs).to(self.device)
        if inputs.input_ids.shape[1] > self.max_input_tokens:
            raise ValueError('complete joint producer exceeds input budget; no truncation')
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False).last_hidden_state
        positions = inputs.attention_mask.shape[1] - 1 - inputs.attention_mask.flip(dims=[1]).argmax(dim=1)
        rows = torch.arange(output.shape[0], device=output.device)
        vectors = torch.nn.functional.normalize(output[rows, positions].float(), dim=-1)
        if not torch.isfinite(vectors).all():
            raise ValueError('nonfinite joint producer vectors')
        records = [{'input_tokens': int(mask.sum().item()), 'input_ids_sha256': hashlib.sha256(
            ids[mask.bool()].detach().cpu().numpy().tobytes()).hexdigest()}
            for ids, mask in zip(inputs.input_ids, inputs.attention_mask)]
        return vectors, records


class JointEmbeddingRelevance:
    def __init__(self, path, device, image_max_pixels, batch_size=4, max_input_tokens=8192):
        self.encoder = JointEncoder(path, device, image_max_pixels, max_input_tokens)
        self.batch_size = batch_size

    def score(self, query, edges, atoms_by_id, load_image):
        query_vector, query_record = self.encoder.encode([{'text': query, 'images': []}])
        values, records = [], []
        for begin in range(0, len(edges), self.batch_size):
            documents, batch_records = [], []
            for edge in edges[begin:begin + self.batch_size]:
                refs = images_for_atoms([atoms_by_id[identity] for identity in edge['atom_ids']])
                loaded = [load_image(ref) for ref in refs]
                documents.append({'text': edge['embedding_text'],
                                  'images': [image for image, _ in loaded]})
                batch_records.append({'edge_id': edge['edge_id'],
                    'image_sha256': [checksum for _, checksum in loaded]})
            vectors, input_records = self.encoder.encode(documents)
            query_tensor = query_vector[0].to(vectors.device)
            scores = torch.mv(vectors, query_tensor).cpu().tolist()
            values.extend(scores)
            records.extend({**record, **input_record, 'relevance_cosine': score}
                           for record, input_record, score in zip(batch_records, input_records, scores))
        return values, {'query': query_record[0], 'edges': records,
                        'model': str(self.encoder.path), 'pooling': 'official lasttoken L2 cosine'}


def rank_indices(vectors, query, identities, limit):
    if vectors.ndim != 2 or query.shape != (vectors.shape[1],) or len(identities) != len(vectors):
        raise ValueError('joint index/query shape mismatch')
    if limit < 1 or not np.isfinite(vectors).all() or not np.isfinite(query).all():
        raise ValueError('positive retrieval limit and finite vectors required')
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate producer identity')
    scores = vectors @ query
    positions = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), identities[index]))[:limit]
    return positions, [float(scores[position]) for position in positions]


class JointIndex:
    def __init__(self, path, prepared_audit_sha256, person_id, encoder_files, image_max_pixels,
                 archive_sha256):
        self.path = Path(path)
        metadata = json.loads((self.path / 'index.json').read_text())
        if not metadata.get('complete') or metadata['evaluation_label_files_read']:
            raise ValueError('incomplete or label-dependent producer index')
        expected = (prepared_audit_sha256, str(person_id), encoder_files, image_max_pixels, archive_sha256)
        actual = (metadata['prepared_audit_sha256'], metadata['person_id'],
                  metadata['model_files_sha256'], metadata['image_max_pixels'], metadata['archive_sha256'])
        if actual != expected or digest(self.path / 'vectors.npy') != metadata['vectors_sha256']:
            raise ValueError('joint producer identity or payload mismatch')
        self.vectors = np.load(self.path / 'vectors.npy', allow_pickle=False)
        self.edges = metadata['edges']
        if self.vectors.shape != (len(self.edges), metadata['dimension']):
            raise ValueError('joint index payload shape mismatch')
        self.edge_positions = {edge['edge_id']: index for index, edge in enumerate(self.edges)}
        if len(self.edge_positions) != len(self.edges):
            raise ValueError('duplicate exchange identity')
        self.views = metadata.get('evidence_views', [])
        self.view_vectors = None
        if self.views:
            payload = self.path / 'evidence_view_vectors.npy'
            if digest(payload) != metadata.get('evidence_view_vectors_sha256'):
                raise ValueError('evidence-view payload hash mismatch')
            self.view_vectors = np.load(payload, allow_pickle=False)
            if (self.view_vectors.shape != (len(self.views), metadata['dimension'])
                    or not np.isfinite(self.view_vectors).all()
                    or not np.allclose(np.linalg.norm(self.view_vectors, axis=1), 1., atol=1e-4)
                    or len({view['view_id'] for view in self.views}) != len(self.views)
                    or {view.get('edge_id') for view in self.views} != set(self.edge_positions)
                    or any(view.get('view_type') not in {'text', 'image'} for view in self.views)):
                raise ValueError('evidence-view identity or shape mismatch')
            for view in self.views:
                edge = self.edges[self.edge_positions[view['edge_id']]]
                if view.get('atom_ids') != edge['atom_ids']:
                    raise ValueError('evidence-view provenance differs from parent exchange')
                if view['view_type'] == 'image':
                    ref = view.get('image_ref', {})
                    if (view.get('source_atom_id') not in edge['atom_ids']
                            or ref.get('status') not in {'exact', 'unique_user_basename'}
                            or view.get('image_identity') != ref.get('resolved', ref.get('normalized'))):
                        raise ValueError('image evidence must retain its observed source')
        self.metadata = metadata
        self.hits = 0
        self.edge_scope = metadata.get('edge_scope', 'all_complete_exchange')

    def search(self, query, limit):
        indices, scores = rank_indices(self.vectors, query, [edge['edge_id'] for edge in self.edges], limit)
        self.hits += len(indices)
        return [self.edges[index] for index in indices], scores

    def search_evidence_units(self, query, limit):
        if self.view_vectors is None or not self.views:
            raise ValueError('independent text/image evidence views are required')
        query = np.asarray(query, dtype=self.view_vectors.dtype)
        if (query.shape != (self.view_vectors.shape[1],) or not np.isfinite(query).all()
                or not np.isclose(np.linalg.norm(query), 1., atol=1e-4) or limit < 1):
            raise ValueError('finite aligned evidence-view vectors and positive limit required')
        scores = self.view_vectors @ query
        best = {}
        for index, (view, score) in enumerate(zip(self.views, scores)):
            edge_id = view['edge_id']
            candidate = (float(score), view['view_id'], index)
            if (edge_id not in best or candidate[0] > best[edge_id][0]
                    or candidate[0] == best[edge_id][0] and candidate[1] < best[edge_id][1]):
                best[edge_id] = candidate
        ranked = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
        self.hits += len(ranked)
        edges = [self.edges[self.edge_positions[edge_id]] for edge_id, _ in ranked]
        vectors = np.stack([self.view_vectors[record[2]] for _, record in ranked])
        records = [{'edge_id': edge_id, 'view_id': record[1],
                    'view_type': self.views[record[2]]['view_type'],
                    'image_identity': self.views[record[2]].get('image_identity'),
                    'score': record[0], 'score_semantics': 'shared_encoder_cosine_max_over_provenance_views'}
                   for edge_id, record in ranked]
        return edges, [record[0] for _, record in ranked], vectors, records


class TrainedMultimodalReranker:
    def __init__(self, path, device, image_max_pixels, batch_size=4, max_input_tokens=8192,
                 profile=False, patch_projection='conv'):
        self.device = torch.device(device)
        self.profile = profile
        if patch_projection not in {'conv', 'linear'}:
            raise ValueError('patch_projection must be conv or linear')
        self.patch_projection = patch_projection
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        official = import_official(Path(path) / 'scripts/qwen3_vl_reranker.py', 'frozen_qwen3_joint_reranker')
        self.wrapper = official.Qwen3VLReranker(
            str(path), max_pixels=image_max_pixels, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True)
        self.wrapper.processor.image_processor.size['longest_edge'] = image_max_pixels
        self.wrapper.processor.tokenizer.padding_side = 'left'
        configure_patch_projection(self.wrapper.model.visual, patch_projection)

    def score(self, query, edges, atoms_by_id, load_image):
        scoring_started = time.perf_counter()
        values, records = [], []
        timings = {'image_load_seconds': 0.0, 'prompt_build_seconds': 0.0,
                   'patch_projection': self.patch_projection,
                   'processor_transfer_seconds': 0.0, 'model_forward_seconds': 0.0,
                   'batch_count': 0, 'candidate_count': len(edges), 'image_count': 0,
                   'synchronized_cuda': self.profile and self.device.type == 'cuda',
                   'padded_input_tokens': 0, 'valid_input_tokens': 0,
                   'vision_in_forward_seconds': 0.0, 'vision_calls': 0,
                   'batches': []}
        for begin in range(0, len(edges), self.batch_size):
            timings['batch_count'] += 1
            batch = edges[begin:begin + self.batch_size]
            prompts, images, batch_records = [], [], []
            for edge in batch:
                refs = images_for_atoms([atoms_by_id[identity] for identity in edge['atom_ids']])
                phase_started = time.perf_counter()
                loaded = [load_image(ref) for ref in refs]
                timings['image_load_seconds'] += time.perf_counter() - phase_started
                timings['image_count'] += len(loaded)
                phase_started = time.perf_counter()
                messages = self.wrapper.format_mm_instruction(
                    query, None, None, edge['embedding_text'], None, None,
                    instruction=self.wrapper.default_instruction)
                content = messages[1]['content']
                insert_at = next(index + 1 for index, item in enumerate(content)
                                 if item.get('text') == '\n<Document>:')
                content[insert_at:insert_at] = [{'type': 'image', 'image': image} for image, _ in loaded]
                prompts.append(self.wrapper.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True))
                images.extend(image for image, _ in loaded)
                batch_records.append({'edge_id': edge['edge_id'],
                    'image_sha256': [checksum for _, checksum in loaded],
                    'prompt_sha256': hashlib.sha256(prompts[-1].encode()).hexdigest()})
                timings['prompt_build_seconds'] += time.perf_counter() - phase_started
            kwargs = {'text': prompts, 'padding': True, 'truncation': False,
                      'return_tensors': 'pt'}
            if images:
                kwargs['images'] = images
            phase_started = time.perf_counter()
            inputs = self.wrapper.processor(**kwargs).to(self.device)
            if timings['synchronized_cuda']:
                torch.cuda.synchronize(self.device)
            timings['processor_transfer_seconds'] += time.perf_counter() - phase_started
            if inputs.input_ids.shape[1] > self.max_input_tokens:
                raise ValueError('complete reranker exchange exceeds input budget; no truncation')
            phase_started = time.perf_counter()
            vision_context = (measure_vision(self.wrapper.model,
                lambda: torch.cuda.synchronize(self.device)
                if self.device.type == 'cuda' else None) if self.profile else nullcontext())
            with vision_context as vision_timing, torch.inference_mode():
                hidden = self.wrapper.model(**inputs, use_cache=False).last_hidden_state[:, -1]
                logits = self.wrapper.score_linear(hidden).float().squeeze(-1)
            if timings['synchronized_cuda']:
                torch.cuda.synchronize(self.device)
            batch_forward_seconds = time.perf_counter() - phase_started
            timings['model_forward_seconds'] += batch_forward_seconds
            if vision_timing is not None:
                timings['vision_in_forward_seconds'] += vision_timing['vision_in_prefill_seconds']
                timings['vision_calls'] += vision_timing['vision_calls']
            if self.profile:
                timings['batches'].append({'edge_ids': [edge['edge_id'] for edge in batch],
                    'forward_seconds': batch_forward_seconds,
                    'vision_seconds': None if vision_timing is None else vision_timing['vision_in_prefill_seconds'],
                    'padded_sequence_length': inputs.input_ids.shape[1],
                    'valid_tokens': inputs.attention_mask.sum(dim=-1).cpu().tolist(),
                    'image_grid_thw': (inputs['image_grid_thw'].cpu().tolist()
                                       if 'image_grid_thw' in inputs else [])})
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite trained reranker logits')
            values.extend(logits.cpu().tolist())
            records.extend(batch_records)
            timings['padded_input_tokens'] += inputs.input_ids.numel()
            timings['valid_input_tokens'] += int(inputs.attention_mask.sum().item())
        timings['total_seconds'] = time.perf_counter() - scoring_started
        timings['residual_seconds'] = timings['total_seconds'] - sum(
            timings[key] for key in ('image_load_seconds', 'prompt_build_seconds',
                                    'processor_transfer_seconds', 'model_forward_seconds'))
        return values, records, timings
