"""Reconstruct exact processor operands before measuring non-oracle prefix opportunities."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import torch
from transformers import AutoProcessor

from omni_visual_feature_store import VisualFeatureStore
from run_omni_raw_image_smoke import SYSTEM


def reconstruct(processor, archive, run, row, manifest):
    if (manifest['split'] != 'development' or manifest['person_id'] != row['person_id']
            or hashlib.sha256(manifest['question'].encode()).hexdigest() != row['question_sha256']):
        raise ValueError('development serving question differs')
    headers = {'local_hyperedge': '局部对话超边：\n',
               'provenance_milp': '可观察完整交互证据：\n'}
    if run['evidence_policy_name'] not in headers:
        raise ValueError('unregistered evidence rendering protocol')
    content = [{'type': 'text', 'text': '原始对话记忆：\n' + headers[run['evidence_policy_name']]
        + row['hypergraph_evidence_plan']['text']
        + f"\n\n查询时间协议：完整历史末端（{manifest['effective_timestamp']}）"
        + f"\n问题：{manifest['question']}\n请依据文字和随后附上的原始图片作答。"}]
    images = []
    for image in row['images']:
        raw = archive.read(image['archive_member'])
        if hashlib.sha256(raw).hexdigest() != image['sha256']:
            raise ValueError('original image digest mismatch')
        decoded = Image.open(io.BytesIO(raw)).convert('RGB')
        images.append(decoded)
        content.extend([{'type': 'text', 'text':
            f"以下原图来自会话 {image['source_session_id']} 的对话轮 {image['source_atom_id']}："},
            {'type': 'image', 'image': decoded}])
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    kwargs = {'text': [text], 'return_tensors': 'pt'}
    if images:
        kwargs['images'] = images
    inputs = processor(**kwargs)
    hashes = {name: VisualFeatureStore.tensor_digest(value) for name, value in inputs.items()
              if isinstance(value, torch.Tensor)}
    if hashes != row['processor_inputs_sha256'] or inputs.input_ids.shape[1] != row['input_tokens']:
        raise ValueError('reconstructed complete processor tensors do not match serving')
    return inputs


def longest_prefix(previous, current):
    return max((next((index for index, (left, right) in enumerate(zip(old, current)) if left != right),
                     min(len(old), len(current))) for old in previous), default=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh prefix audit required')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    manifests_path = args.prepared / 'manifests.jsonl'
    if hashlib.sha256(manifests_path.read_bytes()).hexdigest() != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('prepared manifests changed')
    manifests = {row['question_id']: row for row in map(json.loads, manifests_path.read_text().splitlines())}
    torch.set_num_threads(2)
    users = []
    for path in args.runs:
        run = json.loads(path.read_text())
        if not run.get('complete'):
            raise ValueError('complete serving run required')
        processor = AutoProcessor.from_pretrained(run['reader'], local_files_only=True)
        processor.image_processor.size['longest_edge'] = run['image_max_pixels']
        prior = []
        records = []
        with zipfile.ZipFile(args.archive) as archive:
            for row in run['rows']:
                inputs = reconstruct(processor, archive, run, row, manifests[row['question_id']])
                tokens = inputs.input_ids[0].tolist()
                image_token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
                cut = tokens.index(image_token) if image_token in tokens else len(tokens)
                text_prefix = tokens[:cut]
                common = longest_prefix(prior, text_prefix)
                if common == len(text_prefix) and common:
                    common -= 1
                records.append({'question_id': row['question_id'], 'input_tokens': len(tokens),
                                'maximum_prior_identical_token_prefix_before_first_image': common,
                                'image_boundary_token': cut, 'all_processor_tensors_match': True})
                prior.append(text_prefix)
        users.append({'person_id': run['rows'][0]['person_id'], 'records': records})
    result = {'complete': True, 'per_user': users,
        'requests': sum(len(user['records']) for user in users),
        'possible_matching_token_prefixes': sum(record['maximum_prior_identical_token_prefix_before_first_image']
                                               for user in users for record in user['records']),
        'reader_state_reuse_qualified': False, 'actual_state_restore_executed': False,
        'evaluation_label_files_read': [],
        'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.runs},
        'boundary': 'identical token-prefix opportunity only, not legal recurrent/KV equivalence or latency savings; before first image only, no arbitrary KV concatenation or future knowledge used for serving'}
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ('per_user', 'inputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
