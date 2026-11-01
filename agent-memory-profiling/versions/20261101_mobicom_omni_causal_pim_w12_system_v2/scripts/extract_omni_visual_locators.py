"""Question/label-free, content-addressed visible image extraction, once per image."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import time
import zipfile

from omni_proof_runtime import (authorize_history_users, checksum, file_digest, load_reader,
                                parse_object, write_json)
from run_omni_raw_image_smoke import DEFAULT_READER


PROMPT = ('只记录这张原图中能看清的文字、数值、日期、对象及它们的可见关系，保留专有名词和细节。'
          '不要推断人物身份、事件、不可辨文字，不要借用文件名或任何问题。'
          '输出JSON {"regions":[{"box":[0,0,1,1],"text":"可见细节，未知明确写未知"}]}。'
          'box是原图归一化坐标，可以用全图；不要只说这是截图。限简洁的可见记录。')


def valid_regions(value, image_size=None):
    if not isinstance(value, dict) or not isinstance(value.get('regions'), list):
        return []
    result = []
    for region in value['regions']:
        if not isinstance(region, dict):
            continue
        box, text = region.get('box'), region.get('text')
        if (isinstance(box, list) and len(box) == 4 and image_size
                and any(float(coordinate) > 1 for coordinate in box)):
            width, height = image_size
            if width > 0 and height > 0:
                box = [float(box[0]) / width, float(box[1]) / height,
                       float(box[2]) / width, float(box[3]) / height]
        if (isinstance(box, list) and len(box) == 4
                and all(type(coordinate) in (int, float) and 0 <= coordinate <= 1 for coordinate in box)
                and box[0] < box[2] and box[1] < box[3] and isinstance(text, str) and text.strip()
                and text.strip() not in {'可见细节，未知明确写未知', '未知明确写未知'}):
            result.append({'box': box, 'text': text})
    return result


def main():
    import torch
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--reader', type=Path, default=DEFAULT_READER)
    parser.add_argument('--users', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--shard-count', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allowed-splits', nargs='+', default=['development'],
                        choices=['development', 'validation', 'holdout'])
    args = parser.parse_args()
    if (not args.resume and args.output.exists()) or not 0 <= args.shard_index < args.shard_count or args.batch_size < 1:
        raise ValueError('fresh shard output, valid shard and positive batch size required')
    started = time.perf_counter()
    if args.resume:
        status_path = args.output / 'status.json'
        registration_path = args.output / 'registration.json'
        jobs_path = args.output / 'jobs.json'
        records_path = args.output / 'records.jsonl'
        if not all(path.is_file() for path in (status_path, registration_path, jobs_path, records_path)):
            raise ValueError('resume requires registration, jobs, records, and status files')
        status = json.loads(status_path.read_text())
        registration = json.loads(registration_path.read_text())
        if status.get('complete'):
            raise ValueError('cannot resume a complete locator shard')
        if (registration.get('shard_index') != args.shard_index
                or registration.get('shard_count') != args.shard_count):
            raise ValueError('resume shard identity differs from registration')
        jobs = [tuple(job) for job in json.loads(jobs_path.read_text())]
        rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        seen = {row.get('image_sha256') for row in rows}
        if len(seen) != len(rows) or None in seen:
            raise ValueError('resume records contain duplicate or missing image identities')
        expected = {identity for identity, _ in jobs}
        if not seen <= expected:
            raise ValueError('resume records contain identities outside the registered jobs')
        jobs = [(identity, member) for identity, member in jobs if identity not in seen]
        batches = list(status.get('batches', []))
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        graphs = authorize_history_users(args.prepared, args.users,
                                         allowed_splits=set(args.allowed_splits))
        members = sorted({ref.get('resolved', ref['normalized']) for graph in graphs
            for atom in graph['atoms'] for ref in atom.get('image_refs', [])
            if ref.get('status') in {'exact', 'unique_user_basename'}})
        args.output.mkdir(parents=True)
        registration = {'evaluation_label_files_read': [], 'questions_read': False,
            'prepared_audit_sha256': file_digest(args.prepared / 'audit.json'),
            'archive_sha256': file_digest(args.image_archive), 'users': args.users,
            'prompt': PROMPT, 'prompt_sha256': checksum(PROMPT), 'image_max_pixels': 262144,
            'shard_index': args.shard_index, 'shard_count': args.shard_count,
            'batch_size': args.batch_size, 'max_new_tokens': 384,
            'source_sha256': file_digest(__file__),
            'reader_files_sha256': {path.name: file_digest(path) for path in args.reader.iterdir()
                if path.suffix in {'.json', '.safetensors', '.jinja', '.txt'}},
            'extraction_status': 'model_locator_only_never_answer_fact'}
        write_json(args.output / 'registration.json', registration)
        by_hash, hashes = {}, {}
        with zipfile.ZipFile(args.image_archive, metadata_encoding='gb18030') as archive:
            for member in members:
                identity = checksum(archive.read(member))
                hashes[member] = identity
                by_hash.setdefault(identity, member)
        jobs = [(identity, member) for identity, member in sorted(by_hash.items())
                if int(identity, 16) % args.shard_count == args.shard_index]
        write_json(args.output / 'image_hashes.json', hashes)
        write_json(args.output / 'jobs.json', jobs)
        rows, batches = [], []
    processor, model = load_reader(args.reader)
    load_seconds = time.perf_counter() - started
    total_jobs = len(rows) + len(jobs)
    mode = 'a' if args.resume else 'x'
    with zipfile.ZipFile(args.image_archive, metadata_encoding='gb18030') as archive, (args.output / 'records.jsonl').open(mode) as stream:
        for begin in range(0, len(jobs), args.batch_size):
            batch = jobs[begin:begin + args.batch_size]
            images = [Image.open(io.BytesIO(archive.read(member))).convert('RGB') for _, member in batch]
            prompts = [processor.apply_chat_template([{'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPT}, {'type': 'image', 'image': image}]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for image in images]
            inputs = processor(text=prompts, images=images, padding=True, return_tensors='pt').to('cuda:0')
            if inputs.input_ids.shape[1] > 8192:
                raise ValueError('visual extraction context exceeds registered budget')
            torch.cuda.synchronize()
            batch_started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=384, do_sample=False)
            torch.cuda.synchronize()
            duration = time.perf_counter() - batch_started
            batch_record = {'batch_id': len(batches), 'image_sha256s': [identity for identity, _ in batch],
                            'seconds': duration, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
            batches.append(batch_record)
            for (identity, _), image, sequence, grid in zip(batch, images, output,
                                                            inputs['image_grid_thw']):
                generated = sequence[inputs.input_ids.shape[1]:].detach().cpu().tolist()
                text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
                stops = model.config.text_config.eos_token_id
                stops = stops if isinstance(stops, list) else [stops]
                eos = any(token in stops for token in generated)
                try:
                    regions = valid_regions(parse_object(text), image_size=image.size) if eos else []
                except (ValueError, TypeError):
                    regions = []
                merge = int(model.model.visual.spatial_merge_size)
                row = {'image_sha256': identity, 'image_size': list(image.size),
                    'regions': regions,
                    'raw_locator': text, 'parse_valid': bool(regions), 'eos': eos,
                    'generated_token_ids': generated, 'batch_id': batch_record['batch_id'],
                    'reader_image_tokens': int(grid.prod().item()) // merge**2,
                    'grid_thw': grid.detach().cpu().tolist(),
                    'locator_sha256': checksum(text), 'status': 'locator_only_original_image_required'}
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                rows.append(row)
            write_json(args.output / 'status.json', {'complete': False, 'completed': len(rows),
                'total': total_jobs, 'invalid': sum(not row['parse_valid'] for row in rows),
                'batches': batches, 'elapsed_seconds': time.perf_counter() - started})
            print(json.dumps({'completed': len(rows), 'total': total_jobs,
                              'batch_seconds': duration, 'elapsed_seconds': time.perf_counter() - started}), flush=True)
            del output, inputs
    write_json(args.output / 'status.json', {'complete': True, 'completed': len(rows),
        'total': total_jobs, 'invalid': sum(not row['parse_valid'] for row in rows),
        'load_and_identity_seconds': load_seconds, 'batches': batches,
        'elapsed_seconds': time.perf_counter() - started,
        'boundary': 'each batch charged once; zero QA/labels; invalid/truncated records retained, no retries or invented fields'})


if __name__ == '__main__':
    main()
