"""Reconstruct the paper-count-compatible release without looking at QA answers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from omni_prepare import prepare


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write_new_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def register(source, prior_split, output):
    if output.exists():
        raise ValueError('fresh registration directory required')
    full = [json.loads(line) for line in (source / 'questions.jsonl').read_text().splitlines() if line]
    filtered = [json.loads(line) for line in (source / 'filtered_questions.jsonl').read_text().splitlines() if line]
    histories = [json.loads(line) for line in (source / 'data.jsonl').read_text().splitlines() if line]
    users = {str(record['uuid']) for record in filtered}
    paper_questions = [record for record in full if str(record['uuid']) in users]
    paper_histories = [record for record in histories if str(record['uuid']) in users]
    question_ids = {(str(record['uuid']), str(question['question_id']))
                    for record in paper_questions for question in record['questions']}
    filtered_ids = {(str(record['uuid']), str(question['question_id']))
                    for record in filtered for question in record['questions']}
    counts = Counter(question['question_type'] for record in paper_questions for question in record['questions'])
    expected = {'single_hop': 986, 'multi_hop': 1135, 'knowledge_update': 1010,
                'temporal_reasoning': 773, 'implicit_preference': 1226,
                'abstention': 1208, 'visual_reasoning': 1077}
    total = sum(len(record['questions']) for record in paper_questions)
    if (len(users) != 16 or total != 7415 or dict(counts) != expected
            or len(question_ids) != total or not filtered_ids.issubset(question_ids)
            or len(paper_histories) != 16):
        raise ValueError('paper statistics / filtered lineage fail; do not guess a release')
    output.mkdir(parents=True)
    for name, records in (('questions_paper_users.jsonl', paper_questions),
                          ('history_paper_users.jsonl', paper_histories)):
        with (output / name).open('x') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    audit = prepare(output / 'history_paper_users.jsonl', output / 'questions_paper_users.jsonl',
                    output / 'prepared', source / 'image.zip', prior_split)
    split = json.loads((output / 'prepared' / 'split.json').read_text())
    manifest = {'schema_version': 'omni-main-closure-protocol-v1',
        'dataset': 'MobileMem-Omni paper-count-compatible 16-user reconstruction',
        'selection': 'all original questions of the 16 users present in upstream filtered release; no per-question quality filtering',
        'users': sorted(users, key=int), 'questions': total, 'question_type_counts': dict(counts),
        'source_files_sha256': {name: digest(source / name) for name in
                                ('data.jsonl', 'questions.jsonl', 'filtered_questions.jsonl')},
        'prepared_audit': audit, 'split': split,
        'reader': 'Qwen3.5-4B', 'training': False, 'quantization': False,
        'judge': {'model': 'Qwen3-VL-8B-Instruct', 'official_model': 'Qwen3-14B',
                  'official_score': False, 'temperature': 0.0,
                  'rubric': 'Omni strict coverage, gold, prediction and relevant evidence',
                  'denominators': ['all frozen requests', 'valid judge labels; invalid count always reported']},
        'allowed_serving_information': ['question', 'observable history', 'provenance', 'pretrained neural scores', 'runtime state'],
        'prohibited': ['gold serving route', 'question-type routing', 'keyword answer matching',
                       'dataset-fitted scoring weights', 'fine-tuning', 'unqualified state restore'],
        'timing': 'full request retrieval + evidence optimization + image preparation + prefill + decode; startup/index also reported separately',
        'pim_status': 'unqualified; modeled host and PIM must share one target-platform clock/resource model',
        'release_boundary': 'all seven published category counts match; upstream release identity still documented, no leaderboard/SOTA equivalence inferred from counts alone',
        'superseded': ['omni_full_prepare_v1_20261021: automatic split reassignment; never used',
                       'omni_full_prepare_preserved_v2_20261021: full 20-user pilot universe; not final release scope']}
    write_new_json(output / 'protocol_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior-split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = register(args.source, args.prior_split, args.output)
    print(json.dumps({'users': result['users'], 'questions': result['questions'],
                      'categories': result['question_type_counts'], 'audit': result['prepared_audit']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
