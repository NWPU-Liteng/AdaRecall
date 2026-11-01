"""Merge only complete, disjoint, identically frozen development shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from mobilemem_runtime import select_development


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.run_root.glob('part_*/shard_0.json'))
    parts = [json.loads(path.read_text()) for path in paths]
    if not parts or any(not part['complete'] for part in parts):
        raise ValueError('all shards must be complete')
    first = parts[0]
    count = first['configuration']['shard_count']
    indices = sorted(part['configuration']['shard_index'] for part in parts)
    if len(parts) != count or indices != list(range(count)):
        raise ValueError('missing or duplicate shard')
    invariant_config = {key: value for key, value in first['configuration'].items()
                        if key not in {'output', 'shard_index', 'device'}}
    for part in parts:
        configuration = {key: value for key, value in part['configuration'].items()
                         if key not in {'output', 'shard_index', 'device'}}
        if part['provenance'] != first['provenance'] or configuration != invariant_config:
            raise ValueError('cannot mix code, models, profile or execution contracts')
        if [row['question_id'] for row in part['rows']] != part['selected_question_ids']:
            raise ValueError('shard does not cover its registered questions')
    manifests = [json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    expected = select_development(manifests, limit=invariant_config['limit'], sample_count=invariant_config['sample_count'])
    rows = [row for part in sorted(parts, key=lambda part: part['configuration']['shard_index']) for row in part['rows']]
    if [row['question_id'] for row in rows] != [item['question_id'] for item in expected]:
        raise ValueError('merged results have holes, duplicates, reordering or unauthorized queries')
    output = args.run_root / 'merged.json'
    if output.exists():
        raise ValueError('fresh merge output required')
    payload = {'complete': True, 'configuration': invariant_config, 'provenance': first['provenance'],
               'rows': rows, 'selected_question_ids': [item['question_id'] for item in expected],
               'shards': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
               'official_success_rate': None,
               'boundary': ('development only; one chronological sampled request stream with persistent state; not mobile wall time'
                            if count == 1 and invariant_config['sample_count'] else
                            'development only; one chronological request stream with persistent state; not mobile wall time'
                            if count == 1 else
                            'development only; contiguous cold-state shards, not one persistent-device stream or mobile wall time')}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(output), 'rows': len(rows),
                      'runtime_failures': sum(row['status'] != 'ok' for row in rows)}))


if __name__ == '__main__':
    main()
