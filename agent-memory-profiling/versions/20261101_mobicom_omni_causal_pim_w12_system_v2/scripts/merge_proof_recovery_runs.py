"""Reassemble one label-free candidate's disjoint compute shards without exclusions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omni_proof_runtime import file_digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    cohort = json.loads((args.root / 'cohort.json').read_text())
    for user, identities in cohort['question_ids_by_user'].items():
        paths = sorted(args.root.glob(f'proof_user_{user}_shard_*.json'))
        paths = [path for path in paths if not path.name.endswith('.partial.json')]
        shards, by_id = [], {}
        for path in paths:
            run = json.loads(path.read_text())
            if not run.get('complete') or run['person_id'] != user or run['evaluation_label_files_read']:
                raise ValueError('complete source-isolated candidate shard required')
            expected = [identity for position, identity in enumerate(identities)
                        if position % run['shard_count'] == run['shard_index']]
            if [row['question_id'] for row in run['rows']] != expected:
                raise ValueError('shard changes the registered original-order denominator')
            for row in run['rows']:
                if row['question_id'] in by_id:
                    raise ValueError('overlapping compute shards are not extra candidate trials')
                by_id[row['question_id']] = row
            shards.append((path, run))
        if not shards or set(by_id) != set(identities):
            raise ValueError('all registered requests must survive shard merge')
        for key in ('reader_model', 'embedder_model', 'prepared_audit_sha256', 'cohort_sha256',
                    'index_sha256', 'max_new_tokens', 'max_images', 'total_text_tokens'):
            if len({run[key] for _, run in shards}) != 1:
                raise ValueError('compute shards must share one frozen candidate contract')
        output = args.root / f'proof_user_{user}.json'
        if output.exists():
            raise ValueError('fresh merged candidate output required')
        result = dict(shards[0][1])
        result.update(question_ids=identities, rows=[by_id[identity] for identity in identities],
            shard_index=None, shard_count=len(shards),
            compute_shards=[{'path': str(path), 'sha256': file_digest(path),
                             'requests': len(run['rows'])} for path, run in shards],
            model_load_and_index_seconds=sum(run['model_load_and_index_seconds'] for _, run in shards),
            execution_order='per-shard original order; merged back to original user order; not online arrival replay')
        write_json(output, result)


if __name__ == '__main__':
    main()
