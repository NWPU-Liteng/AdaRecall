import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from eval_evidence_recall import (SCHEMA, build_report, digest,
                                  resolve_source_indexes, write_report)


SHA_A = hashlib.sha256(b'image-a').hexdigest()
SHA_B = hashlib.sha256(b'image-b').hexdigest()

REF_A_LABEL = {
    'raw': 'image/uid0/event/0_event_A.png',
    'normalized': 'uid0/event/0_event_A.png',
    'resolved': 'uid0/camera_photos/0_event_A.png',
    'status': 'unique_user_basename',
}
REF_A_SOURCE = {
    'raw': 'image/uid0/camera_photos/0_event_A.png',
    'normalized': 'uid0/camera_photos/0_event_A.png',
    'status': 'exact',
}
REF_B_LABEL = {
    'raw': 'image/uid0/money/0_money_B.png',
    'normalized': 'uid0/money/0_money_B.png',
    'resolved': 'uid0/transaction_records/0_money_B.png',
    'status': 'unique_user_basename',
}
REF_B_SOURCE = {
    'raw': 'image/uid0/transaction_records/0_money_B.png',
    'normalized': 'uid0/transaction_records/0_money_B.png',
    'status': 'exact',
}
REF_MISSING = {
    'raw': 'image/uid0/event/0_missing.png',
    'normalized': 'uid0/event/0_missing.png',
    'resolved': 'uid0/camera_photos/0_missing.png',
    'status': 'unique_user_basename',
}


def text_object(session_id):
    return {
        'object_id': f'text:{session_id}',
        'kind': 'text',
        'atom_id': f'{session_id}:turn:0',
        'session_id': session_id,
    }


def image_object(sha256, session_id, ref):
    return {
        'object_id': f'image:{sha256}',
        'kind': 'image',
        'source_sha256': sha256,
        'image_ref': ref,
        'sources': [{
            'atom_id': f'{session_id}:turn:1',
            'session_id': session_id,
            'image_ref': ref,
        }],
    }


def label(question_id, *, question_type='single_hop', sessions,
          images=None):
    return {
        'question_id': question_id,
        'person_id': '0',
        'question_type': question_type,
        'source_session_ids': sessions,
        'evidence_image_refs': images or [],
    }


def run_row(question_id, proof_status, accepted):
    return {
        'person_id': '0',
        'question_id': question_id,
        'proof_plan': {
            'proof_status': proof_status,
            'accepted_object_ids': accepted,
        },
    }


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                            for row in rows))


def write_run(run_dir, rows, index_sha256, *, complete=True):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'complete': complete,
        'person_id': '0',
        'index_sha256': index_sha256,
        'rows': rows,
    }
    (run_dir / 'user_0_shard_0.json').write_text(json.dumps(payload))


def create_fixture(root, *, rows=None, labels=None, objects=None):
    prepared = root / 'prepared'
    prepared.mkdir(parents=True)
    labels = labels or [
        label('0_q_1', sessions=['0_1', '0_2'],
              images=[REF_A_LABEL, REF_B_LABEL]),
        label('0_q_2', question_type='temporal_reasoning', sessions=['0_3']),
        label('0_q_3', sessions=['0_3']),
        label('0_q_4', sessions=['0_missing'],
              images=[REF_MISSING]),
    ]
    write_jsonl(prepared / 'labels.jsonl', labels)
    graph = [{
        'person_id': '0',
        'label_free': True,
        'sessions': [{'session_id': value}
                     for value in ('0_1', '0_2', '0_3', '0_4')],
        'atoms': [],
    }]
    (prepared / 'omni_graph.json').write_text(json.dumps(graph))

    index_dir = root / 'indexes_final' / 'proof_index_user_0'
    index_dir.mkdir(parents=True)
    (index_dir / 'index.json').write_text(json.dumps({'complete': True}))
    objects = objects or {
        'text:0_1': text_object('0_1'),
        'text:0_2': text_object('0_2'),
        'text:0_3': text_object('0_3'),
        'text:0_4': text_object('0_4'),
        f'image:{SHA_A}': image_object(SHA_A, '0_2', REF_A_SOURCE),
        f'image:{SHA_B}': image_object(SHA_B, '0_3', REF_B_SOURCE),
    }
    source = {
        'schema': 'source-object-container-v2',
        'person_id': '0',
        'objects': objects,
    }
    (index_dir / 'source.json').write_text(json.dumps(source))
    index_sha256 = digest(index_dir / 'index.json')

    rows = rows or [
        run_row('0_q_1', 'source_located_not_semantically_verified',
                ['text:0_1', f'image:{SHA_A}']),
        run_row('0_q_2', 'no_complete_source_located_proof', ['text:0_4']),
        run_row('0_q_3', 'no_complete_source_located_proof', ['text:0_3']),
        run_row('0_q_4', 'no_complete_source_located_proof', []),
    ]
    run_dir = root / 'run'
    write_run(run_dir, rows, index_sha256)
    (run_dir / 'user_0_shard_0.partial.json').write_text(json.dumps({
        'complete': False,
        'person_id': '0',
        'index_sha256': index_sha256,
        'rows': [run_row('0_q_partial', 'unused', [])],
    }))
    return {
        'prepared': prepared,
        'labels': prepared / 'labels.jsonl',
        'graph': prepared / 'omni_graph.json',
        'index_root': root / 'indexes_final',
        'index_dir': index_dir,
        'run_dir': run_dir,
        'index_sha256': index_sha256,
        'rows': rows,
    }


class EvalEvidenceRecallTests(unittest.TestCase):
    def test_recall_uses_source_mapping_and_requested_groupings(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = create_fixture(Path(directory))
            report = build_report(fixture['run_dir'], fixture['labels'])

        self.assertEqual(report['schema'], SCHEMA)
        self.assertEqual(report['source_files'], ['user_0_shard_0.json'])
        self.assertEqual(report['mapping_failures'], {
            'session_ids': 1,
            'image_refs': 1,
            'total': 2,
        })
        self.assertEqual(report['evidence_recall_session'], {
            'questions_with_labels': 3,
            'labeled_units': 5,
            'mapped_units': 4,
            'hits': 3,
            'mean_per_question': 2 / 3,
            'pooled_recall': 3 / 4,
        })
        self.assertEqual(report['evidence_recall_image'], {
            'questions_with_labels': 1,
            'labeled_units': 3,
            'mapped_units': 2,
            'hits': 1,
            'mean_per_question': 0.5,
            'pooled_recall': 0.5,
        })

        questions = {row['question_id']: row for row in report['questions']}
        self.assertEqual(questions['0_q_1']['accepted_session_ids'],
                         ['0_1', '0_2'])
        self.assertEqual(questions['0_q_1']['session_recall'], 1.0)
        self.assertEqual(questions['0_q_1']['image_recall'], 0.5)
        self.assertEqual(questions['0_q_2']['session_recall'], 0.0)
        self.assertEqual(questions['0_q_3']['session_recall'], 1.0)
        self.assertIsNone(questions['0_q_4']['session_recall'])
        self.assertIsNone(questions['0_q_4']['image_recall'])
        self.assertEqual(questions['0_q_4']['mapping_failures'], {
            'session_ids': ['0_missing'],
            'image_refs': [REF_MISSING],
            'total': 2,
        })

        no_proof = report['by_proof_status']['no_complete_source_located_proof']
        self.assertEqual(no_proof['questions'], 3)
        self.assertEqual(no_proof['evidence_recall_session']['hits'], 1)
        self.assertEqual(no_proof['evidence_recall_session']['mapped_units'], 2)
        self.assertEqual(no_proof['mapping_failures']['total'], 2)
        self.assertEqual(
            report['by_question_type']['single_hop']
            ['evidence_recall_session']['pooled_recall'], 1.0)
        self.assertIn(
            'single_hop',
            report['by_proof_status_and_question_type']
            ['no_complete_source_located_proof'])
        self.assertEqual(report['candidate_recall_session']['status'],
                         'skipped_no_candidate_dump')
        self.assertFalse(report['evidence_levels']['atom'].startswith('computed'))

    def test_write_report_emits_json_and_uses_auto_discovered_index(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = create_fixture(Path(directory))
            output = Path(directory) / 'reports' / 'recall.json'
            path, report = write_report(
                fixture['run_dir'], fixture['labels'], output=output)
            written = json.loads(path.read_text())

        self.assertEqual(path, output)
        self.assertEqual(written, report)
        self.assertEqual(written['questions'][0]['question_id'], '0_q_1')
        self.assertEqual(written['evidence_recall_session']['hits'], 3)
        self.assertFalse(output.with_suffix('.json.tmp').exists())

    def test_index_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = create_fixture(Path(directory))
            (fixture['index_dir'] / 'index.json').write_text(
                json.dumps({'complete': True, 'changed': True}))
            with self.assertRaisesRegex(ValueError, 'does not match run shard'):
                build_report(fixture['run_dir'], fixture['labels'])

    def test_incomplete_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = create_fixture(Path(directory))
            write_run(fixture['run_dir'], fixture['rows'],
                      fixture['index_sha256'], complete=False)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                build_report(fixture['run_dir'], fixture['labels'])

    def test_run_question_without_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = create_fixture(Path(directory))
            rows = fixture['rows'] + [run_row('0_q_unknown', 'missing', [])]
            write_run(fixture['run_dir'], rows, fixture['index_sha256'])
            with self.assertRaisesRegex(ValueError, 'has no label'):
                build_report(fixture['run_dir'], fixture['labels'])

    def test_unknown_final_object_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = create_fixture(Path(directory))
            rows = [dict(row) for row in fixture['rows']]
            rows[2] = run_row('0_q_3', 'no_complete_source_located_proof',
                              ['text:not-in-source'])
            write_run(fixture['run_dir'], rows, fixture['index_sha256'])
            with self.assertRaisesRegex(ValueError, 'absent from source index'):
                build_report(fixture['run_dir'], fixture['labels'])

    def test_conflicting_image_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conflicting = {
                'text:0_1': text_object('0_1'),
                'text:0_2': text_object('0_2'),
                'text:0_3': text_object('0_3'),
                'text:0_4': text_object('0_4'),
                f'image:{SHA_A}': image_object(SHA_A, '0_2', REF_A_SOURCE),
                f'image:{SHA_B}': image_object(SHA_B, '0_3', REF_A_SOURCE),
            }
            fixture = create_fixture(root, objects=conflicting)
            with self.assertRaisesRegex(ValueError, 'ambiguous image reference'):
                build_report(fixture['run_dir'], fixture['labels'])

    def test_missing_index_root_requires_explicit_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / 'prepared' / 'labels.jsonl'
            labels.parent.mkdir()
            write_jsonl(labels, [label('0_q_1', sessions=['0_1'])])
            with self.assertRaisesRegex(FileNotFoundError, 'pass --index-root'):
                resolve_source_indexes(labels, None, ['0'])


if __name__ == '__main__':
    unittest.main()
