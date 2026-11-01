import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_evidence_state_index import build


def write_index(root, records):
    index = root / 'index'
    index.mkdir()
    records_path = index / 'records.json'
    records_path.write_text(json.dumps(records))
    import hashlib
    digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    (index / 'index.json').write_text(json.dumps({
        'schema': 'proof-index-v2',
        'complete': True,
        'person_id': 'test-user',
        'records_sha256': digest,
    }))
    return index


class EvidenceStateIndexTests(unittest.TestCase):
    def test_image_records_normalize_missing_text_span_to_null(self):
        records = [{
            'record_id': 'visual:0',
            'kind': 'image_locator',
            'session_id': 's0',
            'container_id': 'c0',
            'primary_object_id': 'image:0',
            'object_ids': ['image:0'],
            'source_atom_ids': ['turn:0'],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index = write_index(root, records)
            output = root / 'sidecar.json'
            result = build(index, output)
        self.assertEqual(result['record_count'], 1)
        self.assertIsNone(result['records'][0]['source_span'])

    def test_text_records_require_valid_span(self):
        records = [{
            'record_id': 'atom:0',
            'kind': 'source_text',
            'session_id': 's0',
            'container_id': 'c0',
            'primary_object_id': 'text:0',
            'object_ids': ['text:0'],
            'source_atom_ids': ['turn:0'],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            index = write_index(Path(temp_dir), records)
            with self.assertRaisesRegex(ValueError, 'source_span'):
                build(index, Path(temp_dir) / 'sidecar.json')


if __name__ == '__main__':
    unittest.main()
