from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from omni_multi_witness import CANDIDATE_ONLY, UNVERIFIED
from omni_witness_descriptor import (SCHEMA, compile_descriptor_index,
                                     descriptor_from_record,
                                     rank_source_family_frontier,
                                     validate_descriptor_index)


def text_record(record_id='atom:0_41:turn:1', tokens=12):
    return {'record_id': record_id, 'kind': 'source_text',
            'primary_object_id': 'text:0_41:turn:1',
            'object_ids': ['text:0_41:turn:1'],
            'source_atom_ids': ['0_41:turn:1'], 'source_span': [0, 4],
            'retrieval_tokens': tokens}


def image_record(record_id, image_hash, region=0):
    object_id = 'image:' + image_hash
    return {'record_id': record_id, 'kind': 'image_locator',
            'primary_object_id': object_id, 'object_ids': [object_id],
            'source_atom_ids': ['0_1:turn:' + str(region)],
            'source_region_index': region}


class WitnessDescriptorTests(unittest.TestCase):
    def test_text_descriptor_is_candidate_only(self):
        descriptor = descriptor_from_record(text_record(), {})
        self.assertEqual(descriptor['source_family'], 'atom:0_41')
        self.assertEqual(descriptor['materialization_tokens'], 12)
        self.assertEqual(descriptor['verification_state'], CANDIDATE_ONLY)
        self.assertEqual(descriptor['semantic_sufficiency'], UNVERIFIED)
        self.assertEqual(descriptor['anchor']['text_index'], 0)

    def test_visual_regions_and_image_object_share_family(self):
        image_hash = 'a' * 64
        first = descriptor_from_record(image_record(
            'visual:' + image_hash + ':region:1', image_hash, 1), {image_hash: 256})
        second = descriptor_from_record(image_record(
            'visual:' + image_hash + ':region:2', image_hash, 2), {image_hash: 256})
        self.assertEqual(first['source_family'], second['source_family'])
        self.assertEqual(first['source_family'], 'image:' + image_hash)
        self.assertEqual(first['materialization_tokens'], 256)

    def test_different_images_are_different_families(self):
        first = descriptor_from_record(image_record(
            'visual:' + 'a' * 64 + ':region:0', 'a' * 64), {'a' * 64: 256})
        second = descriptor_from_record(image_record(
            'visual:' + 'b' * 64 + ':region:0', 'b' * 64), {'b' * 64: 256})
        self.assertNotEqual(first['source_family'], second['source_family'])

    def test_compiled_index_round_trips_and_validates(self):
        records = [text_record(), image_record(
            'visual:' + 'a' * 64 + ':region:0', 'a' * 64)]
        index = compile_descriptor_index(
            records, {'a' * 64: 256}, person_id='0',
            source_index_sha256='source', records_sha256='records')
        self.assertEqual(index['schema'], SCHEMA)
        by_record = validate_descriptor_index(
            index, records, person_id='0', source_index_sha256='source',
            records_sha256='records')
        self.assertEqual(set(by_record), {item['record_id'] for item in records})

    def test_validation_rejects_semantic_upgrade(self):
        records = [text_record()]
        index = compile_descriptor_index(
            records, {}, person_id='0', source_index_sha256='source',
            records_sha256='records')
        index['semantic_claims'] = True
        with self.assertRaises(ValueError):
            validate_descriptor_index(
                index, records, person_id='0', source_index_sha256='source',
                records_sha256='records')

    def test_family_frontier_keeps_one_best_record_per_family(self):
        records = [text_record('atom:0_41:turn:1', 8),
                   text_record('atom:0_41:turn:2', 10),
                   text_record('atom:0_42:turn:1', 6)]
        descriptors, seen = [], set()
        for record in records:
            descriptor = descriptor_from_record(record, {})
            if record['record_id'] == 'atom:0_41:turn:2':
                descriptor['source_family'] = 'atom:0_41'
            descriptors.append(descriptor)
            seen.add(descriptor['record_id'])
        by_record = {item['record_id']: item for item in descriptors}
        frontier = rank_source_family_frontier(
            records, [0.9, 0.8, 0.7], by_record, limit=8)
        self.assertEqual(frontier['source_family_count'], 2)
        self.assertTrue(frontier['candidate_or_available'])
        self.assertEqual({item['record_id'] for item in frontier['branches']},
                         {'atom:0_41:turn:1', 'atom:0_42:turn:1'})
        self.assertTrue(all(item['verification_state'] == CANDIDATE_ONLY
                            for item in frontier['branches']))

    def test_family_frontier_respects_limit_and_modality(self):
        records = [text_record('atom:0_41:turn:1'),
                   text_record('atom:0_42:turn:1'),
                   image_record('visual:' + 'a' * 64 + ':region:0', 'a' * 64)]
        descriptors = {record['record_id']: descriptor_from_record(
            record, {'a' * 64: 256}) for record in records}
        frontier = rank_source_family_frontier(
            records, [0.9, 0.8, 0.95], descriptors, limit=1,
            required_kind='source_text')
        self.assertEqual(frontier['source_family_count'], 1)
        self.assertFalse(frontier['candidate_or_available'])
        self.assertEqual(frontier['branches'][0]['record_id'], 'atom:0_41:turn:1')


if __name__ == '__main__':
    unittest.main()
