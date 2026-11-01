import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_w12_bank_capacity as audit


class W12CapacityAuditTests(unittest.TestCase):
    def fixture(self):
        matrices = [
            {'name': f'm{index}', 'out_features': 8, 'in_features': 2560}
            for index in range(249)]
        pool = {
            'token_table_bytes': 1024,
            'grammar_state_bytes': 256,
            'scratch_bytes': 512,
            'total_bytes': 4096,
            'headroom_bytes': 0,
            'feasible': True,
        }
        return (
            {'decode_token_accounting': {
                'decoder_bank_residency': {'matrices': matrices},
                'pool_residency': pool}},
            {'decode_token_accounting': {'pool_residency': pool}})

    def test_w12_is_accounted_with_one_plane_and_alias_options(self):
        r0, r2 = self.fixture()
        result = audit.build_audit(r0, r2, capacity_bytes=64 * 1024**2)
        w12 = result['variants']['w12a16']
        plan = w12['decoder_plan']
        self.assertEqual(plan['weight_bits'], 12)
        self.assertEqual(plan['weight_copies'], 1)
        self.assertEqual(
            w12['pool_with_bf16_embedding_alias_bytes'],
            plan['total_bytes'] + 1024 + 256 + 512)
        self.assertEqual(
            w12['pool_with_bank_resident_embedding_row_fetch_bytes'],
            plan['total_bytes'] + 256 + 512)
        self.assertGreater(
            w12['pool_with_bf16_embedding_alias_bytes'],
            w12['pool_with_bank_resident_embedding_row_fetch_bytes'])

    def test_wrong_inventory_fails_closed(self):
        r0, r2 = self.fixture()
        r0['decode_token_accounting']['decoder_bank_residency']['matrices'].pop()
        with self.assertRaises(ValueError):
            audit.build_audit(r0, r2, capacity_bytes=64 * 1024**2)


if __name__ == '__main__':
    unittest.main()
