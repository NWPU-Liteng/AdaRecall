import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from project_omni_device_capacity import checkpoint_tensor_bytes, project


class DeviceCapacityTests(unittest.TestCase):
    def fixture(self):
        smoke = {'rows': [{'service_peak_cuda_allocated_bytes': 11}]}
        joint = {'reader_checkpoint_tensor_bytes': 100,
                 'observed_embedding_state_peak_bytes': 10,
                 'observed_visual_state_peak_bytes': 5,
                 'pim_pool_capacity_bytes': 50, 'known_joint_pool_fits': True,
                 'known_reservations': {'dense_fp32_index_bytes': 2, 'embedding': 10, 'visual': 5}}
        return smoke, joint

    def test_device_budget_separates_reserves_and_pim_subreservation(self):
        result = project(*self.fixture(), device_capacity_bytes=200,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)
        self.assertEqual(result['application_budget_bytes'], 180)
        self.assertEqual(result['payload_budget_after_runtime_reserve_bytes'], 160)
        self.assertEqual(result['known_unique_lower_bound_bytes'], 117)
        self.assertTrue(result['known_lower_bound_fits_application_budget'])
        self.assertTrue(result['pim_pool_is_subreservation_not_extra_device_memory'])
        self.assertEqual(result['projected_peak_with_declared_runtime_reserve_bytes'], 137)

    def test_runtime_is_charged_exactly_once_and_embedder_is_not_omitted(self):
        result = project(*self.fixture(), device_capacity_bytes=157,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)
        self.assertTrue(result['projected_peak_fits_application_budget'])
        result = project(*self.fixture(), device_capacity_bytes=157,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50,
                         embedder_tensor_bytes=1)
        self.assertFalse(result['projected_peak_fits_application_budget'])

    def test_full_store_budget_not_observed_peak_determines_fit(self):
        smoke, joint = self.fixture()
        joint['known_reservations']['visual'] = 50
        result = project(smoke, joint, device_capacity_bytes=157,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)
        self.assertFalse(result['projected_peak_fits_application_budget'])

    def test_pim_pool_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            project(*self.fixture(), device_capacity_bytes=200,
                    os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=51)

    def test_functional_layout_is_charged_without_double_counting_padding(self):
        smoke, joint = self.fixture()
        smoke.update(reader_quantization='w8a8', reader_integer_primary={
            'model_footprint_after_bytes': 65, 'integer_primary_bytes': 40,
            'retained_bf16_embedding_alias_bytes': 15, 'released_bf16_parameter_bytes': 60,
            'replaced_source_parameter_bytes': 75, 'source_weight_copies_retained_by_converter': False,
            'conversion_peak_cuda_allocated_bytes': 110},
            reader_functional_backend='pimnast_bank_layout_reference',
            pim_functional_startup={'layout_resident_bytes': 45,
                'functional_extra_cuda_bytes': 47,
                'declared_weight_pool_bytes': 50,
                'physical_primary_aliasing_verified': False},
            pim_functional_dispatch={'resident_layout_bytes': 45})
        joint['known_reservations']['additional_bank_layout_padding_bytes'] = 5
        result = project(*self.fixture(), device_capacity_bytes=150,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)
        self.assertEqual(result['functional_duplicate_layout_bytes'], 0)
        result = project(smoke, joint, device_capacity_bytes=150,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)
        self.assertEqual(result['functional_duplicate_layout_bytes'], 45)
        self.assertEqual(result['functional_declared_payload_increment_bytes'], 40)
        self.assertEqual(result['known_unique_lower_bound_bytes'], 65 + 17 + 45)
        self.assertEqual(result['declared_unique_payload_bytes'], 127)
        self.assertEqual(result['projected_peak_with_declared_runtime_reserve_bytes'], 147)
        self.assertFalse(result['pim_layout_alias_assumed'])
        self.assertFalse(result['projected_peak_fits_application_budget'])
        smoke['pim_functional_startup']['physical_primary_aliasing_verified'] = True
        with self.assertRaises(ValueError):
            project(smoke, joint, device_capacity_bytes=150,
                    os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)

    def test_actual_integer_primary_replaces_not_duplicates_checkpoint(self):
        smoke, joint = self.fixture()
        smoke.update(reader_quantization='w8a8', reader_integer_primary={
            'model_footprint_after_bytes': 65, 'integer_primary_bytes': 40,
            'retained_bf16_embedding_alias_bytes': 15, 'released_bf16_parameter_bytes': 60,
            'replaced_source_parameter_bytes': 75, 'source_weight_copies_retained_by_converter': False,
            'conversion_peak_cuda_allocated_bytes': 110})
        result = project(smoke, joint, device_capacity_bytes=150,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)
        self.assertEqual(result['reader_checkpoint_tensor_bytes'], 100)
        self.assertEqual(result['reader_resident_tensor_bytes'], 65)
        self.assertEqual(result['declared_unique_payload_bytes'], 82)
        self.assertEqual(result['observed_server_conversion_peak_cuda_allocated_bytes'], 110)
        self.assertFalse(result['startup_capacity_qualified'])
        joint['known_reservations']['hypothetical_int8_primary_bytes'] = 40
        with self.assertRaises(ValueError):
            project(smoke, joint, device_capacity_bytes=150,
                    os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)

    def test_integer_mode_without_conversion_inventory_fails_closed(self):
        smoke, joint = self.fixture()
        smoke['reader_quantization'] = 'w8a8'
        with self.assertRaises(ValueError):
            project(smoke, joint, device_capacity_bytes=200,
                    os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)

    def test_tensor_payload_excludes_headers_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'
            raw = json.dumps({'__metadata__': {'format': 'pt'},
                'weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(4))
            size, hashes = checkpoint_tensor_bytes(Path(directory))
            self.assertEqual(size, 4)
            self.assertEqual(len(hashes['model.safetensors']), 64)
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(3))
            with self.assertRaises(ValueError):
                checkpoint_tensor_bytes(Path(directory))

    def test_pim_capacity_does_not_create_free_device_memory(self):
        result = project(*self.fixture(), device_capacity_bytes=40,
                         os_reserve_bytes=10, runtime_reserve_bytes=0, pim_pool_bytes=50)
        self.assertFalse(result['pim_pool_fits_application_budget'])
        self.assertFalse(result['capacity_scenario_qualified_only_under_declared_reserves'])

    def test_negative_budget_is_not_reported_as_fit(self):
        result = project(*self.fixture(), device_capacity_bytes=100,
                         os_reserve_bytes=20, runtime_reserve_bytes=20, pim_pool_bytes=50)
        self.assertFalse(result['known_lower_bound_fits_application_budget'])
        self.assertFalse(result['projected_peak_fits_application_budget'])


if __name__ == '__main__':
    unittest.main()
