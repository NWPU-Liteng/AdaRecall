import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from profile_qwen35_pim_candidates import group_for, load_tensor_headers, profile


class Qwen35PimCandidateTests(unittest.TestCase):
    def write_shard(self, model_dir, tensors, name="model.safetensors"):
        header = {}
        offset = 0
        for tensor_name, shape in tensors.items():
            elements = 1
            for dimension in shape:
                elements *= dimension
            header[tensor_name] = {"dtype": "BF16", "shape": shape,
                                   "data_offsets": [offset, offset + elements * 2]}
            offset += elements * 2
        encoded = json.dumps(header).encode()
        (model_dir / name).write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))

    def write_config(self, model_dir, tied=True):
        (model_dir / "config.json").write_text(json.dumps({
            "architectures": ["Test"], "tie_word_embeddings": tied,
            "text_config": {"layer_types": ["linear_attention", "full_attention"]}}))

    def test_profile_requires_real_safetensors_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(json.dumps({"architectures": ["Test"]}))
            with self.assertRaises(ValueError):
                profile(model_dir, activation_bytes=2, pim_weight_bytes=1)

    def test_mtp_and_visual_mlp_are_not_language_gemv_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            self.write_config(model_dir)
            self.write_shard(model_dir, {
                "model.language_model.layers.0.mlp.up_proj.weight": [4, 3],
                "model.visual.blocks.0.mlp.up_proj.weight": [4, 3],
                "mtp.layers.0.mlp.up_proj.weight": [4, 3]})
            result = profile(model_dir, activation_bytes=2, pim_weight_bytes=2)
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["weight_summary"]["language_mlp"]["bytes"], 24)
            self.assertEqual(result["weight_summary"]["auxiliary_mtp"]["bytes"], 24)
            self.assertEqual(result["total_tensor_bytes"], 72)

    def test_tied_output_projection_is_counted_once_without_lookup_gemv(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            self.write_config(model_dir)
            self.write_shard(model_dir, {"model.language_model.embed_tokens.weight": [8, 3]})
            result = profile(model_dir, activation_bytes=2, pim_weight_bytes=1)
            candidate = result["candidate_layers"][0]
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(candidate["runtime_role"], "output_projection")
            self.assertEqual(candidate["source_bytes"], 48)
            self.assertEqual(candidate["modeled_weight_bytes"], 24)
            self.assertEqual(candidate["activation_bytes"], 6)
            self.assertEqual(candidate["output_bytes"], 16)
            self.assertAlmostEqual(candidate["arithmetic_intensity_macs_per_byte"], 24 / 46)
            self.write_config(model_dir, tied=False)
            self.assertEqual(profile(model_dir, activation_bytes=2, pim_weight_bytes=1)["candidate_count"], 0)

    def test_nonmatrix_and_auxiliary_weights_remain_in_capacity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            self.write_config(model_dir)
            self.write_shard(model_dir, {
                "model.language_model.layers.0.linear_attn.A_log": [4],
                "model.language_model.layers.0.linear_attn.conv1d.weight": [4, 1, 2]})
            result = profile(model_dir, activation_bytes=2, pim_weight_bytes=2)
            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["total_tensor_bytes"], 24)

    def test_invalid_modeled_widths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                profile(Path(directory), activation_bytes=-1, pim_weight_bytes=2)
            with self.assertRaises(ValueError):
                profile(Path(directory), activation_bytes=2, pim_weight_bytes=0)

    def test_duplicate_tensors_across_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            self.write_shard(model_dir, {"weight": [2, 3]}, "one.safetensors")
            self.write_shard(model_dir, {"weight": [2, 3]}, "two.safetensors")
            with self.assertRaises(ValueError):
                load_tensor_headers(model_dir)

    def test_truncated_header_and_payload_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            path = model_dir / "model.safetensors"
            path.write_bytes(b"x")
            with self.assertRaises(ValueError):
                load_tensor_headers(model_dir)
            path.write_bytes(struct.pack("<Q", 999))
            with self.assertRaises(ValueError):
                load_tensor_headers(model_dir)
            self.write_shard(model_dir, {"weight": [2, 3]})
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                load_tensor_headers(model_dir)

    def test_grouping_priority(self):
        self.assertEqual(group_for("model.visual.blocks.0.mlp.up_proj.weight"), "vision")
        self.assertEqual(group_for("mtp.layers.0.self_attn.q_proj.weight"), "auxiliary_mtp")


if __name__ == "__main__":
    unittest.main()
