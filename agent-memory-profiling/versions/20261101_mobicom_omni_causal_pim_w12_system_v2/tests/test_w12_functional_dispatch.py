import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pim_backend.w12_functional_dispatch import W12FunctionalDispatch
from run_proof_mainline import (aggregate_w12_stage_dispatch,
                                resolve_reader_quantization)


class _Block(torch.nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.linear = torch.nn.Linear(width, hidden)

    def forward(self, value):
        return self.linear(value)


class _Reader(torch.nn.Module):
    def __init__(self, width=256, hidden=128, output=32):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            [_Block(width, hidden)])
        self.lm_head = torch.nn.Linear(hidden, output, bias=False)

    def forward(self, value, past_key_values=None):
        value = self.model.language_model.layers[0](value)
        return self.lm_head(value)


class W12FunctionalDispatchTests(unittest.TestCase):
    def reader(self):
        torch.manual_seed(23)
        return _Reader().to(torch.bfloat16).eval()

    def test_prefill_is_counted_fallback_and_decode_hits_bank(self):
        reader = self.reader()
        dispatch = W12FunctionalDispatch(
            reader, mode="replace", bank_count=4, tile_rows=16,
            group_size=128, release_source=False).__enter__()
        try:
            prefill = torch.randn(1, 3, 256, dtype=torch.bfloat16)
            decode = torch.randn(1, 1, 256, dtype=torch.bfloat16)
            reader(prefill)
            reader(decode, past_key_values=object())
            summary = dispatch.summary()
            self.assertEqual(summary["selected_modules"], 2)
            self.assertEqual(summary["bank_calls"], 2)
            self.assertEqual(summary["decode_fallback_calls"], 0)
            self.assertEqual(summary["prefill_fallback_calls"], 2)
            self.assertTrue(summary["all_selected_modules_executed"])
            self.assertFalse(summary["silent_fallback"])
            self.assertFalse(summary["physical_pim_executed"])
            self.assertGreater(summary["resident_bytes"], 0)
        finally:
            dispatch.__exit__(None, None, None)
        self.assertIsInstance(reader.lm_head, torch.nn.Linear)

    def test_shadow_records_bank_error_and_returns_host_contract(self):
        reader = self.reader()
        dispatch = W12FunctionalDispatch(
            reader, mode="shadow", bank_count=4, group_size=128,
            numeric_mode="fp32_bank", release_source=False).__enter__()
        try:
            activation = torch.randn(1, 1, 256, dtype=torch.bfloat16)
            reader(activation, past_key_values=object())
            summary = dispatch.summary()
            self.assertEqual(summary["bank_calls"], 2)
            self.assertEqual(summary["shadow"]["observations"], 2)
            self.assertIsNotNone(summary["shadow"]["max_relative_l2_error"])
            self.assertLess(summary["shadow"]["max_relative_l2_error"], 1e-2)
        finally:
            dispatch.__exit__(None, None, None)

    def test_decode_multirow_fallback_fails_closed(self):
        reader = self.reader()
        dispatch = W12FunctionalDispatch(
            reader, mode="replace", bank_count=4, group_size=128,
            release_source=False, fail_on_decode_fallback=True).__enter__()
        try:
            with self.assertRaises(RuntimeError):
                reader(torch.randn(1, 2, 256, dtype=torch.bfloat16),
                       past_key_values=object())
        finally:
            dispatch.__exit__(None, None, None)

    def test_counter_snapshots_form_stage_deltas_with_one_layout(self):
        reader = self.reader()
        dispatch = W12FunctionalDispatch(
            reader, mode="replace", bank_count=4, tile_rows=16,
            group_size=128, release_source=False).__enter__()
        try:
            before = dispatch.counter_snapshot()
            reader(torch.randn(1, 3, 256, dtype=torch.bfloat16))
            after_prefill = dispatch.counter_snapshot()
            reader(torch.randn(1, 1, 256, dtype=torch.bfloat16),
                   past_key_values=object())
            after_decode = dispatch.counter_snapshot()
            prefill = dispatch.counter_delta(before, after_prefill)
            decode = dispatch.counter_delta(after_prefill, after_decode)
            self.assertEqual(prefill["bank_calls"], 0)
            self.assertEqual(prefill["prefill_fallback_calls"], 2)
            self.assertEqual(decode["bank_calls"], 2)
            self.assertEqual(decode["decode_fallback_calls"], 0)
            self.assertTrue(decode["layout_identity_unchanged"])
            self.assertEqual(before["layout_instance_sha256"],
                             after_decode["layout_instance_sha256"])
        finally:
            dispatch.__exit__(None, None, None)

    def test_layout_spec_is_stable_across_equivalent_dispatchers(self):
        specs = []
        for _ in range(2):
            dispatch = W12FunctionalDispatch(
                self.reader(), mode="replace", bank_count=4, tile_rows=16,
                group_size=128, release_source=False).__enter__()
            try:
                specs.append(dispatch.counter_snapshot()["layout_spec_sha256"])
            finally:
                dispatch.__exit__(None, None, None)
        self.assertEqual(specs[0], specs[1])

        torch.manual_seed(24)
        changed = W12FunctionalDispatch(
            _Reader().to(torch.bfloat16).eval(), mode="replace", bank_count=4,
            tile_rows=16, group_size=128, release_source=False).__enter__()
        try:
            changed_spec = changed.counter_snapshot()["layout_spec_sha256"]
        finally:
            changed.__exit__(None, None, None)
        self.assertNotEqual(specs[0], changed_spec)

    def test_mainline_stage_aggregation_requires_all_three_stages(self):
        def stage(bank_calls, *, instance="layout", spec=None):
            return {
                "bank_calls": bank_calls,
                "decode_fallback_calls": 0,
                "prefill_fallback_calls": 2,
                "fallback_rows": 4,
                "bank_external_bytes": 100,
                "fallback_external_bytes": 200,
                "selected_modules": 2,
                "unique_layouts": 2,
                "layout_instance_sha256": instance,
                "layout_spec_sha256": spec,
                "layout_identity_unchanged": True,
            }

        rows = [{"w12_stage_dispatch": {
            "query_parser": stage(2),
            "source_grounding": stage(3),
            "reader_generation": stage(4),
        }}]
        summary = aggregate_w12_stage_dispatch(rows)
        self.assertTrue(summary["all_requests_all_stages_hit_bank"])
        self.assertTrue(summary["decode_fallback_zero"])
        self.assertEqual(summary["stages"]["reader_generation"]["bank_calls"], 4)
        self.assertFalse(summary["physical_pim_executed"])

        rows[0]["w12_stage_dispatch"]["source_grounding"]["bank_calls"] = 0
        summary = aggregate_w12_stage_dispatch(rows)
        self.assertFalse(summary["all_requests_all_stages_hit_bank"])

        rows = []
        for instance in ("process-a", "process-b"):
            rows.append({"w12_stage_dispatch": {
                "query_parser": stage(2, instance=instance, spec="layout-spec"),
                "source_grounding": stage(3, instance=instance,
                                          spec="layout-spec"),
                "reader_generation": stage(4, instance=instance,
                                           spec="layout-spec"),
            }})
        summary = aggregate_w12_stage_dispatch(rows)
        self.assertTrue(summary["single_layout"])
        self.assertEqual(summary["layout_instance_count"], 2)
        self.assertEqual(summary["layout_instance_sha256s"],
                         ["process-a", "process-b"])
        self.assertEqual(summary["layout_spec_sha256"], "layout-spec")
        self.assertTrue(summary["layout_spec_complete"])
        self.assertTrue(summary["resumed_processes"])

        rows[1]["w12_stage_dispatch"]["source_grounding"][
            "layout_spec_sha256"] = "other-layout-spec"
        with self.assertRaises(RuntimeError):
            aggregate_w12_stage_dispatch(rows)

        contextual = [{
            "proof_plan": {"grounding": {
                "stop_reason": "not_applicable_contextual"}},
            "w12_stage_dispatch": {
                "query_parser": stage(2),
                "source_grounding": stage(0),
                "reader_generation": stage(4),
            },
        }]
        summary = aggregate_w12_stage_dispatch(contextual)
        self.assertTrue(summary["all_requests_all_stages_hit_bank"])
        self.assertFalse(summary["all_requests_all_raw_stages_hit_bank"])
        self.assertEqual(
            summary["not_applicable_source_grounding_requests"], 1)

    def test_mainline_resolves_host_w12_control_without_enabling_bank_route(self):
        self.assertEqual(resolve_reader_quantization("none", None, "off"), "bf16")
        self.assertEqual(
            resolve_reader_quantization("none", "w12a16", "off"), "w12a16")
        self.assertEqual(
            resolve_reader_quantization("replace", None, "off"), "w12a16")
        with self.assertRaises(ValueError):
            resolve_reader_quantization("replace", "bf16", "off")
        with self.assertRaises(ValueError):
            resolve_reader_quantization("none", "w12a16", "replace")


if __name__ == '__main__':
    unittest.main()
