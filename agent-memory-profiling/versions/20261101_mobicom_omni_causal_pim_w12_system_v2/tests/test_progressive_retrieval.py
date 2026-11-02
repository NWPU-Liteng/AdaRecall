import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.progressive_retrieval import (aggregate_retrieval_stats,
                                               candidate_order_v1,
                                               disabled_stats,
                                               overlap_at_k,
                                               progressive_candidate_order,
                                               quantized_candidate_order,
                                               select_dense_candidates)
from pim_backend.progressive_topk import quantize_plane_major


def _ranked(scores, mask, limit):
    indices = np.flatnonzero(mask)
    indices = indices[np.argsort(-scores[indices], kind='stable')]
    return indices[:limit]


class ProgressiveRetrievalTests(unittest.TestCase):
    def test_union_theorem_over_one_thousand_random_cases(self):
        rng = np.random.default_rng(101)
        for case in range(1000):
            rows = int(rng.integers(1, 260))
            scores = rng.normal(size=rows)
            allowed = rng.random(rows) < 0.82
            if not allowed.any():
                allowed[0] = True
            kind = rng.random(rows) < 0.38
            full = candidate_order_v1(
                scores, required_kind_mask=kind, row_mask=allowed)

            overall = _ranked(scores, allowed, 80)
            primary = _ranked(scores, allowed & kind, 32)
            union = np.unique(np.concatenate((overall, primary)))
            local = candidate_order_v1(
                scores[union], required_kind_mask=kind[union],
                row_mask=allowed[union])
            selected = union[local]

            if not np.array_equal(full, selected):
                self.fail(f'union theorem failed for random case {case}')

    def test_progressive_matches_quantized_full_selection(self):
        rng = np.random.default_rng(211)
        for seed in range(30):
            rows = int(rng.integers(60, 240))
            cols = int(rng.choice((128, 256, 1024)))
            weights = rng.normal(size=(rows, cols)).astype(np.float32)
            query = rng.normal(size=cols)
            query /= np.linalg.norm(query)
            matrix = quantize_plane_major(weights, bits=12, group_size=128)
            allowed = rng.random(rows) < 0.9
            if not allowed.any():
                allowed[0] = True
            kind = rng.random(rows) < 0.4
            use_kind = seed % 3 != 0
            required = kind if use_kind else None

            full_order, full_scores = quantized_candidate_order(
                matrix, query, required_kind_mask=required, row_mask=allowed)
            order, scores, stats = progressive_candidate_order(
                matrix, query, required_kind_mask=required, row_mask=allowed)

            with self.subTest(seed=seed, rows=rows, cols=cols):
                self.assertTrue(np.array_equal(full_order, order))
                self.assertTrue(np.array_equal(full_scores, scores))
                self.assertTrue(set(order.tolist()) <= set(np.flatnonzero(allowed)))
                self.assertLessEqual(
                    stats['weight_bytes_read'], stats['weight_bytes_full'])
                self.assertEqual(stats['codes_sha256'], matrix.codes_sha256)

    def test_overlap_and_invisible_rows(self):
        rng = np.random.default_rng(307)
        weights = rng.normal(size=(90, 128)).astype(np.float32)
        query = rng.normal(size=128)
        matrix = quantize_plane_major(weights)
        allowed = np.ones(90, dtype=bool)
        allowed[::7] = False
        float_scores = weights @ query
        float_order = candidate_order_v1(float_scores, row_mask=allowed)
        quantized_order, _, _ = progressive_candidate_order(
            matrix, query, row_mask=allowed)

        count, fraction = overlap_at_k(float_order, quantized_order)
        self.assertGreaterEqual(count, 0)
        self.assertLessEqual(count, 48)
        self.assertIsNotNone(fraction)
        self.assertFalse(set(quantized_order.tolist()) & set(np.flatnonzero(~allowed)))

    def test_default_selection_preserves_v1_scores_and_order(self):
        rng = np.random.default_rng(401)
        scores = rng.normal(size=180).astype(np.float32)
        kind = rng.random(180) < 0.4
        hidden = rng.random(180) < 0.15
        scores[hidden] = -np.inf

        expected = candidate_order_v1(
            scores, required_kind_mask=kind)
        order, values, stats = select_dense_candidates(
            scores, mode='off', required_kind_mask=kind)

        self.assertTrue(np.array_equal(order, expected))
        self.assertTrue(np.array_equal(values, scores[expected]))
        self.assertIsNone(stats)
        self.assertEqual(aggregate_retrieval_stats(
            'off', [], bits=12, group_size=128, codes_sha256=None),
            disabled_stats())

    def test_shadow_preserves_v1_selection_and_asserts_exactness(self):
        rng = np.random.default_rng(409)
        weights = rng.normal(size=(220, 256)).astype(np.float32)
        query = rng.normal(size=256)
        matrix = quantize_plane_major(weights)
        scores = weights @ query
        kind = rng.random(220) < 0.35
        allowed = rng.random(220) < 0.9

        expected = candidate_order_v1(
            scores, required_kind_mask=kind, row_mask=allowed)
        order, values, stats = select_dense_candidates(
            scores, mode='shadow', required_kind_mask=kind, row_mask=allowed,
            matrix=matrix, query=query)

        self.assertTrue(np.array_equal(order, expected))
        self.assertTrue(np.array_equal(values, scores[expected]))
        self.assertEqual(stats['exactness_assertion'], 'passed')
        self.assertTrue(stats['quantized_matches_full'])
        self.assertLessEqual(
            stats['weight_bytes_read'], stats['weight_bytes_full'])
        self.assertIsNotNone(stats['overlap_at_48'])

    def test_replace_uses_quantized_order_and_maps_filtered_rows(self):
        rng = np.random.default_rng(419)
        weights = rng.normal(size=(150, 128)).astype(np.float32)
        query = rng.normal(size=128)
        matrix = quantize_plane_major(weights)
        keep = rng.random(150) < 0.7
        keep[0] = True
        index_map = np.full(150, -1, dtype=np.int64)
        index_map[keep] = np.arange(int(keep.sum()), dtype=np.int64)
        current_weights = weights[keep]
        current_scores = current_weights @ query

        quantized, quantized_scores = quantized_candidate_order(
            matrix, query, row_mask=keep)
        expected = index_map[quantized]
        order, values, stats = select_dense_candidates(
            current_scores, mode='replace', matrix=matrix, query=query,
            progressive_row_mask=keep, index_map=index_map)

        self.assertTrue(np.array_equal(order, expected))
        self.assertTrue(np.array_equal(values, quantized_scores))
        self.assertTrue(set(order.tolist()) <= set(range(int(keep.sum()))))
        self.assertEqual(stats['exactness_assertion'], 'not_run_replace')
        self.assertIsNone(stats['quantized_matches_full'])

    def test_aggregate_reports_shadow_assertions_and_totals(self):
        demands = [{
            'rounds_total': 12,
            'weight_bytes_read': 30,
            'weight_bytes_full': 60,
            'overlap_at_48': 0.75,
            'exactness_assertion': 'passed',
        }, {
            'rounds_total': 8,
            'weight_bytes_read': 20,
            'weight_bytes_full': 40,
            'overlap_at_48': 0.5,
            'exactness_assertion': 'passed',
        }]
        summary = aggregate_retrieval_stats(
            'shadow', demands, bits=12, group_size=128, codes_sha256='a' * 64)
        self.assertTrue(summary['enabled'])
        self.assertEqual(summary['demand_count'], 2)
        self.assertEqual(summary['rounds_total'], 20)
        self.assertEqual(summary['weight_read_fraction'], 0.5)
        self.assertEqual(summary['mean_overlap_at_48'], 0.625)
        self.assertTrue(summary['shadow_assertions_passed'])
        self.assertFalse(summary['physical_pim_executed'])


if __name__ == '__main__':
    unittest.main()
