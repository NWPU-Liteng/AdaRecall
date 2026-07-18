# Development Gate

- run: `v24_development_20260718`
- backend: `qwen3`
- query rows: `290` queries / `870` method rows
- passed: `True`

The development run is the only place where this configuration may be assessed. The test split was not read by this script.

```json
{
  "backend": "qwen3",
  "candidate_gate_passed": true,
  "config_hash": "65d3bba1339cf63ef019b151bfe07b0cc72c1aedcc609f536ceb14bd5faf5fdd",
  "cost": {
    "candidate_reduction": 0.75,
    "dense_semantic_selection_ms": {
      "count": 290,
      "mean": 787.8371884580702,
      "median": 787.8513964824378
    },
    "full_semantic_selection_ms": {
      "count": 290,
      "mean": 3120.9261224532884,
      "median": 3115.1884575374424
    },
    "net_gain_ms": {
      "count": 290,
      "mean": 2328.1605583655746,
      "median": 2325.7022919133306
    },
    "ours_semantic_selection_ms": {
      "count": 290,
      "mean": 792.7655640877141,
      "median": 793.2406064355746
    }
  },
  "cost_passed": true,
  "deltas": {
    "candidate_recall_final": {
      "ours_minus_dense": {
        "count": 290,
        "mean": 0.0034482758620689655,
        "median": 0.0
      },
      "ours_minus_full": {
        "count": 290,
        "mean": 0.0,
        "median": 0.0
      }
    },
    "evidence_group_coverage_final": {
      "ours_minus_dense": {
        "count": 290,
        "mean": 0.006896551724137931,
        "median": 0.0
      },
      "ours_minus_full": {
        "count": 290,
        "mean": 0.00632183908045977,
        "median": 0.0
      }
    },
    "ndcg_at_10": {
      "ours_minus_dense": {
        "count": 290,
        "mean": 0.0014850915795634244,
        "median": 0.0
      },
      "ours_minus_full": {
        "count": 290,
        "mean": -0.002114300664708477,
        "median": 0.0
      }
    },
    "precision_at_k": {
      "ours_minus_dense": {
        "count": 290,
        "mean": 0.0006896551724137932,
        "median": 0.0
      },
      "ours_minus_full": {
        "count": 290,
        "mean": 0.0,
        "median": 0.0
      }
    }
  },
  "epsilon_dense": 0.005,
  "epsilon_full": 0.01,
  "full_non_inferior": true,
  "index_run_id": "v24_index_20260718",
  "passed": true,
  "positive_quality_signal": true,
  "quality_non_negative": true,
  "query_count": 290,
  "run_id": "v24_development_20260718",
  "safety_passed": true,
  "stage": "development",
  "test_read_status": "not_read"
}
```
