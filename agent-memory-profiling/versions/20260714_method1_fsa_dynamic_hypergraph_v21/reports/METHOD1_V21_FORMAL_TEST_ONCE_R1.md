# Method 1 v21 Formal Test Single Run

**Status: formal_test_single_run_ineligible_missing_frozen_test_embeddings.** The formal-test access was attempted exactly once after final freeze.

- formal-test runs recorded: `1`
- formal query-plan rows read: `290`
- missing frozen test embeddings: `290`
- calibration access count: `1`

Reason: the frozen query embedding index contains no med_test/formal_test vectors for the formal query IDs. No embedding, label, model, selector, hyperparameter, or dynamic policy was fabricated or changed after final freeze. This ineligible formal attempt is retained as a closure negative result; a second formal run is prohibited.
