# Pinned MobileMem Evaluation Sources

`MANIFEST.json` identifies the MemBase commit and verifies each vendored file by Git blob SHA-1 and SHA-256. The active protocol sources live under `membase/`; the legacy top-level `llm_judge.py` was downloaded during an earlier investigation and is not used by any scoring process.

The scoring adapter executes unchanged normalization and TokenF1 functions from the verified source. The text judge prompt and single-/multiple-reference serialization are checked against the pinned operator. The local judge changes the reference model, so its accuracy is development diagnosis, not an official benchmark score. No external judge is invoked without explicit configuration.

Our effective-timestamp truncation also differs from the pinned dataset wrapper's full-history query timestamp. Report this as a temporal-causal MobileMem text protocol rather than an identical official leaderboard setting.
