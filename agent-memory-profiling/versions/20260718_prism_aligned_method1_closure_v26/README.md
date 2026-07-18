# Method 1 v26: OOF and Fresh-Holdout Closure

This directory is an additive continuation of v24/v25. It never reruns the
revealed v24 test split and never uses the v25 development split for v26
selection. Algorithm selection is restricted to strict five-fold train OOF.

The fresh-holdout path is fail closed: gold cannot be parsed until the selected
configuration, Git commit, dataset manifest, and single-read lock exist.

## Commands

```bash
conda run -n agent-memory-profiling python scripts/01_run_oof_frontier.py
conda run -n agent-memory-profiling pytest -q tests
```

Run commands from this version directory unless a script says otherwise.
