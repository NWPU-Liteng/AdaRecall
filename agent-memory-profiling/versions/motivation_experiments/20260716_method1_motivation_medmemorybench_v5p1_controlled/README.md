# Motivation v5.1 Controlled Supplement

This package replaces the non-comparable v5/v4 M1 history trend with a fixed-
cohort experiment. It does not modify Method 1, FSA, QESA, QHDS, or selector
experiments.

The controlled M1 cohort is frozen from the 240-query M2 workload. A data-only
audit found 81 queries feasible at every scale in `{128, 256, 512}` under both
`medical_only` and `medical_plus_official_noise`; all six query types remain
represented. The exact IDs and hash are in
`aggregate/v5p1_m1_common_query_ids.json`.

Run from the repository root:

```bash
conda run --no-capture-output -n agent-memory-profiling python \
  agent-memory-profiling/versions/20260716_method1_motivation_medmemorybench_v5p1_controlled/scripts/00_audit_m1_common_cohort.py

conda run --no-capture-output -n agent-memory-profiling python \
  agent-memory-profiling/versions/20260716_method1_motivation_medmemorybench_v5p1_controlled/scripts/01_run_m1_controlled.py \
  --config agent-memory-profiling/versions/20260716_method1_motivation_medmemorybench_v5p1_controlled/configs/m1_controlled.yaml \
  --mode formal

python agent-memory-profiling/versions/20260716_method1_motivation_medmemorybench_v5p1_controlled/scripts/02_build_v5p1_final_data.py
python agent-memory-profiling/versions/20260716_method1_motivation_medmemorybench_v5p1_controlled/scripts/04_make_v5p1_figures.py
python agent-memory-profiling/versions/20260716_method1_motivation_medmemorybench_v5p1_controlled/scripts/03_validate_v5p1.py
```

The single source of truth for the motivation section is
`aggregate/v5p1_motivation_final_data.csv`. The main M1 figure uses the
dimensionless `host_traffic_amplification_vs_128` metric; absolute bytes remain
only as a secondary audit field. The controlled invocation and summary files are retained alongside it. S1, M2, and M3 are reused from the
validated v5 alignment package; S2 is explicitly appendix-only.

If CUDA is unavailable, the runner refuses formal execution by default. An
explicit `--allow-cpu` produces an annotated CPU fallback in the manifest; its
latency must not be described as GPU latency.

The recommended single-axis figure set is under
`runs/v5p1_figures_revised/`; its data audit and required supplement plan are
documented in `reports/MOTIVATION_FIGURE_DATA_AUDIT_AND_SUPPLEMENT_PLAN.md`.
The compatibility command `python scripts/04_make_v5p1_figures.py` regenerates
that revised set and copies it to the package `figures/` directory.
