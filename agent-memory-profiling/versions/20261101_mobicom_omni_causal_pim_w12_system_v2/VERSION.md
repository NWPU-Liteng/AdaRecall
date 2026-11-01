# 20261031_mobicom_omni_causal_pim_w12_system_v1

Status: integrated system version; algorithm v3 plus the frozen W12 PIM backend.

Source snapshots:

- Algorithm and experiment harness: `20261030_mobicom_omni_causal_pim_v3_fullresume`.
- W12 functional backend and target-model closure: `20261019_mobicom_omni_causal_pim_v2`.

## Integrated Contract

- One run produces task success and latency evidence from the same Reader output.
- Query parser, source grounding and final Reader share one resident W12 decoder layout.
- Route is `replace`; numeric mode is exact `bf16_reference`.
- W12A16 uses group size 128, 256 banks, 16 tile rows and 4096 Host contract rows.
- Decode fallback is forbidden. Prefill remains an explicit Host fallback.
- Evaluation labels are read only by the incremental Judge, never by the Reader.
- Functional Torch wall time and target-model PIM projection are reported separately.

## Active Pilot

- Config: `configs/main_experiment_pilot112_w12.yaml`.
- Cohort: 8 users x 7 question types x 2 = 112 development/validation questions.
- Reader GPUs: `0,1,3,5`; Judge GPU: `6`; GPUs `2,4` are not used.
- Run root: `results/pilot112_w12_v1`.

## Commands

```bash
cd /data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20261031_mobicom_omni_causal_pim_w12_system_v1

CONFIG=configs/main_experiment_pilot112_w12.yaml bash scripts/start_full_experiment.sh
CONFIG=configs/main_experiment_pilot112_w12.yaml bash scripts/monitor_full_experiment.sh
CONFIG=configs/main_experiment_pilot112_w12.yaml bash scripts/monitor_full_experiment.sh --once
```

## Result Analysis

```bash
PYTHON=../../.venvs/flamr-qwen35/bin/python
BASELINE=../20261030_mobicom_omni_causal_pim_v3_fullresume/results/main_run_full_resume_v1

$PYTHON scripts/analyze_w12_system_run.py \
  --config configs/main_experiment_pilot112_w12.yaml \
  --baseline-reader-root "$BASELINE" \
  --baseline-judge-root "$BASELINE/judge_rows" \
  --pim-reference artifacts/pim_backend_w12_reference_20261030/closure_manifest.json \
  --output artifacts/analysis/pilot112_w12_v1/final_audit.json
```

The audit reports per-type, per-user and per-split success; paired Host-reference
deltas; W12 stage coverage; zero decode fallback; measured functional-emulation
wall time; and the frozen target-model PIM projection.

The pilot closure chain is:

```bash
bash scripts/run_pilot_closure_chain.sh
```

It waits for the W12 pilot, audits it, runs the same-algorithm Host BF16 control,
and writes the final paired audit. It does not promote directly to core840.

Design evidence: `reports/HYPERPIM_SC_W12_TILEALIGNED_CLOSURE_20261030_ZH.md`.

## Persistence

Reader checkpoints are atomically replaced after every question. Judge rows are
separate atomic files. Re-running the start command resumes both sides without
regenerating completed questions.

## Boundary

This is a simulation-backed architecture contribution. The quality score is from
the fixed local Qwen3-VL-8B Judge, not the official Qwen3-14B leaderboard score.
There is no physical PIM, RTL, phone, energy or thermal measurement.
