# Full integrated-system run with W12 PIM

This release is the complete MobileMem-Omni system snapshot: memory organization,
query contract, proof grounding, Reader generation and the W12 near-memory decoder
all execute in one path. Accuracy and latency therefore come from the same run.

## Scope

- Active pilot: `pilot112_v1`, 8 users x 7 types x 2 questions.
- Reader GPUs: `0,1,3,5`; Judge GPU: `6`.
- GPUs `2` and `4` are excluded by the user.
- W12 route: `replace`, numeric mode: `bf16_reference`.
- Decode fallback is disabled; prefill is intentionally Host-owned.
- Existing Host artifacts are used only as a paired reference, never seeded as new
  W12 answers.

## Start and monitor

```bash
cd /data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20261031_mobicom_omni_causal_pim_w12_system_v1

CONFIG=configs/main_experiment_pilot112_w12.yaml bash scripts/start_full_experiment.sh
CONFIG=configs/main_experiment_pilot112_w12.yaml bash scripts/monitor_full_experiment.sh
```

For one machine-readable snapshot:

```bash
CONFIG=configs/main_experiment_pilot112_w12.yaml bash scripts/monitor_full_experiment.sh --once
```

Do not omit `CONFIG`; the default config targets a different run root.

## Resume contract

- Reader checkpoints are replaced atomically after every completed question.
- Judge results are independent atomic files under `judge_rows/<user>/`.
- `run_full_resumable.py --seed-only` audits reusable rows before launch.
- Restarting the same command resumes both Reader and Judge.
- Complete Reader files are never overwritten by a resumed W12 run.

## Main artifacts

- `user_<id>_shard_0.partial.json`: atomic Reader progress.
- `user_<id>_shard_0.json`: complete Reader output.
- `judge_rows/<user>/<question>.json`: one atomic Judge result.
- `judge_status.json`: incremental Judge status.
- `monitor_snapshot.json`: operational monitor snapshot.
- `orchestrator_status.json`: user-to-GPU worker state.
- `artifacts/analysis/pilot112_w12_v1/`: quality, pairing and W12 execution audits.

## Reporting boundary

There are three distinct measurements and they must not be mixed:

1. Task success from the fixed local Qwen3-VL-8B Judge.
2. Measured Torch functional-bank wall time, which includes emulation overhead.
3. Target-model PIM service/TTFT projection from the frozen W12 closure.

The paper should report (1) for quality, (3) for PIM latency and bytes, and use (2)
only to audit execution coverage and functional exactness. The current work does not
claim physical silicon, RTL, phone, energy or thermal results.

## Promote to core840

`configs/main_experiment_core840_w12.yaml` is prepared with the frozen
`artifacts/subsets/core840_v1/cohort.json` and the same Reader/Judge GPU policy. Do
not launch it until the 112-question pilot completes with full W12 coverage, zero
decode fallback and a passed paired noninferiority gate.
