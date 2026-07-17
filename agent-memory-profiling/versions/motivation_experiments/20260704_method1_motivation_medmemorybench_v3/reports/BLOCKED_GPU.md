# GPU Runtime Note

Initial sandboxed checks could not see the GPU, but escalated execution confirmed working server GPUs. Smoke GPU runs were executed on `CUDA_VISIBLE_DEVICES=3`.

Observed in `conda run -n agent-memory-profiling python`:

```text
torch 2.1.0+cu121
cuda 12.1
torch.cuda.is_available() = False
CUDA_VISIBLE_DEVICES = None
torch.cuda.device_count() = 0
```

Observed from `nvidia-smi`:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

Per the v3 instruction, M1/M2/O1 inference, embeddings, reranking, controller calls, and latency measurements must use real GPU execution. Smoke runs now satisfy that requirement; formal runs remain pending.

Run formal commands only on a server session where `nvidia-smi` works:

```bash
conda activate agent-memory-profiling
export CUDA_VISIBLE_DEVICES=0
cd agent-memory-profiling/versions/20260704_method1_motivation_medmemorybench_v3

python scripts/run_m1_history_scale.py \
  --config configs/m1_history_scale.yaml \
  --split configs/query_split_medmemorybench_en_v1.json \
  --mode smoke

python scripts/run_m2_frontier_budget.py \
  --config configs/m2_budget.yaml \
  --split configs/query_split_medmemorybench_en_v1.json \
  --mode smoke

python scripts/run_o1_stage_trace.py \
  --config configs/o1_stage_locality.yaml \
  --split configs/query_split_medmemorybench_en_v1.json \
  --mode smoke
```
