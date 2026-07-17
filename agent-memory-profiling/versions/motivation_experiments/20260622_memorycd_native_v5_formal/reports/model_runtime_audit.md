# Model Runtime Audit

日期：2026-06-22

## Conda Environment

- Environment: `agent-memory-profiling`
- Python: `3.10.20`
- Torch: `2.1.0+cu121`
- Torch CUDA runtime: `12.1`
- Transformers: `4.51.3`
- NumPy: `1.26.4`
- Hugging Face Hub: installed (`0.36.2`)

## GPU Visibility

Plain sandbox execution:

```text
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
nvidia-smi: failed to communicate with NVIDIA driver
```

Authorized execution:

```text
torch.cuda.is_available(): True
torch.cuda.device_count(): 8
device0: NVIDIA GeForce RTX 3090
```

`nvidia-smi` under authorized execution reports 8 idle RTX 3090 GPUs with approximately 15 MiB used on each device.

## Fixed V5 Model Paths

- Dense retriever: `models/bge-base-en-v1.5`
- Cross-encoder: `models/cross-encoder-ms-marco-MiniLM-L6-v2`
- Local Qwen: `../dir` from the project root (`/data2/user25263093/userliteng/26_code/preTest/dir`)

## Config Hashes

```text
bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591  models/bge-base-en-v1.5/config.json
380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc  models/cross-encoder-ms-marco-MiniLM-L6-v2/config.json
8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a  ../dir/config.json
```

## Local Qwen Identifier

The local Qwen config reports:

```text
architectures: Qwen3ForCausalLM
model_type: qwen3
hidden_size: 2560
num_hidden_layers: 36
torch_dtype: bfloat16
max_position_embeddings: 40960
```

## Missing Runtime Dependency

`faiss` is not installed in the `agent-memory-profiling` environment:

```text
faiss False
```

V5 formal requires `faiss_flat` and a brute-force Top-R consistency audit. Do not run V5 formal under `ann_numpy` fallback.

