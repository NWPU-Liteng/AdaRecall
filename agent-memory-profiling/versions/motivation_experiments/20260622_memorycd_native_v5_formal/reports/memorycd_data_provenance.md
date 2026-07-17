# MemoryCD Data Provenance

日期：2026-06-22

## 官方代码仓库

- Repository: `https://github.com/AgentMemoryWorld/MemoryCD.git`
- Local path: `third_party/MemoryCD`
- Commit SHA: `4cee5ebbf8a1ac4c933bde147006217e04678370`
- Clone status: success

## 官方 Hugging Face 数据集

- Dataset: `WZDavid/MemoryCD`
- Expected local root: `data/raw/memorycd`
- Expected data file: `data/raw/memorycd/data/cross_domain.jsonl`
- Expected metadata files: `data/raw/memorycd/meta/meta_<domain>.jsonl.gz`
- Download status: failed
- Dataset revision / commit SHA: unavailable because the Hugging Face dataset metadata request did not complete.

## Download Attempts

1. Sandbox download:

```bash
conda run -n agent-memory-profiling huggingface-cli download WZDavid/MemoryCD \
  --repo-type dataset \
  --local-dir data/raw/memorycd
```

Result:

```text
NameResolutionError: Failed to resolve 'huggingface.co'
```

2. Authorized `huggingface-cli` download:

```bash
conda run -n agent-memory-profiling huggingface-cli download WZDavid/MemoryCD \
  --repo-type dataset \
  --local-dir data/raw/memorycd
```

Result:

```text
ConnectionResetError: [Errno 104] Connection reset by peer
LocalEntryNotFoundError: cannot find the appropriate snapshot folder on local disk
```

3. Authorized direct file probe:

```bash
curl -L --fail --head \
  https://huggingface.co/datasets/WZDavid/MemoryCD/resolve/main/data/cross_domain.jsonl
```

Result:

```text
curl: (35) Recv failure: connection reset by peer
```

4. Authorized Hugging Face dataset git clone:

```bash
git clone https://huggingface.co/datasets/WZDavid/MemoryCD \
  data/raw/memorycd_hf_git_probe
```

Result:

```text
gnutls_handshake() failed: Error in the pull function.
```

## Local File Inventory

No official MemoryCD data files were found locally under `data/raw/memorycd`, in the workspace, or in the searched user cache paths. The directory currently contains no `cross_domain.jsonl` and no `meta_*.jsonl.gz`.

## Compliance Decision

V5 formal execution is stopped at preflight. The guide explicitly requires official `WZDavid/MemoryCD` release files and forbids replacing them with self-built Amazon data or previous LoCoMo / MSC / LongMemEval artifacts. Since the official dataset files are unavailable in this environment, no MemoryCD task manifests, profiling traces, or full-task metrics were generated.

