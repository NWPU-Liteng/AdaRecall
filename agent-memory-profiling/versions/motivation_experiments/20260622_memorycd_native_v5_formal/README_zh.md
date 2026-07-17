# MemoryCD Native V5 Formal Preflight

日期：2026-06-22

本目录按 `codex_memorycd_v5_rerun_guide.md` 创建，用于 V5 MemoryCD 原生任务重跑。当前状态不是正式实验结果，而是一次可复核的 preflight / provenance 记录。

## 当前状态

- 官方 MemoryCD 代码仓库已克隆到 `third_party/MemoryCD`。
- 官方仓库 commit：`4cee5ebbf8a1ac4c933bde147006217e04678370`。
- Hugging Face 官方数据集 `WZDavid/MemoryCD` 未能下载成功。
- 因缺少 `data/raw/memorycd/data/cross_domain.jsonl` 和 `data/raw/memorycd/meta/meta_<domain>.jsonl.gz`，未启动任何 V5 formal profiling 或 full task run。
- 未使用 LoCoMo、MSC、LongMemEval 或自建 Amazon Review 数据替代 MemoryCD。

## 阻塞项

1. 官方数据下载失败。
   - `huggingface-cli download WZDavid/MemoryCD --repo-type dataset --local-dir data/raw/memorycd` 在授权联网后仍因连接重置失败。
   - `curl` direct resolve URL 访问 `cross_domain.jsonl` 同样被连接重置。
   - `git clone https://huggingface.co/datasets/WZDavid/MemoryCD` 同样因 TLS/gnutls pull 失败。

2. 当前 conda 环境未安装 `faiss`。
   - V5 formal 要求 `faiss_flat` 作为 exact dense retrieval correctness reference。
   - 在安装 FAISS 前不能把 `ann_numpy` 或 brute force fallback 称为 V5 formal。

3. GPU 访问需要授权执行路径。
   - 普通沙箱内 `torch.cuda.is_available() == False`。
   - 授权执行下 `torch.cuda.is_available() == True`，可见 8 张 RTX 3090。

## 恢复步骤

在网络可访问 Hugging Face 或配置好代理后，从 `agent-memory-profiling` 目录执行：

```bash
conda run -n agent-memory-profiling huggingface-cli download WZDavid/MemoryCD \
  --repo-type dataset \
  --local-dir data/raw/memorycd
```

下载完成后必须确认：

```text
data/raw/memorycd/data/cross_domain.jsonl
data/raw/memorycd/meta/meta_Beauty_and_Personal_Care.jsonl.gz
data/raw/memorycd/meta/meta_Books.jsonl.gz
data/raw/memorycd/meta/meta_Electronics.jsonl.gz
data/raw/memorycd/meta/meta_Home_and_Kitchen.jsonl.gz
```

随后安装或切换到带 FAISS 的环境，并重新执行数据 inventory、protocol build、smoke test、profiling 和 full task run。

