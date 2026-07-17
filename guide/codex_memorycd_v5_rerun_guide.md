# Codex 执行指导：V5 MemoryCD 原生数据替换与纠偏后的真实模型重跑

> **归档说明（2026-07-17）**：本文对应的 V4/V5 MemoryCD 动机实验已迁移到
> `agent-memory-profiling/versions/motivation_experiments/`。旧路径和命令保留为
> 历史协议记录，不应在已整理的 v21 目录中直接执行。

> **版本目的**：在不更换现有模型的前提下，以官方发布的 **MemoryCD** 替换不适合作为主压力集的 LongMemEval-S / MSC，并修正 V4 中会削弱论文说服力的运行设置。V5 的目标不是立刻实现超图或 PIM，而是先获得一套**原生任务、真实用户历史、真实模型、无人工 phase 强制调用、可复核且可作为后续方法基线**的正式结果。
>
> **本轮固定模型**：
>
> ```text
> Dense retriever : models/bge-base-en-v1.5
> Cross-encoder   : models/cross-encoder-ms-marco-MiniLM-L6-v2
> Task executor   : 现有 local_qwen backend 对应的本地 Qwen 模型
> ```
>
> **本轮不做**：
>
> - 不替换 BGE 或 cross-encoder；
> - 不引入更大的 reranker；
> - 不引入 synthetic query、synthetic memory、synthetic hard negative；
> - 不实现超图、PIM timing、row/bank layout 或复杂 learned controller；
> - 不将 MemoryCD 人为改造成 `planning / reasoning / output` 三阶段 Agent benchmark。
>
> **本轮要完成**：
>
> 1. 接入官方 MemoryCD release，并严格保留它的四个原生任务与官方数据切分；
> 2. 消除 V4 中人工 phase 调用、controller 额外开销、selection-only 口径混乱、旧人工 tag/relation expansion 等问题；
> 3. 在真实用户历史的自然 memory scale 上，使用现有 BGE + MiniLM + Qwen 重跑 retrieval、rerank 与最终任务质量；
> 4. 形成后续多视图超图 candidate selector 的可信 baseline 和 trace 数据。

---

## 0. 执行总原则（不可违反）

### 0.1 原生数据与原生任务优先

MemoryCD 的单条样本是一个真实用户的跨年、多领域 Amazon review interaction history；每条 interaction 含 `parent_asin`、`rating`、`title`、`text`、`timestamp`，而 ranking 任务另含官方提供的 `negative_parent_asins`。该 benchmark 原生定义了 **rating prediction、item ranking、review summarization、review generation** 四项任务。

V5 必须直接使用官方数据文件与官方任务语义，不能把 review history 改写成 LoCoMo 风格问答，也不能凭模板生成 fact / temporal / relation / plan-continuity query。

### 0.2 不将 MemoryCD 假装成 phase-aware Agent trace

MemoryCD 的核心是长期个性化与跨领域 user memory，而不是一个带 `planning / reasoning / output` 标注的真实 Agent trajectory。因此：

- **禁止**在主实验中固定跑两次或三次 retrieval；
- **禁止**在 query 后追加人工 phase suffix；
- **禁止**让 Qwen 先判断“planning 是否检索、reasoning 是否检索”后再执行相同检索；
- **禁止**把 task type 伪称为 Agent phase。

V5 的主路径应是一个真实、单次任务调用：

```text
原生 MemoryCD task input
      ↓
同一用户、满足 scope 与时间约束的真实历史 memory
      ↓
BGE dense retrieve
      ↓
MiniLM cross-encoder rerank（可选 baseline）
      ↓
Top-K memory context
      ↓
本地 Qwen 完成 MemoryCD 原生任务
```

后续论文可将 V5 表述为：

> **Real-user lifelong memory selection benchmark**。

而不是 phase-aware agent benchmark。V4 的 LoCoMo phase profiling 仍可单独保留为受控 motivation，不得与 V5 的结果混合计账。

### 0.3 模型不变，但运行方式必须更真实

保留已有 BGE、MiniLM 和本地 Qwen；修改的是错误或不合理的**实验控制变量**：

- 移除主路径的 Qwen retrieval controller；
- 去除固定三阶段或当前实际两阶段的重复检索；
- 关闭旧的 synthetic task/entity tag filter；
- 关闭旧的 relation expansion；
- 使用 query-local、tenant-local、time-valid 的真实 memory scope；
- 主结果采用 query-local cross-encoder mini-batch，而不是对候选对逐条串行 `batch_size=1`；
- `selection_only` 与 full task run 严格分离；
- 使用明确的冷/热启动与在线/离线记账口径。

---

## 1. 现有 V4 需要修正的设置

现有 V4 可保留为“真实模型的 selection-path profiling”，但存在以下不能直接带入 V5 的问题。

| V4 现象 | 为什么不适合 MemoryCD 主实验 | V5 强制修复 |
|---|---|---|
| `shared_once / fixed_three / adaptive_phase` 被用作三种主要模式 | MemoryCD 不具备原生 Agent phase；固定触发会人为放大检索 | 删除这些模式；主路径每个原生 task event 只执行一次 selection |
| adaptive controller 在 LoCoMo / LongMemEval-S / MSC 上平均仍接近两次 retrieval，且自身 LLM 代价高 | controller 没有形成有效 gate，只是在 selection 前增加 LLM 调用 | controller 不进入 V5 主路径；如未来研究 adaptive trigger，单独作为后续 ablation |
| `selection_only=true` 跳过 output phase，却将结果以 total ms 解读 | selection latency 与完整任务响应延迟不是同一概念 | 明确拆分 profiling run 与 full task run；只有 full task run 报 task end-to-end latency |
| `reranker_batch_size=1` | 将同一 query 的候选对强制串行处理，容易把 rerank 代价夸大为实现工件 | 主结果使用 query-local mini-batch；batch=1 仅保留为 microbenchmark 附录 |
| 旧 `QuerySpec`、人工 task/entity tags、metadata filter | 不是 MemoryCD 原生在线输入，可能构成隐性先验 | 只允许 user id、domain、timestamp、原始 interaction 文本、原始 target item / candidate items 作为 runtime 输入 |
| relation expansion | MemoryCD 没有原生 memory relation graph；继续使用会引入人为结构 | V5 完全关闭；未来只能由过去真实 trace 学习得到 |
| `ann_numpy` 可能 fallback 到全扫描 | 延迟容易受仓库自定义实现影响，难以说明为 selection 算法本身现象 | 以 `faiss_flat` exact dense retrieval 作为 V5 正式 correctness reference；保留 `ann_numpy` 只做历史兼容检查，不进入主表 |
| 对 LongMemEval-S / MSC 试图制造大范围 candidate pool | 会破坏其原始 query scope 或 tenant 隔离 | V5 的主规模压力由 MemoryCD 每个真实用户的长历史自然提供，不做复制或跨用户混池 |

---

## 2. MemoryCD 官方资产与数据获取

### 2.1 必须使用官方 release，不得自行从原始 Amazon Reviews 重建后仍称为 MemoryCD

先克隆官方代码与下载官方 dataset。官方 repository 已明确发布统一 `cross_domain.jsonl` 与每领域 item metadata；其 task split、ranking negatives、prompt 语义与 evaluator 都应视为基准定义的一部分。

```bash
export REPO_ROOT=/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling
cd "$REPO_ROOT"

mkdir -p third_party data/raw/memorycd

git clone https://github.com/AgentMemoryWorld/MemoryCD.git third_party/MemoryCD

# 在当前实验环境安装仅下载所需依赖；不要覆盖已有 torch / transformers。
python -m pip install -U huggingface_hub

huggingface-cli download WZDavid/MemoryCD \
  --repo-type dataset \
  --local-dir data/raw/memorycd
```

下载后必须确认至少存在：

```text
data/raw/memorycd/data/cross_domain.jsonl
data/raw/memorycd/meta/meta_<domain>.jsonl.gz
```

实际目录结构以下载结果为准。若 dataset 卡目录结构不同，写入一份 path mapping，而不是复制或改写数据。

### 2.2 下载与来源核验产物

新增：

```text
reports/memorycd_data_provenance.md
reports/memorycd_dataset_inventory.json
```

其中必须包含：

- 官方 repository commit SHA；
- Hugging Face dataset revision / commit SHA；
- 原始数据文件相对路径、字节数、SHA256；
- 用户数、领域数、各领域 interaction 数、时间跨度；
- `cross_domain.jsonl` 的 schema 检查结果；
- metadata 文件发现情况；
- 任何无法下载或字段缺失的错误信息。

禁止悄悄切换到自建 Amazon Review 数据。若官方 release 不完整或下载失败，直接停止并在报告中明确失败原因。

---

## 3. V5 新目录、命名和保留原则

不得覆盖当前 V4 正式结果：

```text
versions/20260621_real_trace_adaptive_phase_v4_formal/
```

创建新的正式版本：

```text
versions/20260622_memorycd_native_v5_formal/
```

建议目录：

```text
versions/20260622_memorycd_native_v5_formal/
├── README_zh.md
├── manifests/
│   ├── memorycd_official_single/
│   ├── memorycd_official_cross/
│   └── memorycd_causal_cross/
├── configs_snapshot/
├── runs/
│   ├── profiling/
│   └── full_task/
├── reports/
│   ├── memorycd_data_provenance.md
│   ├── memorycd_dataset_inventory.json
│   ├── conversion_parity_report.md
│   ├── leakage_audit.md
│   ├── tenant_scope_audit.md
│   ├── time_scope_audit.md
│   ├── model_runtime_audit.md
│   ├── profiling_report.md
│   └── full_task_report.md
└── logs/
```

新增代码优先放在：

```text
experiments/memorycd/
configs/memorycd_v5/
src/memorycd/
```

不要污染旧 LoCoMo / LongMemEval / MSC importer；如需复用公共接口，仅抽取可复用函数。

---

## 4. V5 数据模型与严格隔离规则

### 4.1 Runtime memory item

每条真实历史 interaction 转为一个原子 memory item：

```json
{
  "memory_id": "<user_id>|<domain>|<timestamp>|<parent_asin>|<ordinal>",
  "tenant_id": "<user_id>",
  "domain": "<domain>",
  "timestamp": 1700000000,
  "parent_asin": "...",
  "rating": 5,
  "review_title": "...",
  "review_text": "...",
  "memory_text": "[Domain] ...\n[Item] ...\n[Rating] 5/5\n[Review Title] ...\n[Review] ...",
  "source": "memorycd_official",
  "metadata_source": "official_interaction+official_item_meta"
}
```

规则：

- `memory_text` 仅由官方 interaction 和官方 item metadata 的字段直接格式化得到；
- 不允许 Qwen 为 memory 文本生成摘要、标签、实体或关系；
- `ordinal` 只用于保证 ID 唯一，不能影响 ranking；
- memory embedding 可离线用现有 BGE 真实计算并缓存；
- offline embedding 不计入 query online latency，但必须单列 report ingestion / embedding build cost。

### 4.2 Runtime task event

每个任务 event 的 runtime 字段必须与 evaluator label 分开存储：

```json
{
  "event_id": "...",
  "tenant_id": "<user_id>",
  "setting": "official_single | official_cross | causal_cross",
  "task": "rating_prediction | item_ranking | review_summarization | review_generation",
  "target_domain": "...",
  "source_domains": ["..."],
  "target_timestamp": 1700000000,
  "query_text": "<严格按官方 task prompt 的 target-side 输入构造，且不含隐藏 label>",
  "candidate_item_metadata": ["..."],
  "runtime_memory_ids": ["..."],
  "runtime_memory_scope_size": 0
}
```

`evaluation_labels.jsonl` 另存：

```json
{
  "event_id": "...",
  "gold_rating": 5,
  "positive_parent_asin": "...",
  "gold_review_title": "...",
  "gold_review_text": "..."
}
```

**在线 selector、BGE、MiniLM、Qwen prompt builder、faiss index builder 一律不得读取 `evaluation_labels.jsonl`。**

### 4.3 四项任务的输入泄漏边界

| 任务 | Runtime 可见 target 输入 | Runtime 不可见 label / 信息 |
|---|---|---|
| rating prediction | target item metadata；用户历史 | target rating、target review title、target review text |
| item ranking | target candidate item metadata 列表；用户历史 | 哪个 candidate 为 positive；任何 ranking label |
| review summarization | target item metadata、目标 rating、目标完整 review text；用户历史 | gold review title |
| review generation | target item metadata、目标 rating；用户历史 | gold review title、gold review text |

对 review summarization，任务本身将目标 full review 作为输入，历史 memory 主要用于风格/偏好辅助。因此它可以保留作为 **MemoryCD task coverage**，但不得把它作为“evidence retrieval 最困难”或“rerank bottleneck 最强”的主结论。主分析应聚焦 `rating_prediction` 与 `item_ranking`，其次是 `review_generation`。

---

## 5. 两种评测协议：必须显式区分，不可混报

### 5.1 Protocol O：官方复现协议（主可比结果）

严格遵照官方 MemoryCD repository 的切分逻辑：

- **single-domain**：目标 domain 的最后 `num_test=3` 条 interaction 为 test；此前同 domain interaction 为 memory；
- **cross-domain**：memory 仅来自官方指定 source domain(s)，test 来自 target domain；
- ranking 的 positive 与 negative candidates 使用官方 release 字段；
- task prompt 语义、输出格式、metrics 对齐官方 evaluator；
- 该协议的所有输出命名带 `official_`。

在开始自定义转换前，先运行官方 repository 的小样本命令作为 parser / schema sanity check：

```bash
cd "$REPO_ROOT/third_party/MemoryCD"

python evaluation_all_4task.py \
  --task rating_prediction \
  --domain Books \
  --input "$REPO_ROOT/data/raw/memorycd/data/cross_domain.jsonl" \
  --method rag \
  --max-memory-items 10 \
  --max-users 5
```

该命令不替代你的正式系统，只用于验证原始数据、官方 split 与 task prompt 可以被正常读取。

### 5.2 Protocol C：严格时间因果协议（系统主分析辅助结果）

为支持“长期在线 memory”叙事，额外构造严格时间版本：

```text
runtime memory item is eligible iff
    memory.tenant_id == event.tenant_id
    AND memory.timestamp < event.target_timestamp
    AND memory.domain ∈ allowed_source_domains(setting)
```

对于 single-domain，通常与官方最后若干条测试切分接近；对于 cross-domain，它可能比官方 protocol 更严格，因为会排除 timestamp 晚于 target 的 source-domain interactions。

规则：

- **Protocol C 不得被标为官方复现分数**；
- **Protocol O 与 Protocol C 不得放进同一张平均表后混合比较**；
- V5 README、每个 run manifest 和图标题必须显式写出 `protocol=official` 或 `protocol=causal`。

### 5.3 本轮结果优先级

1. `official_single`：主任务质量与可比性；
2. `official_cross`：跨领域 memory 的可比性；
3. `causal_cross`：严格长期 online memory 解释与后续 trace 基础。

---

## 6. 运行模型与新的主路径

### 6.1 固定模型配置

```yaml
embedding:
  backend: transformer
  model_path: models/bge-base-en-v1.5
  dtype: bf16_or_fp16_as_supported
  max_length: 512
  normalize: true

retrieval:
  backend: faiss_flat
  metric: inner_product
  precomputed_memory_embeddings: true

reranker:
  backend: cross_encoder
  model_path: models/cross-encoder-ms-marco-MiniLM-L6-v2
  max_length: 512
  pair_batch_size: 16
  sort_descending: true

generation:
  backend: local_qwen
  model_identifier: <从当前V4 run_manifest中读取并写入>
  temperature: 0.0
  seed: 20260622
```

约束：

- 不下载或替换为 BGE-large、BGE-M3、bge-reranker、其他 cross-encoder；
- 若模型目录缺失，停止并报错，不得用公网其他模型悄悄替代；
- `pair_batch_size=16` 是单一请求内候选 pair 的 micro-batch，不是跨用户 request batching；
- 若显存不足，按 `16 → 8 → 4` 逐级回退，且将实际 batch size 写入每条 trace；
- `batch_size=1` 只允许用于附录 microbenchmark，不允许作为主系统 latency。

### 6.2 为什么使用 `faiss_flat`

V5 当前目的不是比较 ANN 系统，而是得到可靠的 rerank frontier 和候选质量基线。对单用户自然 history 的 query-local retrieval，exact flat inner-product search 可以避免 `ann_numpy` fallback 行为、probe 参数与自定义索引实现混杂结果。

实现规则：

- 单个 event 的 index / candidate scope 只能含该 tenant 的合法历史 memory；
- 若 `faiss` 不可用，安装与当前 Python / CUDA 环境兼容的 FAISS 包；
- 不得降级回 `ann_numpy` 并仍称为 V5 formal；
- 额外用一小组 event 进行 `faiss_flat` 与 brute-force dot product Top-R 一致性检查，要求候选 ID 完全一致。

### 6.3 新主路径：一次 selection，一次任务执行

```text
1. 构造原生 target task prompt（不含隐藏标签）
2. 按 tenant / setting / timestamp 构造合法 memory scope
3. BGE 编码 query
4. FAISS exact Top-R dense retrieve
5. 可选 MiniLM 对 Top-R 成对打分并重排
6. 选取 Top-K memory text 进入官方风格 Qwen task prompt
7. Qwen 输出原生任务 prediction / generation
8. evaluator 从独立 labels 文件计算指标
```

禁止：

- controller decide-retrieve；
- pre-run shared selection；
- planning / reasoning / output suffix；
- synthetic metadata tags；
- relation graph expansion；
- oracle label pruning。

---

## 7. V5 需要实现的三条简单基线

V5 只建立必要且公平的基础线，不进入复杂结构算法。

| ID | 方法 | 目的 | 是否用 reranker |
|---|---|---|---|
| `recency_k8` | 合法 scope 中 timestamp 最近的 K=8 条 interaction | 对齐 MemoryCD 官方 long-context / recency 类选择思想 | 否 |
| `dense_k8` | BGE exact dense Top-8 | 观察 dense retrieval 本身能提供多少效果 | 否 |
| `dense_rerank_r128_k8` | BGE Top-128 → MiniLM rerank → Top-8 | V5 主 baseline；为后续结构化 pre-rerank selection 提供参照 | 是 |

额外仅用于 profiling 的敏感性：

```text
R ∈ {32, 64, 128}
K = 8
```

即：

```text
profile_dense_rerank_r32_k8
profile_dense_rerank_r64_k8
profile_dense_rerank_r128_k8
```

不运行 `R=256` 或更大值来人为扩大 rerank 时间；先让 MemoryCD 的自然 user history scale 决定实际压力。

### 7.1 公平性约束

- 所有方法使用完全相同的合法 memory scope；
- 所有 methods 的最终 Qwen context 数量均为 K=8；
- 排名任务使用完全相同的 candidate item list；
- same task / same event / same Qwen decoding configuration；
- 仅 `dense_rerank_*` 调用 MiniLM；
- `recency_k8` 不是 no-memory baseline；必要时另增加 `no_memory`，但只用于 task quality 下界，不能参与 latency 主比较；
- 不在 V5 使用 heuristic rerank，因为它的金标 bonus / task tag 逻辑不属于 MemoryCD 原生输入。

---

## 8. 数据转换实现要求

### 8.1 新增模块

```text
src/memorycd/
├── __init__.py
├── schema.py
├── loader.py
├── item_metadata.py
├── protocol_builder.py
├── prompt_builder.py
├── faiss_store.py
├── task_executor.py
├── evaluator.py
└── audit.py

experiments/memorycd/
├── inspect_memorycd_release.py
├── build_memorycd_manifests.py
├── validate_memorycd_protocols.py
├── profile_memorycd_selection.py
├── run_memorycd_full_tasks.py
├── audit_memorycd_runs.py
└── summarize_memorycd_v5.py

configs/memorycd_v5/
├── official_single.yaml
├── official_cross.yaml
├── causal_cross.yaml
├── profiling_r32.yaml
├── profiling_r64.yaml
├── profiling_r128.yaml
└── full_task_r128_k8.yaml
```

### 8.2 不修改已有通用组件的语义

可复用：

```text
src/embedding_client.py
src/reranker_client.py
```

但要新增清晰的 adapter，不要把 MemoryCD task 伪塞进旧 `QuerySpec` 的 `fact / temporal / relation / plan_continuity / constraint` 枚举中。

允许给 `reranker_client.py` 加入 batch scoring 接口，例如：

```python
score_pairs(query: str, documents: list[str], batch_size: int) -> list[float]
```

但不得改变旧 V3/V4 run 的行为；必要时新增 `score_pairs_batched()`。

### 8.3 Prompt parity

`prompt_builder.py` 必须以官方 MemoryCD prompt 为语义来源：

- rating：只输出 `1..5` 单个整数；
- ranking：只输出 candidate indices 的排序列表；
- summarization：只输出 3–10 words review title；
- generation：只输出 review body。

可以将 memory context 从官方 `long_context` / BM25 RAG 的结果替换为 V5 的 selected memories；但**不得修改 target 输入、输出格式或 hidden-label 规则**。

每个 event trace 保存：

```text
prompt_template_version
selected_memory_ids
selected_memory_token_counts
full_prompt_token_count
raw_qwen_output
parsed_prediction
```

---

## 9. 必做审计：任何一项失败，正式 run 无效

### 9.1 Tenant isolation audit

对每个 event：

```text
all(selected_memory.tenant_id == event.tenant_id)
```

要求：100%。

### 9.2 Time scope audit

对 `causal_*` event：

```text
all(selected_memory.timestamp < event.target_timestamp)
```

要求：100%。

对 `official_*` event，不强制上述条件，但必须报告：

```text
future_memory_count
future_memory_ratio
max_future_seconds
```

不得隐藏官方 protocol 与 causal protocol 的差异。

### 9.3 Label leakage audit

扫描 runtime manifest、memory bank、query prompt 和 candidate selector input，确认不含以下内容：

- `gold_rating` / target rating（rating prediction）；
- positive candidate ID 或 ranking label；
- `gold_review_title`；
- `gold_review_text`（review generation）；
- evaluator metric 或 official result；
- 从 evaluation labels 派生的字段。

`review_summarization` 的 target review text 和 target rating 属于官方 task input，audit 应做 task-specific allowlist，不能误报。

### 9.4 Dense correctness audit

随机抽取每个 protocol / task / domain 组合至少 20 events，比较：

```text
faiss_flat Top-R IDs == brute-force normalized inner-product Top-R IDs
```

要求：完全一致。

### 9.5 Model runtime audit

每个正式 run 的 `run_manifest.json` 必须保存：

```text
embedding model path + config hash
reranker model path + config hash
Qwen backend/model identifier
CUDA / torch / transformers versions
dtype
reranker pair batch size
max sequence length
GPU name
seed
```

---

## 10. 运行阶段与顺序

### Phase 0：代码与数据 smoke test（不计正式结果）

目的：确认 converter、protocol、模型调用和 evaluator 正确。

```text
users: 每个 setting / task / domain 5 个
num_test: 1
R: 32
K: 8
methods: recency_k8, dense_k8, dense_rerank_r32_k8
```

验收：

- 所有任务都能解析输出；
- ranking output 能被解析为合法候选 index permutation；
- audit 全通过；
- all selected memory IDs 属于合法 tenant / scope；
- `faiss_flat` 与 brute force 完全一致；
- no unexpected empty history；
- GPU 计时字段非空。

### Phase 1：数据分布 census 与正式 domain 选择

运行：

```bash
python experiments/memorycd/inspect_memorycd_release.py \
  --input data/raw/memorycd/data/cross_domain.jsonl \
  --meta-dir data/raw/memorycd/meta \
  --output versions/20260622_memorycd_native_v5_formal/reports/memorycd_dataset_inventory.json
```

统计：

- 每领域用户数；
- 每用户 memory 数的 p50/p90/p95/p99/max；
- time span；
- 单领域与跨领域可用 user-target count；
- 每 task 的可用 event 数；
- strict causal filter 后剩余 scope 分布。

正式 domain 选择规则：

1. 优先使用官方 README 中明确支持且实际存在的四个 domain 的交集：

```text
Books
Electronics
Home_and_Kitchen
Beauty_and_Personal_Care
```

2. 从中选择自然 history p95 最大、且至少有足够 user-target event 的两个 domain 作为 `PRIMARY_SINGLE_DOMAINS`；
3. 若 `Home_and_Kitchen` 存在且可用，优先作为 `PRIMARY_CROSS_TARGET_DOMAIN`；其余三个作为 source domains；
4. 所有选择由 census script 自动写入：

```text
reports/domain_selection.json
```

不得手工挑选“最慢”或“结果最好”的 domain。

### Phase 2：Selection-path profiling（真实数据 + 真实 BGE/MiniLM；不运行 Qwen）

目的：得到未被 controller、answer generation 或 artificial phases 污染的 selection latency 分解。

运行矩阵：

```text
Protocols: official_single, official_cross, causal_cross
Tasks: rating_prediction, item_ranking
R: 32, 64, 128
K: 8
Methods: dense_k8, dense_rerank_r{R}_k8
Users: 全部 eligible users；若资源确实不足，采用固定 seed 的分层样本，不能使用文件前 N 个用户
```

每条 trace 记录：

```json
{
  "event_id": "...",
  "tenant_id": "...",
  "protocol": "official_cross",
  "task": "item_ranking",
  "memory_scope_size": 0,
  "selected_memory_ids": [],
  "retrieval_top_r": 128,
  "final_context_k": 8,
  "query_encode_ms": 0.0,
  "faiss_search_ms": 0.0,
  "rerank_tokenize_ms": 0.0,
  "rerank_model_ms": 0.0,
  "rerank_sort_ms": 0.0,
  "context_build_ms": 0.0,
  "selection_wall_ms": 0.0,
  "rerank_pair_count": 0,
  "rerank_pair_batch_size": 16,
  "avg_rerank_input_tokens": 0.0,
  "selected_context_tokens": 0,
  "logical_selected_payload_bytes": 0
}
```

计时规则：

- CUDA 计时需 `torch.cuda.synchronize()` 或 CUDA event，避免异步误差；
- `rerank_model_ms` 是 MiniLM forward compute；
- `rerank_tokenize_ms` 单列；
- `selection_wall_ms` 为 query encode 到 context assembled 的真实 wall time；
- 不包含 offline memory embedding / index build；这些另报 ingestion cost；
- 不写 `total_response_latency`，因为此 phase 不运行 Qwen task output。

### Phase 3：完整原生任务运行（正式质量 + 端到端）

目的：测量 candidate selection 在真实 MemoryCD 原生任务上的质量与实际 task end-to-end latency。

正式方法：

```text
no_memory                  # 仅 task quality 下界，可选但推荐
recency_k8
 dense_k8
dense_rerank_r128_k8       # 主 baseline
```

正式配置：

```text
Protocols: official_single, official_cross, causal_cross
Tasks: rating_prediction, item_ranking, review_generation, review_summarization
K: 8
R: 128（仅 dense_rerank 方法）
Qwen: current local_qwen, temperature=0, fixed seed
Users: 全部 eligible user-target events
```

分层执行顺序：

1. `rating_prediction` + `item_ranking` 先完成全部正式 event；
2. `review_generation` 运行全部 eligible event；
3. `review_summarization` 运行全部 eligible event，但报告中作为 task coverage / personalization generation 结果，不用于核心 rerank bottleneck 结论；
4. 任何 task 若输出格式解析失败，记录 `parse_error` 并按官方定义计入失败，不允许人工修正输出。

每条 full trace 额外记录：

```text
qwen_prompt_build_ms
qwen_generation_ms
qwen_output_tokens
task_end_to_end_wall_ms
raw_model_output
parsed_prediction
```

口径：

```text
selection_latency_ms = query_encode + retrieve + rerank + context build
qwen_task_latency_ms = prompt build + local Qwen inference
end_to_end_task_latency_ms = selection wall + qwen task wall
```

V5 不存在 controller latency；不得保留 `controller_ms` 空字段并参与平均。

### Phase 4：正式总结与图表

运行：

```bash
python experiments/memorycd/summarize_memorycd_v5.py \
  --version-dir versions/20260622_memorycd_native_v5_formal
```

生成：

```text
reports/profiling_report.md
reports/full_task_report.md
reports/statistical_tests.md
figures/
```

---

## 11. 评价指标与论文口径

### 11.1 MemoryCD 官方 task quality metrics

| 任务 | 主指标 | 辅指标 |
|---|---|---|
| rating prediction | MAE（越低越好） | RMSE |
| item ranking | NDCG@1 / NDCG@3 | Recall@1 / 3 / 5 |
| review summarization | ROUGE-L | ROUGE-1、BLEU-1/4、BERTScore |
| review generation | ROUGE-L | ROUGE-1、BLEU-1/4、BERTScore |

使用与官方 repo 一致或语义等价的 evaluator，版本写入 manifest。

### 11.2 Selection efficiency metrics

必须报告：

```text
memory_scope_size
retrieval_top_r
rerank_pair_count
final_context_k
query_encode_ms
faiss_search_ms
rerank_tokenize_ms
rerank_model_ms
selection_wall_ms
logical_selected_payload_bytes
selected_context_tokens
p50 / p95 / p99 selection wall latency
```

对于 full task run 再报告：

```text
qwen_generation_ms
task_end_to_end_wall_ms
selection_to_e2e_ratio
```

注意：

- `logical_selected_payload_bytes` 是选中 memory 文本的逻辑字节数，不是 DRAM/PIM hardware traffic；
- 不在 V5 中报告 row activation、bank conflict 或真实 Host-memory bus traffic；这些属于下一阶段 PIM physical evaluation；
- 不刻意追求更高 `rerank / selection` 占比；优化之后该占比下降是正常且应被视为正向结果。

### 11.3 必要分桶图

所有主要指标按真实自然 scope 分桶：

```text
[1, 49]
[50, 127]
[128, 255]
[256, 511]
[512, 1023]
[1024, +∞)
```

若实际数据没有某个桶，报告为空，不得补造样本。

---

## 12. 正式图表与预期用途

### Figure V5-1：MemoryCD natural memory scale distribution

- 横轴：memory scope size bucket；
- 纵轴：event count；
- 细分：single / cross / causal cross；
- 目的：证明压力来自真实用户长期历史，而非复制或混池。

### Figure V5-2：Selection latency breakdown vs. natural memory scope

- 仅 profiling；
- 展示 BGE encode、FAISS retrieval、MiniLM tokenize、MiniLM forward、context build；
- 分别画 R=32/64/128；
- 目的：定位真实 pipeline 中哪些阶段随自然 memory scope / candidate budget 变化。

### Figure V5-3：Task quality vs. selector

- task：rating、ranking 为主；
- 方法：no-memory / recency_k8 / dense_k8 / dense_rerank_r128_k8；
- protocol 分开；
- 目的：证明 rerank 并非只增加时间，且 quality gain 建立在真实 history 上。

### Figure V5-4：Rerank latency vs. ranking quality

- 横轴：MiniLM rerank model latency 或 selection wall latency；
- 纵轴：NDCG@3；
- 标点：methods；
- 目的：为后续论文方法提出“在保持 quality 的同时压缩 rerank frontier”建立基线。

### Figure V5-5：Official vs. causal cross-domain scope

- 报告 future-memory ratio 与 task metric 的差异；
- 目的：明确 benchmark comparability 与 online causality之间的差异，避免 reviewer 质疑 temporal leakage。

---

## 13. 验收条件

正式 V5 只有同时满足以下条件才可保留：

### 数据与隔离

- [ ] 数据来自 `WZDavid/MemoryCD` 官方 release；
- [ ] `tenant isolation audit = 100%`；
- [ ] `causal_* time-scope audit = 100%`；
- [ ] labels 与 runtime manifests 物理分离；
- [ ] no synthetic query / memory / hard negative；
- [ ] no cross-user candidate pool。

### 模型与代码

- [ ] 使用既有 BGE-base、MiniLM-L6、local Qwen；
- [ ] 不在主路径调用 Qwen controller；
- [ ] 主 rerank 使用 query-local batch>=4，首选 16；
- [ ] `faiss_flat` 与 brute-force Top-R 一致性检查通过；
- [ ] 旧 V4 模式和旧数据转换代码未被覆盖或破坏。

### 指标与口径

- [ ] profiling 和 full task run 分开输出；
- [ ] selection-only 不报告 end-to-end response latency；
- [ ] full task run 报 official task metrics、selection latency 与 Qwen generation latency；
- [ ] official 与 causal 结果分开汇报；
- [ ] all p50/p95/p99 由 per-event raw traces 计算，不从已平均的 summary 再平均。

### 可复现性

- [ ] 每个 run 有 immutable config snapshot；
- [ ] 每个 run 有 model / data / code version manifest；
- [ ] 所有 raw trace、summary、audit、figure source data 可追溯；
- [ ] README 明确 V4 和 V5 的任务定义、数据集角色和不可直接比较的指标。

---

## 14. 建议的 Codex 执行顺序

严格按以下顺序执行；每一步完成后先检查产物和验收项，再进入下一步。

```text
Step 1  阅读现有 V4 README、configs、profile script、embedding/reranker client。
Step 2  克隆官方 MemoryCD repo，下载官方 dataset，生成 provenance + inventory。
Step 3  运行官方 repo 5-user sanity check，确认数据和 task 语义。
Step 4  实现 MemoryCD loader、manifest builder、protocol builder、task prompts、labels separation。
Step 5  完成 isolation / time / leakage / dense correctness audit。
Step 6  实现 faiss_flat query-local retrieval 与 batched MiniLM pair scoring。
Step 7  先完成 5-user smoke run（所有 task / protocol / method）。
Step 8  跑 Phase 2 profiling：rating + ranking、R=32/64/128。
Step 9  审阅 profiling report；确认没有 controller、phase suffix、relation expansion 或误记账。
Step 10 跑 Phase 3 full task：先 rating/ranking，再 generation/summarization。
Step 11 运行全部 audit、summary、figure、statistical tests。
Step 12 写 V5 README，冻结 `versions/20260622_memorycd_native_v5_formal/`。
```

---

## 15. 预期的 V5 README 结论模板（仅在结果支持时使用）

不要预写具体数值。结果生成后可采用如下严格口径：

> 在 MemoryCD 的真实跨年、跨领域用户历史上，我们以原生 rating prediction、item ranking、review generation 和 review summarization 任务评估了现有 dense-retrieve-and-rerank pipeline。与此前受控的 phase-aware profiling 不同，V5 不强制多阶段 memory invocation，也不引入额外 controller；每个原生 task event 只进行一次合法 tenant-local memory selection。结果同时报告了官方协议与严格时间因果协议，并将 selection-only profiling 与完整 Qwen task latency 分开记账。该版本建立了真实 user-memory workload 下的 rerank frontier、candidate scope 与任务质量基线，为后续研究 task/domain-conditioned memory layout 和 pre-rerank candidate shaping 提供了无标签泄漏的对照基础。

不得写：

- “MemoryCD 证明了 phase-aware locality”；
- “V5 证明了 PIM hardware traffic”；
- “selection 在所有任务中都占端到端时间主导”；
- “cross-domain official protocol 完全没有未来信息”，除非 audit 已经验证；
- “严格因果结果等同官方 benchmark 分数”。

---

## 16. 失败与降级规则

| 问题 | 允许处理 | 不允许处理 |
|---|---|---|
| 官方数据下载失败 | 停止并记录；等待网络/权限恢复 | 用未验证的第三方 copy 或自行重建并称 MemoryCD |
| metadata 缺字段 | 使用官方 interaction 中足以完成任务的字段；报告字段缺失 | 让 LLM 补造 item metadata |
| Qwen 输出格式错误 | 记录 parse error；用官方 parsing fallback（若有） | 人工改写模型输出 |
| reranker OOM | 16→8→4 batch 回退并记录 | 静默改变 max_length / 换模型 / 丢弃高长度样本 |
| Causal scope 过小 | 如实按 scope bucket 报告；可将 scope<8 event 单列 | 从未来或其他用户补 memory |
| 全量成本过高 | 使用固定随机种子、按 scope strata 的明确抽样并声明为 subset | 文件前 N users 截断后称全量 |

---

## 17. 本轮完成后的边界

V5 完成后，下一步才可以在**相同 MemoryCD manifests、同一 BGE、同一 MiniLM、同一 Qwen、同一 K/R 和同一 evaluator**上引入：

```text
Task/domain-conditioned candidate selector
→ trace-derived co-access graph / hypergraph
→ pre-rerank pruning
→ matched-quality rerank-budget reduction
→ 最后才是 PIM logical / physical layout
```

V5 的价值在于让后续所有方法都必须回答一个清楚的问题：

> 在相同真实用户历史、相同模型、相同最终 context budget 和相同任务 evaluator 下，新的结构方法能否以更小的 rerank frontier 保持或提升原生任务质量？
