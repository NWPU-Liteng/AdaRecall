# Codex 执行指导：V5.1 MemoryCD 原生任务上的候选前沿质量—代价实验

> **归档说明（2026-07-17）**：本文对应的 MemoryCD 动机/preflight 实验已迁移到
> `agent-memory-profiling/versions/motivation_experiments/`。文中的旧版本路径仅
> 用于解释历史实验协议，不是当前 v21 的输入或可运行入口。

> **版本定位**：V5.1 是对 V5 的纠偏升级。它仍然保持 **MemoryCD + 现有 BGE-base retriever + MiniLM-L6 cross-encoder + 本地 Qwen**，不引入超图、PIM、合成数据或更大模型；但将实验主问题从“检索和 rerank 谁占多少时间”改为：
>
> > **候选前沿的规模与质量如何共同决定下游任务质量、Host 侧 rerank 工作量和端到端 memory-selection 成本？**
>
> 这轮实验要先建立后续超图 / PIM 方法真正需要的因果基线：在相同 rerank budget 下，候选更准是否能提升任务质量；在相同任务质量下，更高质量的候选前沿是否能以更小的 rerank budget 达成目标。
>
> **本轮固定模型（不得替换）**：
>
> ```text
> Dense retriever : models/bge-base-en-v1.5
> Cross-encoder   : models/cross-encoder-ms-marco-MiniLM-L6-v2
> Task executor   : 当前 local_qwen backend 所使用的本地 Qwen 模型
> ```
>
> **V5.1 的核心约束**：
>
> - 只使用官方 MemoryCD release 中的真实用户历史、真实商品元数据、官方任务切分和官方标签；
> - 不使用 synthetic query / synthetic memory / synthetic hard negative；
> - 不用 `planning / reasoning / output` phase、固定多次检索或 Qwen retrieval controller；
> - 不人为放大 rerank 占比；所有 candidate budget 都是明确的实验自变量；
> - 不把 cross-encoder inference 称作 PIM 将来要直接加速的对象；未来超图与 PIM 的直接优化对象是 **rerank 前候选前沿构造**；
> - 任何 online selector、retriever、reranker、Qwen prompt builder 都不得读取 target label、官方答案或未来 memory。

---

## 1. 本轮需要回答的论文问题

V4 的 phase-aware profiling 可保留为受控动机：它说明在特定 agent pipeline 中，cross-encoder rerank 可以占 selection path 的主要成本。但 V4 不能单独证明超图或 PIM 能解决该热点，因为它没有控制 candidate budget，也没有测量“候选质量—成本”的权衡。

V5.1 要建立下列三条证据链。

### Q1：为何候选前沿是系统瓶颈，而不只是 reranker 模型本身？

在真实 user memory pool 中，随着送入 cross-encoder 的 candidate 数增加，任务质量可能上升或趋于饱和，但 rerank pair 数、payload 读取量、Host side rerank 时间会继续增加。

必须测量：

```text
Task Quality(B)          : 原生 MemoryCD task metric 随 rerank budget B 的变化
T_rerank(B)              : cross-encoder 对 B 个 query-memory pair 的在线时间
Traffic(B)               : 读取 candidate header / payload 的字节数
T_selection(B)           : 从 query 编码到得到 final Top-K context 的总选择时间
```

### Q2：候选前沿质量是否比单纯增大 B 更重要？

固定同一个 rerank budget `B`，所有方法均只允许向 cross-encoder 发送 `B` 条候选。后续结构方法若能在固定 `B` 下取得更高 task quality，就证明其价值来自 **candidate utility density**，而不是多跑了更多 rerank pair。

V5.1 还没有超图，因此先建立可靠的 dense baseline 曲线；后续 V6 的 global graph / hypergraph 必须在同一个 `R, B, K` 协议下与它比较。

### Q3：更高质量的候选前沿能否把系统收益转化为更小的 rerank budget？

定义目标质量阈值 `Q0`。对方法 `m`：

```text
B*_m(Q0) = 达到 Q0 所需的最小 rerank budget B
```

将来若超图方法以 `B=16` 达到 dense baseline `B=64` 的质量，则才可以严谨地说：

```text
超图 / PIM 不直接加速 cross-encoder Transformer；
它们通过构造更高价值的 pre-rerank frontier，减少了必须交给 Host reranker 的 pair 数。
```

---

## 2. 统一术语与阶段边界（论文、代码、日志必须一致）

禁止再将所有操作笼统称为 `retrieval`。本轮统一使用以下符号。

```text
N : 当前 task event 合法可访问的全部真实 memory item 数
R : BGE dense seed retrieval 的 Top-R 数量
C : rerank 前 candidate frontier 的数量
B : 实际送入 cross-encoder 的 rerank pair budget
K : 最终进入 Qwen task context 的 memory item 数
```

V5.1 的 dense baseline 没有超图 / relation expansion，因此：

```text
合法 memory pool N
  -> BGE Top-R seed retrieval
  -> dense-score preselection，取 C=B 条
  -> MiniLM rerank B 条
  -> final Top-K context
  -> Qwen 完成原生 MemoryCD task
```

因此 V5.1 中 `C=B`。这不是最终算法，而是一个干净基线；未来方法应将 `Top-R` 转换为更高质量的 `C=B` frontier。

### 2.1 候选提取属于哪一阶段

以下操作统一归为 **Candidate Frontier Construction（候选前沿构造）**，属于 retrieval / pre-ranking，而不是 cross-encoder rerank：

- BGE dense seed retrieval；
- domain / time / tenant scope routing；
- metadata filtering；
- future 的 phase view routing；
- future 的 hyperedge expansion；
- future的 tile-level filtering；
- candidate dedup、coarse scoring、preselection。

MiniLM joint query-document scoring 属于 **rerank**。后续 PIM 研究的直接对象是：

```text
T_frontier = T_scope + T_seed + T_prefilter + T_expand + T_aggregate
```

并通过降低 `B` 间接降低：

```text
T_transfer + T_rerank
```

### 2.2 时间分解

每个 event 的 trace 必须记录：

```text
T_scope            构造合法 tenant/domain/time memory scope
T_query_encode     BGE 编码 query
T_seed             在合法 scope 中 exact Top-R dense search
T_frontier         Top-R 到 C=B 的 preselection / dedup（V5.1 预期很小）
T_header_transfer  candidate metadata / id 回传或读取量的逻辑统计
T_payload_fetch    为 rerank / Qwen 读取 memory text 的时间与字节数
T_rerank           MiniLM 对 B 个 pair 批处理打分
T_context_pack     Top-K context 拼接
T_task_qwen        Qwen 执行 MemoryCD 原生任务
T_total_selection  T_scope + T_query_encode + T_seed + T_frontier
                   + T_header_transfer + T_payload_fetch + T_rerank + T_context_pack
T_total_task       T_total_selection + T_task_qwen
```

注意：`T_task_qwen` 必须和 `T_total_selection` 分开报告。不要通过隐藏 Qwen 时间、或采用 batch=1 的 reranker，实现上人为制造 rerank 占比。

---

## 3. 保留的 V5 原则与 V5.1 新增修改

### 3.1 必须保留的 V5 修正

- 只执行一次原生 task memory selection；
- 关闭 `shared_once / fixed_three / adaptive_phase` 主模式；
- 关闭 Qwen retrieval controller；
- 关闭 synthetic `QuerySpec` task/entity tags；
- 关闭人为 relation expansion；
- 关闭 oracle / gold-label pruning；
- 使用 tenant-local、protocol-valid、time-valid 的真实 memory scope；
- 使用 `faiss_flat` exact dense retrieval 作为正确性基线；
- 同一 query 的 MiniLM rerank 使用 micro-batch，默认 `pair_batch_size=16`，不得以 batch=1 作为主结果；
- 将 `selection profiling` 与 `full Qwen task` 分开运行和记账。

### 3.2 V5.1 新增的强制修改

| V5 原设计 | V5.1 修改 | 原因 |
|---|---|---|
| 只比较 `recency_k8 / dense_k8 / dense_rerank_r128_k8` | 加入明确的 `R-B-K` candidate-budget sweep | 需要量化候选规模、候选质量与系统成本的关系 |
| 只报告单点 rerank 结果 | 同时报 `Task Quality vs B` 和 `Cost vs B` 曲线 | 单点无法说明超图 / PIM 为什么有必要 |
| 将 rerank latency 当作主要动机 | 将主对象改为 `candidate frontier utility` | 方法直接优化的是候选构造与候选有效密度 |
| 以 Top-R 全量 rerank 作为唯一 dense baseline | 明确 dense preselection 到 `B`，确保 B 是受控变量 | 以后各种结构方法才能在同一 rerank budget 下公平比较 |
| 只看均值 latency | 报告 p50 / p95 / p99、token counts、pair counts 与 traffic | PIM / layout 后续需要 tail 与数据移动证据 |

---

## 4. 数据、协议与严格不可泄漏边界

### 4.1 数据与任务

使用官方 MemoryCD release 的真实用户长期行为记录和官方四项原生任务：

```text
rating_prediction
item_ranking
review_summarization
review_generation
```

V5.1 主分析优先级：

```text
1. rating_prediction
2. item_ranking
3. review_generation
4. review_summarization（只作任务覆盖，不作为候选检索难度主结论）
```

理由：summarization 原生输入包含 target review 文本，历史 memory 对任务的作用更偏风格或偏好辅助，不应将其解释为最强 evidence retrieval 任务。

### 4.2 两套协议不可混报

保留 V5 的两套 manifest：

```text
official_single : 对齐官方单领域 task split
official_cross  : 对齐官方跨领域 task split
causal_cross    : tenant 相同、allowed domain 合法、timestamp < target timestamp
```

所有路径、run name、图题、表题必须包含 `protocol=<...>`。不得将不同 protocol 的分数取均值后称为一个统一 MemoryCD 结果。

### 4.3 Runtime 与 evaluator 分离

在线路径只允许看到：

```text
user/tenant id
合法历史 interactions
官方允许的 item metadata / candidate item metadata
source domain 与 target domain
合法 timestamp
原生 task query input
```

在线路径禁止看到：

```text
target rating
positive item identity / ranking label
gold review title
gold review text（generation task）
任何 evaluator metric
任何由 official labels 派生的字段
未来 timestamp 的 memory
其他 tenant memory
```

`evaluation_labels.jsonl` 必须独立保存，运行代码不得 import 或读取该文件。

---

## 5. 目录、版本和代码改造要求

不要覆盖 V4 或 V5。创建：

```text
versions/20260622_memorycd_candidate_frontier_v5_1_formal/
├── README_zh.md
├── manifests/
│   ├── memorycd_official_single/
│   ├── memorycd_official_cross/
│   └── memorycd_causal_cross/
├── configs_snapshot/
├── runs/
│   ├── selection_sweep/
│   ├── task_curve_subset/
│   └── task_final_full/
├── reports/
│   ├── memorycd_data_provenance.md
│   ├── protocol_and_leakage_audit.md
│   ├── candidate_frontier_sweep_report.md
│   ├── task_quality_curve_report.md
│   ├── pareto_budget_report.md
│   ├── runtime_and_traffic_report.md
│   └── final_v5_1_report.md
└── logs/
```

新增或修改的模块：

```text
src/memorycd/
├── schema.py
├── loader.py
├── protocol_builder.py
├── memory_scope.py
├── prompt_builder.py
├── faiss_store.py
├── candidate_frontier.py
├── task_executor.py
├── evaluator.py
└── audit.py

experiments/memorycd/
├── inspect_memorycd_release.py
├── build_memorycd_manifests.py
├── validate_memorycd_protocols.py
├── profile_candidate_frontier_sweep.py
├── run_memorycd_task_curve.py
├── run_memorycd_final_full.py
├── audit_memorycd_runs.py
└── summarize_memorycd_v5_1.py

configs/memorycd_v5_1/
├── common.yaml
├── official_single_selection_sweep.yaml
├── official_cross_selection_sweep.yaml
├── causal_cross_selection_sweep.yaml
├── task_curve_subset.yaml
├── final_full_baseline.yaml
└── final_full_pareto.yaml
```

不得把 MemoryCD task 硬塞到旧 `QuerySpec` 的 `fact / temporal / relation / plan_continuity / constraint` 枚举里。需要使用独立的 `MemoryCDTaskEvent` 数据类。

---

## 6. 模型与在线索引实现

### 6.1 固定运行模型

```yaml
embedding:
  backend: transformer
  model_path: models/bge-base-en-v1.5
  max_length: 512
  normalize: true
  dtype: bf16_or_fp16_as_supported

retrieval:
  backend: faiss_flat
  metric: inner_product
  embedding_cache: true

reranker:
  backend: cross_encoder
  model_path: models/cross-encoder-ms-marco-MiniLM-L6-v2
  max_length: 512
  pair_batch_size: 16
  sort_descending: true

generation:
  backend: local_qwen
  model_identifier: <从当前 V4/V5 run manifest 中读取并写入>
  temperature: 0.0
  seed: 20260622
```

若显存不足，pair batch size 只允许 `16 -> 8 -> 4` 逐级回退。不得静默改为 `1`。所有 event trace 记录实际 batch size、rerank pair token lengths、dtype、GPU 名称、Torch/Transformers/CUDA 版本。

### 6.2 FAISS 索引的正确实现

不得为每个 event 重建索引后把 build time 偷偷忽略。按协议分别实现：

- **official_single / official_cross**：对具有相同合法 static scope 的 tenant / domain scope 缓存 exact `IndexFlatIP`；索引构建时间单列为 offline preparation，不计入 query online latency。
- **causal_cross**：按用户时间顺序回放。每到一个新 event，只将已发生且允许进入 source domain 的真实 memory embedding 加入对应 tenant-local index；每次增量 add 的时间单列为 `T_index_update`，不得混入 `T_seed`。
- 若某个 event 的合法 scope 太小，不 padding、不跨用户补齐、不引入合成干扰项；如 `N < R`，实际 `R_eff=N`，写入 trace。

每个 protocol/task/domain 至少抽样 20 个 event，验证：

```text
Faiss Top-R IDs == brute-force normalized dot-product Top-R IDs
```

要求完全一致。

### 6.3 Memory payload 读取边界

为真实测量系统代价，区分逻辑 header 与 payload：

```text
header : memory_id, timestamp, domain, embedding position, text length
payload: review title, review text, official item metadata formatted text
```

V5.1 不模拟 PIM，但必须记录：

```text
header_bytes_scanned
header_bytes_returned
payload_bytes_for_rerank
payload_bytes_for_qwen_context
unique_payload_memory_count
```

计算字节数使用 UTF-8 encoded byte length；所有方法使用同一 memory text formatter。该统计是未来 Host-memory traffic / PIM layout 实验的真实 trace 基础，不得用假设带宽换算出“模拟 latency”。

---

## 7. 实验矩阵：先做完整 candidate-budget 基线，再跑最终任务

### 7.1 固定常量

```text
K = 8                    # 进入 Qwen 的最终 memory 数；主结果固定
R = 32, 64, 128          # dense seed candidate count
B = 8, 16, 32, 64        # cross-encoder rerank pair budget，必须 B <= R
```

`R` 表示 BGE seed retrieval 的视野大小；`B` 表示 cross-encoder 真正被允许打分的 pair 数。

在 V5.1 dense baseline 中：

```text
Top-R dense seeds
  -> 按 dense score 取前 B 条作为 frontier
  -> MiniLM rerank B 条
  -> Top-K = 8
```

不要运行 `R=256` 或更大值作为正式主矩阵。若自然 user scope 小于 R，使用 `R_eff=min(N,R)`。V5.1 只探索到 128 是为了观察自然规模下候选质量的饱和与成本，不是为了放大 rerank 时间。

### 7.2 Selection sweep（全量真实 event；不调用 Qwen）

目的：精确刻画 `N, R, B` 如何影响 candidate construction 和 rerank 成本。

对每个 protocol、task、domain 的全部合法真实 event 运行：

```text
Dense-only:
  dense_k8

Dense + rerank:
  (R,B) ∈ {
    (32,8), (32,16), (32,32),
    (64,8), (64,16), (64,32), (64,64),
    (128,8), (128,16), (128,32), (128,64)
  }
```

该阶段仍全部使用真实数据、真实 BGE 和真实 MiniLM；它不执行 Qwen，是为了在 full corpus 上可靠测量 candidate frontier 与 rerank cost，而不将 Qwen generation noise 或执行成本混入选择路径。

每个 event 输出：

```json
{
  "event_id": "...",
  "protocol": "official_single",
  "task": "item_ranking",
  "tenant_id_hash": "...",
  "N": 0,
  "R_requested": 64,
  "R_effective": 64,
  "B_requested": 32,
  "B_effective": 32,
  "K": 8,
  "dense_seed_ids": ["..."],
  "frontier_ids": ["..."],
  "reranked_ids": ["..."],
  "final_context_ids": ["..."],
  "t_scope_ms": 0.0,
  "t_query_encode_ms": 0.0,
  "t_seed_ms": 0.0,
  "t_frontier_ms": 0.0,
  "t_payload_fetch_ms": 0.0,
  "t_rerank_ms": 0.0,
  "t_context_pack_ms": 0.0,
  "header_bytes_scanned": 0,
  "payload_bytes_for_rerank": 0,
  "payload_bytes_for_qwen_context": 0,
  "rerank_pair_count": 0,
  "rerank_pair_token_sum": 0,
  "reranker_pair_batch_size": 16
}
```

不得把任何 label、answer、positive item、target review 写入该 selection trace。

### 7.3 Task-quality curve（固定、真实、可复核的分层 event 子集；调用 Qwen）

目的：建立 `Task Quality vs B`，避免只观察 time 而不知道较小 B 是否损失任务效果。

Qwen 全量跑完整 11 组配置成本较高，因此本阶段使用**固定的、仅由公开 runtime metadata 决定的、分层抽取的真实 test event 子集**；这仍是全部真实数据与真实模型运行，不是模拟数据。

抽样规则：

```text
- 每个 protocol × task × domain 按 hash(event_id, seed=20260622) 排序；
- 从每个 strata 取 min(200, 该 strata 全部 event) 个；
- 若某 task/domain 少于 200，全部保留；
- 抽样不读取 evaluation label；
- 保存 event_id 清单与 hash，后续所有方法、所有 B 使用完全相同的 paired event set。
```

运行矩阵：

```text
recency_k8
dense_k8
(R,B) ∈ {
  (32,8), (32,16), (32,32),
  (64,16), (64,32), (64,64),
  (128,16), (128,32), (128,64)
}
```

为控制预算爆炸，不运行 `(64,8)` 与 `(128,8)` 的 Qwen 全任务，因为它们与 `(32,8)` 的主结论重复；它们保留在全量 selection sweep 中。

必须为每个配置调用真实本地 Qwen，并调用官方任务 evaluator。主输出：

```text
TaskMetric(R,B,K=8)
T_total_selection(R,B)
T_task_qwen(R,B)
T_total_task(R,B)
p50/p95/p99 of selection and rerank
payload bytes
```

### 7.4 Full formal task runs（全部真实 test event；调用 Qwen）

selection sweep 和 subset curve 完成后，自动寻找每个 `protocol × task` 的两个关键点：

```text
Reference-high-budget point:
  Q_ref = 选择 task metric 最高且稳定的 dense-rerank 配置；
  若统计上并列，优先较小 B、较小 R。

Pareto-min-budget point:
  B* = 最小 B，使其 task metric 的 bootstrap 95% CI 下界
       不低于 Q_ref - δ。
```

`δ` 必须按任务预先固定，不可看结果后挑选：

```text
rating prediction : 官方 metric 的 1% 相对容忍度
item ranking      : NDCG/MRR 的 1% 相对容忍度
review generation : 官方主 metric 的 1% 相对容忍度
summarization     : 官方主 metric 的 1% 相对容忍度
```

若官方 metric 是百分数，则用 1 个百分点评价；若是 0–1 比例，则用 `0.01` 绝对差。具体数值以官方 evaluator 的 metric range 确认后写入 `common.yaml` 和 README，禁止根据结果临时改变。

对所有真实 test event，运行：

```text
1. recency_k8
2. dense_k8
3. dense_rerank_reference_high_budget
4. dense_rerank_pareto_min_budget
```

这四组是 V5.1 论文表格中的正式 MemoryCD 成绩；全量 task run 不再跑所有 grid 点。

---

## 8. 指标、统计与绘图要求

### 8.1 主任务质量指标

必须优先使用官方 MemoryCD evaluator 的任务指标、输出格式和 parsing 规则。不要自创一个与官方不兼容的 QA metric。

报告形式：

```text
Protocol × Task × Domain：
  official metric mean
  95% bootstrap confidence interval
  number of evaluated events
  parse failure rate
```

### 8.2 Candidate frontier 的可用指标

MemoryCD 未必提供每个 task 的细粒度 “gold support memory item” 标注，因此 **不得把无标注的 heuristic 叫作 Candidate Precision 或 Evidence Recall**。

V5.1 应使用：

```text
QualityRetention(B) = TaskMetric(B) / TaskMetric(reference-high-budget)

BudgetNormalizedUtility(B) =
  (TaskMetric(B) - TaskMetric(no_memory)) / B

RequiredBudget(Q0) = 最小 B，使 TaskMetric(B) >= Q0
```

其中 `no_memory` 可在 task-quality subset 上运行一次，仅作为 task quality 下界；它不参与 latency 主比较。

额外报告：

```text
Top-K context redundancy
  = 1 - unique_parent_asin_count / K

Cross-domain context fraction
  = selected memory 中 source domain 与 target domain 不同的比例

Temporal span of selected context
  = max(timestamp) - min(timestamp)
```

这些是诊断性指标，不能替代原生 task quality。

### 8.3 系统代价指标

对每个 `(R,B)` 报：

```text
p50 / p95 / p99 T_query_encode
p50 / p95 / p99 T_seed
p50 / p95 / p99 T_frontier
p50 / p95 / p99 T_payload_fetch
p50 / p95 / p99 T_rerank
p50 / p95 / p99 T_total_selection
p50 / p95 / p99 T_task_qwen（仅 task run）
p50 / p95 / p99 T_total_task（仅 task run）

rerank_pair_count
rerank_pair_token_sum
header_bytes_scanned
payload_bytes_for_rerank
payload_bytes_for_qwen_context
unique_payload_memory_count
```

每次正式运行都必须包含：

```text
warmup events: 前 20 个 event 不计入 latency 聚合，但保留原始 trace
cold/warm policy: 明确写入 run manifest
GPU synchronization: CUDA timing 必须同步或使用 CUDA events；禁止只测 Python dispatch 时间
```

### 8.4 必出的图

1. **Budget–Quality Curve**：`TaskMetric` 对 `B`，按 `R` 分线；每个 task/protocol 至少一张，主文可选择 rating + ranking。
2. **Budget–Selection Cost Curve**：`T_total_selection` 与 `T_rerank` 对 `B`，同一横轴。
3. **Quality–Cost Pareto Curve**：横轴 `T_total_selection` 或 `payload bytes`，纵轴 task metric；标记 `B*` 与 reference point。
4. **Rerank Work Decomposition**：`T_seed / T_frontier / T_payload_fetch / T_rerank` 堆叠条形图，不同 B。
5. **Natural Memory Scale Stratification**：按照 N 的分位数分桶（例如 Q1/Q2/Q3/Q4），画 `TaskMetric` 和 `T_selection`；不复制 memory、不跨用户混池。
6. **Tail Plot**：不同 B 的 p50/p95/p99 `T_total_selection`，为后续 PIM latency 论证提供真实 trace。

主文中不要再把“rerank 占比越高”作为成功指标。优化后 rerank 占比下降是预期收益，不是坏结果。

---

## 9. 实验方法与基线命名规范

正式 run 必须使用以下命名：

```text
memorycd_<protocol>_<task>_<setting>_recency_k8
memorycd_<protocol>_<task>_<setting>_dense_k8
memorycd_<protocol>_<task>_<setting>_dense_r<R>_b<B>_k8
memorycd_<protocol>_<task>_<setting>_no_memory_k0   # subset only
```

`<setting>` 例如 `single_books`、`cross_electronics_to_books`，必须源自官方 manifest，而非手写标签。

### 9.1 Baseline 的具体行为

| 方法 | 操作 | 用途 |
|---|---|---|
| `no_memory_k0` | 只给 Qwen 原生 target task input | task quality 下界；仅 subset |
| `recency_k8` | 合法 scope 中按 timestamp 取最近 8 条 | 简单时效基线 |
| `dense_k8` | BGE exact Top-8，直接进 Qwen | 无 rerank dense baseline |
| `dense_r<R>_b<B>_k8` | BGE Top-R → dense top-B preselection → MiniLM rerank B → Top-8 | V5.1 主标准基线 |

注意：若 `B < K` 或 `R_eff < K`，该 event 不应报错；设置 `K_eff=min(K,B,R_eff)` 并写入 trace。正式分析需单列该类小 scope event 占比。

---

## 10. 运行步骤（Codex 必须按序执行）

### Step 0：读取已有环境与 V4/V5 文档

先读取，不得直接重写：

```text
README / 环境说明
versions/20260621_real_trace_adaptive_phase_v4_formal/README_中文.md
versions/20260622_memorycd_native_v5_formal/README_zh.md（如已存在）
当前 embedding_client.py、reranker_client.py、local_qwen backend
```

输出 `reports/v5_1_preflight.md`：

```text
- 当前 Python/conda env
- GPU、CUDA、PyTorch、Transformers
- BGE/MiniLM/Qwen 本地路径与 config hash
- 是否可 import faiss
- 当前 MemoryCD data 路径、文件清单和 schema
- 旧 V4/V5 代码中需复用/隔离的模块
```

### Step 1：官方 MemoryCD 数据检查与 manifest 生成

1. 使用官方 release，生成 `memorycd_data_provenance.md`；
2. 通过官方 repository 的最小命令验证 parser、任务 prompt 与 evaluator；
3. 生成 `official_single / official_cross / causal_cross` manifest；
4. 运行 tenant、time、label leakage pre-audit；
5. 保存固定 event subset 清单（hash 选择，不能读 label）。

任何 audit 不通过，停止；不要进入模型运行。

### Step 2：模型和 indexing smoke test

每个 protocol/task/setting 选 5 个真实 event：

```text
R=32, B=8, K=8
methods=recency_k8,dense_k8,dense_r32_b8_k8
```

验收：

- BGE 与 MiniLM 均来自本地模型目录；
- FAISS 与 brute-force Top-R ID 完全一致；
- Qwen 输出可由官方 evaluator parse；
- memory scope 无跨 tenant / future / label leakage；
- 所有 trace 字段完整；
- 当前 task run 单次 selection，不含 controller 或 phase suffix。

### Step 3：全量 selection sweep

对所有真实 test event（或官方 task 定义的完整 evaluation event）跑 Section 7.2 的 11 个 `(R,B)` 配置。

这里不运行 Qwen，因此可完整获得真实规模下的 `N,R,B`、latency、pair token、payload 流量与 tail 分布。

输出：

```text
runs/selection_sweep/
reports/candidate_frontier_sweep_report.md
reports/runtime_and_traffic_report.md
```

### Step 4：Qwen task-quality curve 子集运行

读取固定 manifest subset；对 Section 7.3 的所有配置运行真实 Qwen 与官方 evaluator。

输出：

```text
runs/task_curve_subset/
reports/task_quality_curve_report.md
reports/pareto_budget_report.md
```

### Step 5：自动选择 B*，人工审核后锁定最终配置

脚本只负责生成候选 `reference-high-budget` 与 `pareto-min-budget`；不要让脚本静默决定论文设置。

在报告中列出：

```text
- 每个 protocol/task/setting 的全部 (R,B) 分数与 95% CI
- Q_ref
- 预注册的 δ
- B* 候选
- 若不存在达标 B*，明确写“未观察到 budget compression”
```

人工确认后，将固定配置写入 `configs_snapshot/`。不得根据 full-test 结果再次选择 B*。

### Step 6：全部真实 test event 的最终 full task run

按 Section 7.4 对所有合法真实 test event 运行：

```text
recency_k8
dense_k8
dense_rerank_reference_high_budget
dense_rerank_pareto_min_budget
```

Qwen、BGE、MiniLM 均为真实本地模型；不得用 task metric proxy 代替 Qwen。

### Step 7：审计、汇总、归档

必须生成：

```text
protocol_and_leakage_audit.md
model_runtime_audit.md
candidate_frontier_sweep_report.md
task_quality_curve_report.md
pareto_budget_report.md
runtime_and_traffic_report.md
final_v5_1_report.md
```

将 manifests、configs、git commit、raw JSONL、summary CSV、plots 一起冻结到 V5.1 目录。不得覆盖 V4/V5。

---

## 11. 必须实现的审计与失败条件

| Audit | 通过条件 | 不通过时处理 |
|---|---|---|
| Tenant isolation | 100% selected memory 属于 event tenant | 停止正式 run，修复 scope builder |
| Time causality（causal） | 100% timestamp < target timestamp | 停止正式 run |
| Label leakage | runtime 文件、prompt、selector 输入不含 hidden labels | 停止正式 run |
| FAISS exactness | 采样 event Top-R 与 brute force 完全一致 | 停止正式 run |
| Model identity | 路径/config hash 与固定模型一致 | 停止正式 run |
| No phase/controller | trace 中 `retrieval_invocations == 1` | 停止正式 run |
| Reranker batching | `pair_batch_size in {4,8,16}` 且实际不为 1 | 不得纳入正式主表 |
| Trace completeness | N,R,B,K、时间、bytes、token、IDs 齐全 | 停止汇总 |
| Task parsing | 解析失败率单独报告；>5% 时人工检查 | 不得只报告成功样本 |

### 11.1 强制泄漏扫描关键词

对 runtime manifest、prompt dump、trace 和 config 做静态扫描：

```text
gold_
label
positive_
negative_parent_asins      # 除 item ranking 的 target candidate list 构造外不得进入 selector
answer
reference
oracle
future_
```

item ranking 中官方 candidate item metadata 可以用于生成 Qwen ranking prompt；但 positive identity 与 ranking label 不能被标记给 runtime 代码。审计工具应支持 task-specific allowlist，避免误报。

---

## 12. 最终交付物与验收标准

### 12.1 代码交付

- 独立 MemoryCD adapter，不污染旧 LoCoMo/LongMemEval/MSC 路径；
- 可重放 manifest、固定 subset manifest、独立 labels；
- exact FAISS retrieval、batched MiniLM rerank、真实 Qwen task executor；
- `R-B-K` sweep runner、bootstrap summarizer、Pareto selector、auditor；
- 单元测试：scope、leakage、FAISS exactness、R/B/K edge case、task parser。

### 12.2 数据与可复核交付

- 数据来源和 hash；
- protocol 定义；
- 用户 / domain / event 数统计；
- subset selection hash 与 ID 列表；
- model/environment manifest；
- raw event traces；
- 聚合 CSV；
- 全部图和原始 plotting input。

### 12.3 V5.1 成功判据

V5.1 不要求现在就证明超图优于 dense baseline；它必须完成的是建立可信、真实的基线和可证伪结论。

至少应能得到以下其中一种明确结论：

1. **存在 budget saturation**：较大 B 的任务质量提升已经饱和，但 selection/rerank cost 持续上升。此时候选前沿质量有明确优化空间。
2. **存在 budget sensitivity**：较小 B 明显损失任务质量。此时未来超图若要成功，必须提升 fixed-B candidate utility；不能只做布局加速。
3. **不存在明显 rerank budget 价值**：dense_k8 已接近高 B rerank。此时应诚实调整后续设计，不应强行把 PIM 动机建立在 rerank compression 上。
4. **数据集 / task 异质性显著**：例如 item ranking 高度依赖候选量，而 rating prediction 不敏感。此时后续论文应按 workload 区分，而非宣称通用优势。

只有在得到第 1 或第 2 类结论后，才进入 V6：以相同 `N,R,B,K` 协议实现 graph / hypergraph candidate frontier，并检验曲线能否向左移动。

---

## 13. 后续 V6 的接口预留（本轮不实现）

V5.1 的 trace 和接口必须为后续方法保留如下输入输出：

```python
class CandidateFrontierSelector:
    def select(
        self,
        event: MemoryCDTaskEvent,
        dense_seed_ids: list[str],  # size R
        rerank_budget: int,         # B
    ) -> list[str]:                 # exactly min(B, len(dense_seed_ids)) IDs
        ...
```

V5.1 的实现：

```text
selector = DenseScorePrefixSelector
frontier = dense_seed_ids[:B]
```

V6 的未来实现：

```text
selector = MultiViewHypergraphSelector
frontier = structure_aware_select(dense_seed_ids, B)
```

两者必须共享：

```text
same event
same legal memory scope N
same BGE query embedding
same dense Top-R seed set
same reranker
same K
same Qwen task executor
same evaluator
```

这保证后续方法收益只能来自候选前沿质量，而不是来自改变模型、扩大 budget、泄漏标签或改变 task prompt。

---

## 14. Codex 最终工作清单

执行结束后，Codex 必须在最终报告中逐项回答：

1. MemoryCD 的真实用户 history 是否按 tenant 和 protocol 正确隔离？
2. causal protocol 是否无未来 memory？
3. 当前主路径是否恰好一次 selection、无 phase/controller 额外调用？
4. `R`、`B`、`K` 分别是多少，是否每个 event 都被记录？
5. `T_seed`、`T_frontier`、`T_payload_fetch`、`T_rerank`、`T_task_qwen` 如何随 B 变化？
6. rating / ranking / generation / summarization 的官方 task metrics 如何随 B 变化？
7. 哪个 B 是 high-budget reference，哪个 B 是 Pareto-min-budget，选择依据和 CI 是什么？
8. 是否观察到“质量近似不变但 rerank budget 显著降低”的真实证据？
9. 哪些 task 不支持 rerank-compression 动机？
10. V6 的超图方法应以哪条 V5.1 budget-quality curve 为主要对照？

---

## 15. 禁止项总结

```text
禁止：为了提高 rerank 占比，故意把 MiniLM 改成更慢的大模型。
禁止：为了制造压力，复制 memory、跨用户混池、注入 synthetic hard negatives。
禁止：把 R、B、K 混为同一个“Top-K”。
禁止：将 BGE seed retrieval、preselection 和 cross-encoder rerank 统称为 retrieval。
禁止：把无 evidence label 的 MemoryCD 指标称作 Candidate Precision / Evidence Recall。
禁止：让 selector 读取 gold labels、future memory 或 official answer。
禁止：将 selection-only 的 latency 当成 full end-to-end task latency。
禁止：在 V5.1 中提前实现超图/PIM 后再宣称得到干净 baseline。
禁止：因某一 workload 没有显著 budget sensitivity 而隐藏该结果。
```

---

## 16. 本轮最终论文表述草案（仅作写作口径，不是实验结论）

在实验完成前，论文可将问题表述为：

> 长期 memory workload 的主要困难并不只在于从大规模历史中进行 dense lookup，而在于如何在有限的 rerank 与 context budget 下构造高效、高价值的候选前沿。低质量候选会迫使系统扩大 cross-encoder 的处理范围以避免遗漏，从而放大 Host-side reranking 与数据搬运成本。V5.1 首先在真实用户长期行为数据上量化该 quality–cost trade-off；后续工作再以多视图超图与 PIM-friendly layout 提升候选前沿的效用密度。

不得在 V5.1 未完成前宣称：

> 超图已经降低了 rerank 时间。

V5.1 的正确产出是：为这一未来结论提供严格、可比较、无模拟数据的真实基线。
