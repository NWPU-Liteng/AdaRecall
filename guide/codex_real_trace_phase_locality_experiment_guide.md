# Codex 执行指导：V4 真实数据、真实模型的 Phase Locality 与多视图超图候选剪枝实验

> **归档说明（2026-07-17）**：本文对应的 V3/V4 动机实验已迁移到
> `agent-memory-profiling/versions/motivation_experiments/`。文中的旧路径
> `versions/20260616_phase_aware_real_agent_v3_formal` 和
> `versions/20260621_real_trace_adaptive_phase_v4_formal` 是历史命名；阅读和
> 复核请使用该目录下同名子目录，本文不再代表 v21 active pipeline。

> **文档目标**：在现有 `Phase-Aware Real-Agent V3` 的基础上，完成下一阶段正式实验。该阶段不再使用任何人为构造的 query、memory、task tag、relation 或 gold-group 作为在线输入；所有正式 workload 必须来自公开数据集的原始对话、原始问答或原始会话记录，所有在线选择、rerank、阶段控制和回答必须由真实模型执行。
>
> **本阶段的核心问题**：
>
> 1. 当阶段性 memory 调用由真实 LLM agent 自主决定，而不是由模板强制触发时，phase-aware selection 是否仍然形成显著成本？
> 2. 从真实历史交互中积累的 phase-conditioned access trace，是否存在可复用的高阶共访问局部性？
> 3. 仅利用过去真实 trace、而不访问任何答案或证据标注时，多视图超图是否能压缩 cross-encoder 前的候选集合，并保持真实 QA 的证据覆盖与回答质量？
>
> **本阶段不做 PIM timing / hardware simulation。** 先用真实数据和真实模型证明 logical layout / candidate pruning 的质量与端到端收益；PIM row/bank 映射留到下一阶段。

---

## 0. 执行前必须阅读的现有资产

执行前先阅读并理解以下文件，不要直接重写现有 V3：

```text
versions/20260616_phase_aware_real_agent_v3_formal/README_zh.md
experiments/motivation_memory_selection/profile_longgraph_memory_selection.py
src/agent_executor.py
src/reranker_client.py
src/embedding_client.py
src/vector_store.py
experiments/motivation_memory_selection/import_public_dataset.py
```

现有 V3 的正式目录必须保持可复现、不可覆盖：

```text
versions/20260616_phase_aware_real_agent_v3_formal/
```

新版本必须在独立目录中创建：

```text
versions/20260621_real_trace_adaptive_phase_v4_formal/
```

---

## 1. V3 已完成的价值与本阶段必须修复的问题

### 1.1 V3 已经可以保留的结论

V3 已完成以下动机层证据：

- 在 `planning / reasoning / output` 三个阶段分别执行 memory selection 时，完整 selection path 的成本会被放大；
- selection 的热点不只在 coarse retrieval，还包括 metadata filtering、relation expansion、candidate aggregation、rerank 和 context construction；
- 使用真实 cross-encoder 后，rerank 往往成为主要成本；
- 在 LoCoMo、LongMemEval 和 MSC 的真实 memory 文本上，规模增长会放大 selection / rerank 压力；
- `oracle_pruned_rerank` 说明：如果 rerank 前候选集合能够被正确收缩，理论上存在明显收益空间。

V3 继续作为论文的 **Motivation / controlled phase-aware profiling**，但不要将其直接写成“真实 agent 必然在三个阶段各检索一次”的最终系统结论。

### 1.2 V3 的主要不足

| 编号 | 当前问题 | 为什么会被质疑 | V4 修复原则 |
|---|---|---|---|
| G1 | memory 文本来自真实数据，但统一 `QuerySpec` 仍经 `synthetic_memory_generator.py` 构造 | 不能严格称为完全真实 workload；真实 query 的语言、约束和任务分布被替换 | 正式 V4 只读取原始 QA、原始聊天 turn 或原始会话；`synthetic_memory_generator.py` 不得位于正式数据路径 |
| G2 | phase query 由固定 suffix 强制改写，三个 phase 固定各调用一次 selection | 会被认为热点来自人为三次调用，而非真实 agent 行为 | 改成真实 Qwen controller 的自适应触发；每个 phase 可 `retrieve` 或 `skip`，output 默认不检索 |
| G3 | phase-aware 路径前还额外执行一次普通 selection pass | 容易使实际 wall-clock 与论文记账口径不一致 | `shared_once`、`fixed_three`、`adaptive_phase` 必须拆成独立运行模式；正式路径不得无意义预跑 |
| G4 | `oracle_pruned_rerank` 使用 `gold_group_id` | group 级 oracle 不等于真实 evidence；任何在线使用会产生标签泄漏 | oracle 仅保留为离线理论上界；在线 selector、layout builder、reranker 均不得读取 gold 字段 |
| G5 | `gold_group_id` 常是 conversation / session / sample 粗粒度分组 | 即使只做评估，也不等价于“支持该答案的 evidence set” | LoCoMo 使用官方 `evidence` dialog ids；LongMemEval 使用 `answer_session_ids` 与 `has_answer`，仅 evaluator 可见 |
| G6 | MSC 大规模 memory 若被直接放入一个无 namespace 的全局索引，会混入不同用户/对话的私有历史 | 不符合 personal agent memory；会引入不真实的跨用户召回 | 所有 runtime 检索必须先按 `tenant_id / conversation_id` 隔离；全局规模只作为 multi-tenant physical packing 统计，不允许跨 tenant 候选 |
| G7 | `ann_numpy` 为仓库内轻量实现并可能 fallback 全扫描 | retrieval latency 可能是实现特性而非问题本身 | V4 主实验保留现有 ANN 以保持连续性，同时加入 `faiss_flat` correctness reference 和 `faiss_hnsw` serving baseline；相同 recall 下比较 |
| G8 | 目前主要强调 latency，缺少与原始 benchmark 对齐的 QA / evidence 质量 | 只证明“更快”不能证明 selection 正确 | 对 LoCoMo、LongMemEval 同时报 evidence recall、answer quality、abstention / coverage 与 latency |
| G9 | relation expansion、task/entity tag 的来源可能混有人工规则或导入阶段派生字段 | 容易在没有明确说明时成为隐性先验 | V4 主路径关闭旧 relation expansion 与手工 tag filter；先只使用原始文本、时间、session、tenant、真实模型生成的阶段意图、历史 trace |
| G10 | 现有 `host_memory_traffic` 是逻辑统计，非真实 DRAM/PIM 总线测量 | 不能提前把它写成硬件访存结论 | V4 报告为 `logical_payload_bytes` / `reranker_input_bytes`；PIM 硬件 traffic 留给下一阶段模拟或硬件评估 |

---

## 2. V4 的不可违反约束

### 2.1 真实数据约束

正式 V4 的每一个 runtime query 必须来自下列之一：

1. **LoCoMo 原始 QA 的 `question` 字段**；
2. **LongMemEval 原始 QA 的 `question` 字段**；
3. **LoCoMo / LongMemEval / MSC 原始多轮会话中真实出现的 user turn**，用于历史在线 replay；
4. **MSC 原始对话中的真实上下文与真实下一轮 target**，仅用于大规模、在线 trace / latency / locality 评估。

禁止：

- 禁止由模板生成 `fact / temporal / relation / plan_continuity / constraint` 等新 query；
- 禁止用 LLM 改写或扩写 benchmark question 后再作为正式评测 query；
- 禁止生成 filler memory、hard negative memory 或伪 relation；
- 禁止为了增加 memory scale 复制、扰动、拼接或重写真实 memory；
- 禁止用答案、evidence、session answer label、gold group 作为在线 retrieval / rerank / layout 特征。

允许：

- 使用真实模型对**原始 query**生成结构化阶段意图；
- 使用真实模型对**原始 memory 文本**离线生成可审计摘要或向量，但主结果优先使用原始 turn / 原始 session 文本；
- 使用公开数据集自身提供的时间戳、session id、speaker role、conversation id 作为 runtime metadata；
- 以真实历史 agent invocation trace 构建在线更新的图 / 超图。

### 2.2 真实模型约束

V4 主配置固定使用已在 V3 中验证过的真实模型：

```text
Embedding model:  BAAI/bge-base-en-v1.5（本地既有模型）
Reranker model:   cross-encoder/ms-marco-MiniLM-L6-v2（本地既有模型）
Agent/controller: 当前 local_qwen backend 对应的真实 Qwen 模型
```

要求：

- 所有模型权重、revision、dtype、device、batch size、tokenizer revision 写入 `run_manifest.json`；
- controller、reasoner、final answer 可以共享同一 Qwen 权重，但每一次 LLM 调用必须写入真实 token usage 与 elapsed time；
- 主结果默认 `temperature=0`、固定 seed；
- `reranker_batch_size=1` 作为在线延迟主结果；另提供 `batch_size=8` 的吞吐敏感性结果，不替代在线结果；
- 如果本地已经有 BGE-large 或 MiniLM-L12，可在后续 robustness 阶段额外运行，但不得用其替换主结果而不报告模型变化。

---

## 3. 数据集与任务定位

### 3.1 LoCoMo：真实 QA + 真实对话 replay

**正式作用**：真实 QA 质量、turn-level evidence recall、phase trace replay。

- memory unit：原始 dialog turn，保留 `sample_id`、`session_id`、timestamp、`dia_id`、speaker、raw text；
- 正式 query：原始 `qa.question`；
- evaluator-only label：`qa.answer`、`qa.evidence`、`qa.category`；
- history replay：按 session 时间与 turn 顺序回放原始 user turn；回放时 memory 仅包含已发生的 earlier turns；
- QA 时，不得把 `qa.answer`、`qa.evidence`、`qa.category` 传入 agent、controller、retriever、reranker 或 hypergraph builder。

### 3.2 LongMemEval：真实 QA + session / turn-level evidence 评估

**正式作用**：真实 QA 质量、session-level 与 turn-level evidence recall、更新 / 时间 /跨 session 等能力覆盖。

- 使用官方 cleaned 数据版本；
- memory unit：原始 `haystack_sessions` 内的 turn，保留 session id、时间、role、原始 text；
- 正式 query：原始 `question`；
- evaluator-only label：`answer`、`answer_session_ids`、turn 中的 `has_answer`、`question_type`；
- 主准确性集使用 `LongMemEval-S` 全部 500 条；
- 大规模端到端压力集使用 `LongMemEval-M` 全部 500 条，输出应与 S 分开报告；
- 对话 replay 必须按每个 session 内的原始 user turn 顺序进行，且时刻 `t` 不得访问后续 turn/session。

### 3.3 MSC：真实多 session 对话，用于在线 locality 与规模

**正式作用**：长期真实交互 replay、trace 稳定性、动态超图更新、规模和 latency。它不进入“有 gold evidence 的 QA accuracy 主表”。

- memory unit：原始历史 session 的 turn；
- runtime query：原始 current user context / user turn；
- target：原始下一轮 response，仅可作为可选 response-similarity 辅助指标；
- 需要严格以 conversation / speaker-pair 为 `tenant_id`；
- 禁止把不同 tenant 的 memory 放入同一个候选池；若底层向量 index 物理共享，必须在 ANN 搜索前或 search result 后以 namespace 做硬隔离，且记录 `cross_tenant_candidate_count == 0`；
- MSC 仅报告 access locality、candidate size、rerank latency、payload bytes、动态更新成本和服务级多 tenant scale；不要把它的 session id 当作 QA gold evidence。

---

## 4. V4 真实自适应 Phase Agent

### 4.1 三种必须独立运行的模式

| 模式 | 调用规则 | 用途 |
|---|---|---|
| `shared_once` | 首次需要 memory 时只 selection 一次，后续阶段复用 context | 成本低的保守 baseline |
| `fixed_three` | planning / reasoning / output 各强制一次 selection | 仅复现 V3 controlled amplification；不是最终真实 agent 主结论 |
| `adaptive_phase` | 每个阶段由真实 Qwen controller 返回 `retrieve` 或 `skip` | V4 正式主路径 |

**重要**：一次 `adaptive_phase` 正式运行中，不得先额外执行“普通 selection pass”用于诊断。若需要单独诊断，应当用独立 config 运行并且不进入总响应 latency。

### 4.2 Controller 的输入与输出

禁止使用固定 phase suffix。Controller 使用 Qwen 对真实原始 query 和当前阶段状态生成 JSON：

```json
{
  "action": "retrieve | skip",
  "retrieval_goal": "只在 action=retrieve 时填写；必须是面向证据需求的简短自然语言描述",
  "needed_evidence": ["time", "preference", "event", "constraint", "relation"],
  "reason": "简短理由",
  "confidence": 0.0
}
```

#### Planning phase

输入：原始用户 query + 当前 session 可见上下文；不得提供未来 memory。

职责：生成初始计划；必要时请求 memory。

#### Reasoning phase

输入：原始 query + planning 的结构化结果 + planning 已选 context 的 ID/text 摘要。

职责：判断是否还缺关键证据，而不是无条件再次检索。

#### Output phase

输入：原始 query + 已获得证据 + reasoning state。

默认 `skip`；仅在模型显式识别到某个结论无法被已有证据支持时才允许 `retrieve`。

### 4.3 Controller 防泄漏检查

controller prompt、runtime state、selector feature 不得含以下字段：

```text
gold_group_id
answer
qa.answer
qa.evidence
answer_session_ids
has_answer
question_type / category（可用于离线分桶，不得作为 runtime prompt 特征）
```

实现 `validate_no_leakage.py`：

1. 静态扫描 V4 runtime selector/controller/hypergraph 文件，禁止上述标识符进入 online feature path；
2. 在每条 `phase_trace.jsonl` 中记录 `runtime_input_fields`；
3. 在 formal run 结束时输出 `leakage_audit.md`，若发现任何违例则整个 run 标记为 invalid。

---

## 5. 基础真实 selection pipeline

V4 的主 pipeline 先去掉 V3 中可能混入人工规则的 filter/expansion，得到可解释的真实模型基线。

```text
Original query / real user turn
        ↓
Qwen adaptive phase controller
        ↓ (only if retrieve)
BGE dense retrieval over this tenant's real prior memories, Top-R
        ↓
[method-dependent locality pruner]
        ↓ Top-B
MiniLM cross-encoder rerank
        ↓ Top-K context
Qwen reasoning / answer with explicit memory citations
```

### 5.1 固定主参数

```yaml
coarse_top_r: 200
rerank_budget_b: [16, 32, 50, 64]
final_context_k: 8
relation_expansion_depth: 0
metadata_filter_mode: tenant_and_time_only
reranker_batch_size_online: 1
controller_temperature: 0.0
answer_temperature: 0.0
```

解释：

- `Top-R=200` 只由真实 embedding 检索产生；
- 主比较固定 `Top-B`，不允许某方法多给 reranker 候选以获得质量优势；
- `relation_expansion_depth=0` 是为了避免旧 relation 图成为不透明先验；
- 只允许 tenant 和时间合法性过滤，不能使用答案相关 metadata；
- 如果某数据集实际可用候选少于 R，则记录真实可用数，不补造候选。

### 5.2 两个检索实现

必须跑：

1. `faiss_flat`：correctness / recall reference；
2. `faiss_hnsw`：realistic online ANN serving baseline。

现有 `ann_numpy` 可作为 V3 连续性对照保留，但不能是 V4 唯一 retrieval backend。

要求：

- 先用 `faiss_flat` 计算 Top-R reference；
- 为 `faiss_hnsw` 调整搜索参数，使其相对 `faiss_flat` 的 `Recall@200 >= 0.98`；
- 正式 latency 表中明确写出 backend、参数、实际 recall；
- 不在“ANN recall 未对齐”的条件下比较 end-to-end latency。

---

## 6. 真实历史 replay 与 trace 构建

### 6.1 为什么必须做 replay

多视图超图需要的是**过去真实访问行为**。不能用最终 QA 的答案/证据，也不能用人为生成的 query 来造 co-access。

对于每个真实对话 / memory timeline：

```text
历史原始 turn 1 → agent 处理 → 若 controller 触发则 selection → 写 trace
历史原始 turn 2 → agent 处理 → 若 controller 触发则 selection → 写 trace
...
最终官方 QA → 使用此时已经积累的 layout / hypergraph → 正式评测
```

这样，最终 QA 前的图结构只来自过去自然发生的原始对话 turn 与真实模型调用。

### 6.2 Replay 规则

- 只回放 role=`user` 的真实 turn；
- 每个 turn 的 query 为该 user 的原始 text，可附带同 session 的已发生上下文；
- memory pool 只包含严格早于当前 turn 的原始 memory；
- 初始 session 或没有历史 memory 时允许 `skip`；
- 所有模型输出固定 `temperature=0`；
- replay 过程中 agent 可产生简短 internal state，但不得使用原始 future assistant response、future session 或 QA label；
- 若 controller `skip`，仍写 trace，便于统计真实 retrieval invocation ratio；
- replay 和最终 QA 必须共享同一 tenant 的在线 hypergraph state；不得在 QA 前使用未来 replay event。

### 6.3 Hyperedge 的真实定义

对每个实际发生 selection 的 phase invocation，形成一个候选 group：

```text
H_t = unique(top-K reranked memory ids ∪ answer-cited memory ids)
```

条件 view 定义为：

```text
view = phase ∈ {planning, reasoning, output}
```

不使用人工 task tag、entity tag、gold group 或知识图谱标签。

只在满足以下条件时更新：

```text
len(H_t) >= 2
```

每条 hyperedge 至少记录：

```json
{
  "tenant_id": "...",
  "event_time": "...",
  "phase": "reasoning",
  "member_memory_ids": ["..."],
  "member_count": 4,
  "source": "real_replay | official_qa",
  "retrieval_goal_hash": "...",
  "rerank_budget": 32
}
```

### 6.4 超图在线更新要求

- 仅在 event `t` 结束后更新 `H_t`；
- event `t` 的候选选择只能使用 `H_{<t}`；
- 按 tenant 独立维护图；
- 用访问频率和时间衰减更新 weight，不允许依据正确答案加分；
- 对于新 tenant 或没有历史 edge 的 cold start，必须退化到 plain dense retrieval，而不是伪造关系；
- 每次 update 记录 edge add / merge / evict 数量、字节数和 elapsed time。

---

## 7. Phase B：多视图超图 candidate-pruning 方法

### 7.1 目标

目标不是替代 cross-encoder，而是在它之前将真实 Top-R embedding candidates 压缩到固定 Top-B，同时尽可能保持官方 evidence coverage 和最终 QA quality。

### 7.2 必须比较的方法

所有方法：

- 使用完全相同的原始 query；
- 使用相同 BGE、相同 ANN、相同 Qwen controller、相同 cross-encoder、相同 Top-R、相同 Top-B、相同 final context K；
- 只允许改变 `Top-R → Top-B` 的结构化候选 pruning 策略。

| ID | 方法 | 在线使用的结构 |
|---|---|---|
| M0 | `shared_once_plain` | 一次 selection，无结构图 |
| M1 | `adaptive_plain` | Qwen 自适应 phase，无结构图 |
| M2 | `adaptive_global_pair_graph` | 过去 trace 的全局 pairwise co-access |
| M3 | `adaptive_phase_pair_graph` | phase 分 view 的 pairwise co-access |
| M4 | `adaptive_global_hypergraph` | 不分 phase 的高阶 hyperedge |
| M5 | `adaptive_phase_multiview_hypergraph` | **完整方法：phase view + 高阶 hyperedge** |
| U0 | `oracle_group_pruned` | 仅离线理论上界；不得参与任何正式在线比较结论 |

### 7.3 多视图超图 pruning 规则

对每个 phase 的真实 dense Top-R：

1. 取 dense Top-8 作为 seed set `S`；
2. 在当前 tenant、当前 phase 的历史超图中查询与 `S` 相交的 hyperedge；
3. 对每个 candidate 计算：

```text
score(c)
= alpha * normalized_dense_score(c)
+ beta  * hyperedge_affinity(c, S, phase)
+ gamma * recency_weight(c)
```

4. `hyperedge_affinity` 只由过去 hyperedge 的 frequency、time decay、与 seed 的 overlap 得到；
5. 从 dense candidates 与 hypergraph-expanded candidates 的并集内取 Top-B；
6. 交给同一个真实 MiniLM cross-encoder rerank。

默认从不使用：

```text
gold evidence
answer text
answer session id
question category
hand-crafted task/entity overlap
future trace
cross-tenant memory
```

### 7.4 参数调优规则

- `alpha / beta / gamma` 只能在开发 split 或 replay 的前段真实 events 上调参；
- 不得在最终 QA test labels 上搜索最优参数；
- 先固定一个配置，例如：

```yaml
alpha: 0.70
beta: 0.25
gamma: 0.05
edge_time_decay_half_life_events: 100
seed_count: 8
max_hyperedges_per_seed: 32
```

- 若要扫描参数，保存完整 grid、开发集结果和最终选定规则；
- 若效果不显著，不得通过添加 gold metadata、人造 hard negative 或强制 phase invocation 来“修正”结果。

---

## 8. 正式实验矩阵

### 8.1 Stage A：真实自适应 phase trace 重新验证

| 数据 | 正式 query / event | 模式 | 目的 |
|---|---|---|---|
| LoCoMo | 全部原始 QA + 原始 turn replay | shared_once / fixed_three / adaptive_phase | 验证 V3 的受控放大与真实自适应差异 |
| LongMemEval-S | 全部 500 原始 QA + 原始 session replay | shared_once / fixed_three / adaptive_phase | 真实 QA、evidence、时间/更新类任务 |
| LongMemEval-M | 全部 500 原始 QA + 原始 session replay | adaptive_phase | 大规模端到端压力 |
| MSC valid | 全部原始 valid 对话事件，分 shard 执行 | adaptive_phase | 大规模真实在线 trace、tenant 隔离、更新稳定性 |

这一阶段先只使用 `adaptive_plain`，不启用 hypergraph pruning，生成可信的真实 trace。

### 8.2 Stage B：真实 trace 驱动的 pruning 评估

| 数据 | 主要方法 | 必报指标 |
|---|---|---|
| LoCoMo | M1–M5，B=16/32/50/64 | evidence recall、QA quality、rerank latency、payload bytes |
| LongMemEval-S | M1–M5，B=16/32/50/64 | session / turn recall、official QA score、abstention、latency |
| LongMemEval-M | M1/M3/M5，B=32/50 | end-to-end latency、tail latency、evidence recall、scale |
| MSC valid | M1/M3/M5，B=32/50 | locality prediction、candidate reduction、tenant isolation、update cost |

### 8.3 Stage C：模型 / index 健壮性（主结果完成后）

只在 LoCoMo 和 LongMemEval-S 上追加：

```text
BGE-base + MiniLM-L6   （主配置）
BGE-large + MiniLM-L6  （retriever sensitivity）
BGE-base + MiniLM-L12  （reranker sensitivity，如本地可用）
faiss_flat / faiss_hnsw
```

该阶段所有输入仍必须是原始真实 query / memory。不要把模型敏感性替换成 synthetic stress test。

---

## 9. 评价指标与正确口径

### 9.1 Evidence 与 QA 质量

#### LoCoMo

- `DialogEvidenceRecall@B`：Top-B 是否覆盖官方 `qa.evidence` dialog ids；
- `DialogEvidenceRecall@K`：最终 Top-K context 的 evidence coverage；
- `EvidenceGroupCoverage`：一个问题所有可用 evidence 是否被联合覆盖；
- 官方 QA metric；
- 按官方 category 离线分桶的结果，category 不得进入 runtime。

#### LongMemEval

- `SessionRecall@B`：Top-B 是否覆盖 `answer_session_ids`；
- `TurnRecall@B`：Top-B 是否覆盖 `has_answer=true` turn；
- `SessionRecall@K`、`TurnRecall@K`；
- 官方 QA evaluator 得分；
- abstention 单独报 accuracy / F1；
- 按 official `question_type` 离线分桶，类型不得进入 runtime。

#### MSC

- 不报告伪造的 gold evidence recall；
- 报告真实 trace 的 `retrieval_invocation_ratio`、candidate reduction、phase-conditioned next-access prediction、layout update cost；
- 可选报告真实 next response 的语义相似性，但必须作为辅助指标，不能替代 LoCoMo / LongMemEval 的 gold-evidence 结论。

### 9.2 性能与成本

主口径：

```text
phase_selection_total_latency_ms
phase_retrieval_total_latency_ms
phase_rerank_total_latency_ms
total_response_latency_ms
```

另外必须输出：

```text
selection_invocation_count
retrieval_invocation_ratio
pre_rerank_candidate_count
rerank_candidate_count
candidate_reduction_ratio
final_context_count
logical_header_bytes_scanned
logical_payload_bytes_fetched
reranker_input_bytes
controller_latency_ms
embedding_latency_ms
ann_latency_ms
pruner_latency_ms
rerank_latency_ms
context_build_latency_ms
p50 / p95 / p99 per-query latency
```

注意：

- V4 中 `logical_payload_bytes_fetched` 只能表示依据真实输入文本长度统计出的逻辑数据量；
- 禁止将该指标写为真实 DRAM bus traffic、PIM traffic 或 energy；
- offline graph build、index build、模型加载不可混入 per-query online latency；单独记录为 setup / maintenance cost。

### 9.3 统计要求

- 正确性与延迟均保留逐 query 记录；
- 对方法间差异使用 paired bootstrap 95% CI；
- latency 额外报告 p50 / p95 / p99；
- 对所有 formal run 使用相同机器、相同模型副本、相同 CUDA / CPU affinity 设置；
- 在每次计时前执行 `torch.cuda.synchronize()`，计时后再次同步；
- 首次 model / index warm-up 单独标记为 cold-start，不得混入 warm-service 主表；
- quality 用固定 seed 单次生成；selection timing 用同一批原始真实 query 重复 5 次，真实模型每次实际执行，报告中位数与分位数。

---

## 10. 必须新增或修改的代码结构

建议目录：

```text
experiments/motivation_memory_selection/
├── build_v4_real_manifest.py
├── replay_real_histories.py
├── profile_real_agent_v4.py
├── real_dataset_adapters.py
├── adaptive_phase_controller.py
├── online_multiview_hypergraph.py
├── evaluate_phase_locality_v4.py
├── evaluate_pruning_v4.py
├── evaluate_official_qa_v4.py
├── validate_no_leakage.py
├── verify_tenant_isolation.py
├── report_real_trace_v4.py
└── configs/
    ├── v4_locomo_shared_once.yaml
    ├── v4_locomo_fixed_three.yaml
    ├── v4_locomo_adaptive_plain.yaml
    ├── v4_locomo_adaptive_phase_hypergraph.yaml
    ├── v4_longmemeval_s_adaptive_plain.yaml
    ├── v4_longmemeval_s_adaptive_phase_hypergraph.yaml
    ├── v4_longmemeval_m_adaptive_phase_hypergraph.yaml
    └── v4_msc_adaptive_phase_hypergraph.yaml

src/
├── agent_executor.py                  # 仅最小接口扩展；保留 V3 行为兼容
├── embedding_client.py
├── reranker_client.py
├── vector_store.py                    # 增加 faiss_flat / faiss_hnsw 和 tenant namespace
└── trace_schema.py

versions/20260621_real_trace_adaptive_phase_v4_formal/
├── README_zh.md
├── configs_snapshot/
├── manifests/
├── runs/
├── reports/
└── figures/
```

### 10.1 不要做的改动

- 不要在 V3 代码上直接覆盖删除已有逻辑；
- 不要更改 V3 configs 或 outputs；
- 不要删除 `synthetic_memory_generator.py`，但必须使其不在 V4 formal import path；
- 不要把 hypergraph 逻辑塞进 dataset importer；它必须只消费 runtime trace；
- 不要在 `agent_executor.py` 内部偷偷读取 evaluator labels。

### 10.2 关键数据文件

#### `runtime_manifest.jsonl`

只包含 online 允许读取的数据：

```json
{
  "dataset": "longmemeval_s",
  "tenant_id": "...",
  "timeline_id": "...",
  "event_id": "...",
  "event_type": "replay_turn | official_qa",
  "event_time": "...",
  "query_text": "...",
  "visible_context": ["..."],
  "memory_ids_available": ["..."],
  "memory_source_paths": ["..."],
  "runtime_allowed_fields": ["text", "time", "session_id", "speaker", "tenant_id"]
}
```

#### `evaluation_labels.jsonl`

只能由 evaluator 打开：

```json
{
  "dataset": "longmemeval_s",
  "event_id": "...",
  "reference_answer": "...",
  "gold_session_ids": ["..."],
  "gold_turn_ids": ["..."],
  "official_question_type": "..."
}
```

#### `phase_trace.jsonl`

每个真正发生或跳过的 phase 都要写一行：

```json
{
  "run_id": "...",
  "dataset": "...",
  "tenant_id": "...",
  "event_id": "...",
  "event_time": "...",
  "phase": "planning",
  "controller_action": "retrieve | skip",
  "controller_goal": "...",
  "controller_latency_ms": 0.0,
  "available_memory_count": 0,
  "dense_top_r_ids": ["..."],
  "pruned_top_b_ids": ["..."],
  "reranked_top_k_ids": ["..."],
  "answer_cited_memory_ids": ["..."],
  "phase_selection_latency_ms": 0.0,
  "embedding_latency_ms": 0.0,
  "ann_latency_ms": 0.0,
  "pruner_latency_ms": 0.0,
  "rerank_latency_ms": 0.0,
  "logical_header_bytes_scanned": 0,
  "logical_payload_bytes_fetched": 0,
  "cross_tenant_candidate_count": 0,
  "hypergraph_edge_count_before": 0,
  "hypergraph_edge_count_after": 0,
  "runtime_input_fields": ["..."]
}
```

---

## 11. 建议的执行顺序

### Step 1：构建真实 manifest 与 evaluator label 隔离

实现：

```text
build_v4_real_manifest.py
real_dataset_adapters.py
validate_no_leakage.py
verify_tenant_isolation.py
```

验收：

- LoCoMo、LongMemEval-S、LongMemEval-M、MSC 均产生 runtime manifest；
- formal manifest 中 `source_is_synthetic=false`；
- `evaluation_labels.jsonl` 与 runtime manifest 分开保存；
- runtime 代码无法 import label 文件；
- MSC 全部 candidate 的 `tenant_id` 与 query tenant 相同；
- 输出 `data_provenance_report.md`，列明每个字段来自原始数据、真实模型还是 evaluator-only label。

### Step 2：实现真实自适应 phase controller 与三模式比较

实现：

```text
adaptive_phase_controller.py
profile_real_agent_v4.py
```

先在少量**真实**样本上做 smoke test：

```text
LoCoMo: 16 原始 QA + 对应 history replay
LongMemEval-S: 16 原始 QA + 对应 history replay
MSC: 128 原始 replay events
```

smoke test 仅验证代码正确性，不进入论文结果。

验收：

- `shared_once` 恰好一次 selection；
- `fixed_three` 恰好三个 phase selection；
- `adaptive_phase` 的 retrieval count 在 0–3 之间，且 trace 内真实记录 `retrieve/skip`；
- phase controller 完全没有固定 suffix；
- no-leakage audit 为 PASS；
- 每条 trace 的 online latency 只包含真正执行的阶段；
- output phase 默认可 skip，而不是强制三次。

### Step 3：生成正式真实 replay trace（先 plain，无超图）

先跑 `adaptive_plain`，并分别保存各数据集的 chronological trace。

建议执行命令模板：

```bash
cd /data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling

python experiments/motivation_memory_selection/build_v4_real_manifest.py \
  --dataset locomo \
  --raw-data <LOCOMO_RAW_JSON> \
  --output versions/20260621_real_trace_adaptive_phase_v4_formal/manifests/locomo

python experiments/motivation_memory_selection/replay_real_histories.py \
  --config experiments/motivation_memory_selection/configs/v4_locomo_adaptive_plain.yaml \
  --manifest versions/20260621_real_trace_adaptive_phase_v4_formal/manifests/locomo/runtime_manifest.jsonl \
  --output-dir versions/20260621_real_trace_adaptive_phase_v4_formal/runs/locomo_adaptive_plain

python experiments/motivation_memory_selection/profile_real_agent_v4.py \
  --config experiments/motivation_memory_selection/configs/v4_locomo_adaptive_plain.yaml \
  --manifest versions/20260621_real_trace_adaptive_phase_v4_formal/manifests/locomo/runtime_manifest.jsonl \
  --output-dir versions/20260621_real_trace_adaptive_phase_v4_formal/runs/locomo_adaptive_plain
```

LongMemEval-S/M、MSC 采用同样流程，配置和 output 目录分开。

验收：

- 所有 trace 事件来自原始 turn 或原始 QA；
- `cross_tenant_candidate_count == 0`；
- memory 在 replay 时严格只含过去内容；
- 输出真实 `retrieval_invocation_ratio`，而不是预设 3 次；
- 输出 adaptive 与 fixed-three、shared-once 的质量 / 成本对比。

### Step 4：真实 phase locality 分析

实现：

```text
evaluate_phase_locality_v4.py
```

分析内容：

1. phase 内 vs phase 间 memory-set overlap；
2. 真实 replay trace 中的 co-access recurrence；
3. global pairwise、phase pairwise、global hypergraph、phase hypergraph 对“下一次真实 selection 中最终 Top-K”的预测能力；
4. pairwise projection 的伪连接数量；
5. cold-start 与长期运行时的 hyperedge coverage。

必须生成：

```text
fig_phase_conditioned_overlap.pdf
fig_phase_access_prediction.pdf
fig_pairwise_projection_loss.pdf
fig_hyperedge_recurrence.pdf
table_trace_statistics.csv
```

注意：这些是对**真实 agent access behavior**的结构分析，不替代 LoCoMo / LongMemEval 的 gold evidence 质量评估。

### Step 5：实现并评估在线多视图超图 pruner

实现：

```text
online_multiview_hypergraph.py
evaluate_pruning_v4.py
```

严格按 M1–M5 对比；先在 LoCoMo 和 LongMemEval-S 上完整运行，再扩展至 LongMemEval-M 和 MSC。

验收：

- 在相同 Top-R、Top-B、cross-encoder、Qwen 条件下对比；
- U0 oracle 单独出图或虚线，不与可部署方法混写；
- 未发生 layout history 的 cold start 自动回退 `adaptive_plain`；
- 所有 graph / hypergraph 更新使用过去 trace；
- 运行结束后 leakage audit 再次为 PASS。

### Step 6：官方 QA 评估、统计、报告

实现：

```text
evaluate_official_qa_v4.py
report_real_trace_v4.py
```

需要生成：

```text
reports/v4_real_agent_experiment_report.md
reports/leakage_audit.md
reports/data_provenance_report.md
reports/tenant_isolation_report.md
reports/latency_methodology.md
reports/official_qa_metrics.md
reports/significance_tests.md
figures/*.pdf
summary.csv
summary.md
run_manifest.json
```

---

## 12. 最终图表与论文对应关系

| 图/表 | 回答的论文问题 | 数据与模式 |
|---|---|---|
| Fig. 1 Adaptive vs fixed-three vs shared-once | V3 的瓶颈是否在真实自适应 agent 下仍存在 | LoCoMo + LongMemEval-S |
| Fig. 2 Latency breakdown / tail latency | selection 的哪一段真正昂贵 | LoCoMo / LongMemEval-S/M |
| Fig. 3 Phase-conditioned locality | phase view 是否真实存在 | replay trace，三数据集 |
| Fig. 4 Pairwise projection loss | 为什么仅用普通图不足 | replay trace |
| Fig. 5 Quality–rerank-budget Pareto | 多视图超图是否在更小 B 下保留 evidence / QA | LoCoMo + LongMemEval-S |
| Fig. 6 Candidate / payload reduction | 是否实质缩小 rerank frontier 与逻辑搬运量 | LoCoMo + LongMemEval-S/M |
| Fig. 7 Online update / cold-to-warm | 长期 trace 积累后收益是否增加 | MSC + LongMemEval-M |
| Table 1 Official QA and evidence metrics | 方法是否保持真实 benchmark 质量 | LoCoMo + LongMemEval-S |
| Table 2 System cost and tail latency | 真实模型条件下的性能收益 | LoCoMo + LongMemEval-S/M + MSC |
| Table 3 Ablation | phase、超边、动态更新各自的作用 | LoCoMo + LongMemEval-S |

---

## 13. 结果解释边界

V4 完成后，可以严谨声称：

> 在真实公开多轮对话与官方 QA workload 中，真实 LLM agent 会以不固定次数触发 phase-dependent memory selection；过去真实 interaction trace 中存在可复用的 phase-conditioned high-order co-access；在不读取答案或证据标签的前提下，在线多视图超图可在固定 rerank budget 下压缩 candidate frontier，并保持或接近官方 evidence / QA 质量。

V4 完成后，**仍不能**声称：

- 已经证明真实 PIM hardware speedup；
- `logical_payload_bytes` 就是 DRAM/PIM bus traffic；
- 所有生产 Agent 一定需要三次 retrieval；
- MSC 的 session grouping 是 QA gold evidence；
- oracle upper bound 是可部署系统。

这些结论必须等到后续 physical layout / PIM experiment 才能进入论文。

---

## 14. Codex 最终交付检查清单

在完成前，逐项确认：

- [ ] V3 不被修改或覆盖；
- [ ] V4 formal path 不 import `synthetic_memory_generator.py`；
- [ ] 所有 runtime query 可以追溯到原始公开数据的 QA 或 user turn；
- [ ] 所有 runtime memory 可以追溯到原始 turn / session；
- [ ] LoCoMo / LongMemEval evaluator labels 与 runtime data 隔离；
- [ ] `gold_group_id` 不在 V4 selector / graph / prompt / reranker feature 中出现；
- [ ] MSC 检索严格 tenant-isolated；
- [ ] shared_once、fixed_three、adaptive_phase 是互斥独立运行模式；
- [ ] adaptive phase 由真实 Qwen 输出 `retrieve/skip`，而非 suffix 强制调用；
- [ ] graph/hypergraph 在每个 tenant 内按时间在线更新；
- [ ] evaluation 至少包含 evidence quality、official QA quality、latency、candidate count、logical payload bytes；
- [ ] `faiss_flat` 与 `faiss_hnsw` recall 对齐后再做 latency 比较；
- [ ] 所有 formal run 产出 config snapshot、模型版本、数据 hash、seed、硬件信息和 raw trace；
- [ ] 生成完整报告，但绝不编造结果；缺失项必须标为未完成而不是估计值。
