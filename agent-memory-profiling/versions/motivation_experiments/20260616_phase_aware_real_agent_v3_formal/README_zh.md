# Phase-Aware Real-Agent V3 实验说明

## 1. 这份版本包是做什么的

这个版本包是当前正式的 `v3` 实验资产，服务于你的论文动机：

- 不把 agent 简化成一次离线检索
- 保留 agent 在真实执行过程中的阶段性 memory 调用
- 观察真正的热点到底在粗检索、候选扩展、候选聚合，还是 rerank
- 为后续“基于 phase locality 构造多视图 / 超图 / PIM layout 优化”提供真实 trace 依据

这份文档的目标不是只报结果，而是把当前实验的实现逻辑讲清楚，避免它对你来说是黑盒。

## 2. 当前版本目录结构

当前正式 `v3` 版本位于：

- [versions/20260616_phase_aware_real_agent_v3_formal](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal)

其中：

- 实验输出在 [runs](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/runs)
- 配置快照在 [configs_snapshot](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot)
- 总报告在 [phase_aware_real_agent_v3_experiment_report_20260616.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/phase_aware_real_agent_v3_experiment_report_20260616.md)

本版本主要依赖的代码路径：

- [profile_longgraph_memory_selection.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)
- [agent_executor.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/agent_executor.py)
- [reranker_client.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/reranker_client.py)
- [embedding_client.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/embedding_client.py)
- [vector_store.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/vector_store.py)
- [import_public_dataset.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/import_public_dataset.py)
- [codex_phase_aware_handoff_20260615.md](/data2/user25263093/userliteng/26_code/preTest/codex_phase_aware_handoff_20260615.md)

## 3. 这个实验到底在测什么

这个实验测的是：

- 在真实三阶段 agent 执行路径中，memory selection 的成本到底有多大
- 这个成本是主要来自 retrieval，还是来自 rerank
- 当 memory selection 不再是“全流程只调一次”，而是随着 agent phase 重复触发时，总成本会不会被放大到接近 agent LLM 本身

当前 agent 侧保留了三个阶段：

1. `planning`
2. `reasoning`
3. `output`

每一次 memory selection 内部又细分为：

1. query encoding
2. coarse retrieval
3. metadata filtering
4. relation expansion
5. candidate aggregation / dedup
6. rerank
7. context construction

因此，这个实验不是只看“搜出来多快”，而是把完整 memory selection path 作为研究对象。

## 4. 为什么要有 `phase_aware`

你的论文核心不是一句“memory retrieval 很慢”，而是更具体的判断：

- agent 的不同执行阶段，对 memory 的调用局部性不同
- 这种 phase-specific locality 应该被建模
- 后续才能进一步映射成不同 view / hypergraph / memory layout
- 然后再讨论 PIM 上该把哪些算子下推、怎样重排 layout

所以 `phase_aware` 的意义在于：

- 不把所有阶段压扁成一个共享 memory context
- 而是显式保留 `planning / reasoning / output` 各自的 memory selection 行为

当前实现中，phase-specific query suffix 如下：

- `planning`: 强调计划建立、约束、长期连续性
- `reasoning`: 强调证据关联、近期变化、因果支撑
- `output`: 强调最终作答所需的直接支撑

这还不是一个学习式 phase controller，而是一个受控的 prompt-level 方案，用来先暴露三阶段 memory locality 差异。

## 5. `shared_once` 和 `phase_aware` 的区别

代码中有两种 memory 调用模式：

- `shared_once`
- `phase_aware`

### 5.1 `shared_once`

含义是：

- 整个 agent 响应过程只做一次 selection
- 后续 planning / reasoning / output 共享这一次选出来的 memory context

这种设置更接近一个保守 baseline，也更接近你之前那类“单次检索 + 单次 rerank + 再执行 agent”的实验。

### 5.2 `phase_aware`

含义是：

- `planning` 单独做一次 selection
- `reasoning` 单独做一次 selection
- `output` 单独做一次 selection
- 每个阶段都会写一条 `phase_trace.jsonl`

也就是说，在当前实现中，三个阶段确实都会重新做一次 retrieval 再 rerank。

### 5.3 为什么这样设置是合理的

这个设置不是在声称“所有生产级 agent 都必须这样设计”，而是在说：

- 如果你的论文要证明 phase-specific locality 是真实存在并且有意义的
- 那实验里就必须把这些 phase-specific memory 调用真实地暴露出来

否则最后测到的只会是一个“共享一次 memory”的平均化结果，无法支持你后续的超图 / 多视图 / PIM 动机。

## 6. 当前 `v3` 的整体执行流程

对于每个 query，当前 profiler 的执行流程是：

1. 读取真实数据集转换后的 memory JSONL
2. 构建或加载 ANN 索引
3. 根据 query type 构造 `QuerySpec`
4. 先跑一次普通 selection pass
5. 如果 `agent_memory_mode = phase_aware`，再为 `planning / reasoning / output` 各跑一次 phase-specific selection pass
6. 将各阶段选出的 memory context 传给真实 LLM agent executor
7. 写出：
   - `raw_profile.jsonl`
   - `phase_trace.jsonl`
   - `summary.csv`
   - `summary.md`
   - `figures`

主控逻辑在：

- [_run_single_query(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

## 7. 检索 backbone 是什么

### 7.1 Embedding 模型

正式 `v3` 全部使用：

- `embedding_backend = transformer`
- 模型目录：[models/bge-base-en-v1.5](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/models/bge-base-en-v1.5)

实际实现类是：

- [TransformerEmbeddingClient](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/embedding_client.py)

具体过程是：

1. 用 HuggingFace tokenizer 对文本分词
2. 调用 `AutoModel`
3. 对最后一层 hidden state 做 mean pooling
4. 用 attention mask 去掉 padding 干扰
5. 最后做 L2 normalize

配置中使用：

- `embedding_dim = 768`

因此这是一个标准 dual-encoder 检索流程：

- query 独立编码
- memory 独立编码
- 最后用向量相似度做匹配

### 7.2 Coarse retrieval

正式 `v3` 使用：

- `retrieval_backend = ann_numpy`

实现类是：

- [AnnNumpyVectorStore](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/vector_store.py)

它的做法不是外部向量库 benchmark，而是仓库内自带的轻量 ANN：

1. 对 memory vectors 做归一化
2. 构建轻量 centroid / cluster 索引
3. query 先和 centroid 匹配
4. 只在 probe 到的 cluster 内打分
5. 如果候选覆盖不够，就 fallback 到全扫描

这个设计的目的不是拿它和 FAISS 拼最优吞吐，而是为了让检索阶段的代价、候选规模和后续 rerank 压力在同一套可控实现里被观察到。

## 8. Rerank 之前还做了什么

coarse retrieval 之后，当前 selection path 还包括：

1. metadata filter
2. relation expansion
3. candidate aggregation / dedup

### 8.1 Metadata filter

实现位置：

- [_metadata_filter(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

它会根据 query 的：

- `task_tags`
- `entity_tags`
- `preferred_memory_types`

过滤候选。

不同 query type 的过滤逻辑不同，例如：

- `fact` 更偏向类型匹配和 task/entity overlap
- `temporal` 更宽松
- `relation` 更强调关联结构
- `plan_continuity` 更强调 procedural / profile / episodic
- `constraint` 更强调 constraint / profile

### 8.2 Relation expansion

实现位置：

- [_expand_relations(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

它会按 `relation_expansion_depth` 沿 memory relation 图向外扩展。

扩展后的分数会衰减：

```text
next_score = inherited_score * (0.92 ** (current_depth + 1))
```

直观含义是：

- 直接命中的 memory 最相关
- 关系扩展出来的 memory 有价值，但相关性会逐层衰减

### 8.3 Candidate aggregation / dedup

实现位置：

- [_aggregate_candidates(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

它会把：

- metadata filter 的结果
- relation expansion 的结果

按 `memory_id` 合并，保留更高的分数，再重新排序。

这一步很重要，因为你的论文动机并不只是“向量检索”，而是包含结构化 memory 访问，所以 candidate fan-out 本身就是热点来源之一。

## 9. `heuristic` rerank 到底是什么

### 9.1 实现位置

- [HeuristicRerankerClient](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/reranker_client.py)
- [_heuristic_rerank(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)
- [_rerank_score(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

### 9.2 具体打分公式

当前 `heuristic` 的 rerank 分数是：

```text
score
= vector_score
+ 0.22 * importance
+ 0.16 * recency
+ 0.10 * task_overlap
+ 0.08 * entity_overlap
+ relation_bonus
```

各项含义：

- `vector_score`: coarse retrieval 阶段得到的相似度
- `importance`: memory 元数据中的重要性
- `recency`: memory 元数据中的时效性
- `task_overlap`: query 和 memory 的 task tag 重叠数
- `entity_overlap`: query 和 memory 的 entity tag 重叠数
- `relation_bonus`: 基于 gold group 和 query type 的额外奖励

额外 bonus 规则包括：

- 如果 `memory.gold_group_id` 属于 `query.gold_group_ids`，加 `0.18`
- `relation` query 再额外加 `0.12`
- `plan_continuity` query 再额外加 `0.08`
- `constraint` query 且 memory 类型是 `constraint` 时，再额外加 `0.12`

### 9.3 它的本质

`heuristic` 不是神经 reranker。

它本质上是一个：

- 基于向量分数
- 再加一些结构化 metadata
- 再做线性重打分

的轻量级 rerank baseline。

保留它的原因是：

- 给出一个低成本 rerank 对照组
- 帮助你区分“rerank 本身就很重”还是“retrieval / expansion 已经足够重”
- 为 cross-encoder 的高成本 rerank 提供对照

## 10. `cross_encoder` 到底是什么

### 10.1 当前正式实验使用的模型

正式 `cross_encoder` 运行使用：

- [models/cross-encoder-ms-marco-MiniLM-L6-v2](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/models/cross-encoder-ms-marco-MiniLM-L6-v2)

本地 `config.json` 显示它是：

- `BertForSequenceClassification`
- `model_type = bert`
- `num_hidden_layers = 6`
- `hidden_size = 384`

模型卡说明它是：

- 基于 MS MARCO Passage Ranking 训练的 cross-encoder reranker
- 用于 Information Retrieval 中的 retrieve-and-rerank 场景

参考：

- [models/cross-encoder-ms-marco-MiniLM-L6-v2/README.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/models/cross-encoder-ms-marco-MiniLM-L6-v2/README.md)

### 10.2 具体实现

实现位置：

- [CrossEncoderRerankerClient](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/reranker_client.py)
- [_cross_encoder_rerank(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

流程是：

1. 构造 `(query_text, memory_text)` 对
2. 用 tokenizer 对 query 和 passage 联合编码
3. 调用 `AutoModelForSequenceClassification`
4. 取 relevance logit
5. 再和一个很小的 vector_score 残差项结合

最终分数为：

```text
combined_score = model_score + 0.05 * vector_score
```

### 10.3 它和 embedding retriever 的原理区别

embedding 检索是 dual encoder：

- query 单独编码
- passage 单独编码
- 最后再做向量相似度

cross-encoder 是 joint encoder：

- query 和 passage 一起送进同一个 transformer
- 模型直接输出 relevance 分数

所以：

- 准确率通常更高
- 但代价显著更大

这恰恰是你当前实验需要的，因为你就是要看：

- 在 phase-aware 场景下，rerank 会不会被放大成真正的主热点

### 10.4 为什么正式 cross-encoder 运行用 `reranker_batch_size = 1`

当前正式配置里：

- `reranker_batch_size = 1`

原因不是说线上系统永远应该 batch size = 1，而是：

- 这样更容易暴露保守 online serving 下的 rerank 单次成本
- 不会因为大 batch 把代价摊薄，导致热点被掩盖

换句话说，这是一个 profiling 取向的配置，而不是吞吐量最优配置。

## 11. `oracle_pruned_rerank` 是什么

### 11.1 实现逻辑

实现位于：

- [_run_selection_pass(...)](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/profile_longgraph_memory_selection.py)

它的逻辑是：

1. 先正常完成 retrieval、filter、expansion、aggregation
2. 把候选分成两部分：
   - `oracle`: `gold_group_id` 命中 query 目标 group 的候选
   - `spill`: 其余候选
3. 重新拼成 `oracle + spill`
4. 再截断到 `oracle_prune_limit`
5. 只对这个截断后的集合做 rerank

正式配置里：

- `oracle_prune_limit = 50`

### 11.2 为什么要设计这个 baseline

它不是部署方案，而是分析工具。

它回答的问题是：

- 如果我们能在 rerank 前，用更聪明的结构化 pruning，把真正相关的 memory group 更早保留下来，那么 rerank 的成本最多能被压缩多少

这和你的论文动机是高度一致的，因为你最终想做的不是“一个更强的通用 reranker”，而是：

- 利用 memory locality / structure
- 在更靠前的位置做更有针对性的 selection / pruning / layout 优化

所以 `oracle_pruned_rerank` 的意义是：

- 它不是最终系统
- 它是“前置智能剪枝的上界式参考”

### 11.3 `gold_group_id` 不是随便加的

`gold_group_id` 在数据导入时就被赋值了：

- `LoCoMo`: `gold_group_id = conversation_id`
- `MSC`: `gold_group_id = session_id` 或 session-like grouping
- `LongMemEval`: `gold_group_id = sample_id`

对应代码在：

- [import_public_dataset.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/import_public_dataset.py)

所以 oracle 不是凭空作弊，而是利用数据集原本就存在的会话 / 样本级 grouping，作为“理想结构化 locality”的代理信号。

## 12. 当前 agent 路径有多“真实”

agent 执行逻辑在：

- [agent_executor.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/src/agent_executor.py)

正式 `v3` 的重要设置是：

- `agent_execution_backend = llm`
- `generation_backend = local_qwen`
- `agent_planning_llm_rounds = 1`
- `agent_reasoning_llm_rounds = 1`
- `agent_output_llm_rounds = 1`

也就是说：

- planning 有真实 LLM 调用
- reasoning 有真实 LLM 调用
- output 有真实 LLM 调用

同时 tool execution 阶段被简化成：

- 对已选 memory 的确定性 inspection

这样做的原因是：

- 避免外部 IO 干扰
- 把热点集中在 memory selection 和 agent 内部阶段上

所以这不是一个完整的复杂工具代理 benchmark，但它是一个保留三阶段结构、且具备真实 LLM 代价的受控 agent pipeline。

## 13. 当前正式实验的数据规模

正式 `v3` 使用的真实 memory 文件如下：

| 数据集 | 文件 | 实际 memory 条数 |
|:--|:--|--:|
| LoCoMo | `data/motivation_memory_selection/locomo_memories_1000.jsonl` | 5882 |
| LongMemEval | `data/motivation_memory_selection/longmemeval_memories_1000.jsonl` | 23867 |
| MSC | `data/motivation_memory_selection/msc_memories_5000.jsonl` | 395934 |

这里要特别注意：

- 文件名里的 `1000` 或 `5000` 来自更早的导出流程命名
- 但正式 `v3` 配置中的 `memory_sizes` 已经设置成真实条数
- 实验实际跑的规模是上表里的真实行数，不是文件名后缀

## 14. Query workload 是怎么构造的

当前实验使用五种 query type：

- `fact`
- `temporal`
- `relation`
- `plan_continuity`
- `constraint`

即使使用真实 memory 文件，系统内部仍然会构造统一的 `QuerySpec`，其中包括：

- `query_id`
- `query_type`
- `task_tags`
- `entity_tags`
- `preferred_memory_types`
- `temporal_window_days`
- `gold_group_ids`

这样做的目的是：

- 用统一接口驱动不同数据集
- 保持可比较的 selection profiling 框架

相关代码在：

- [synthetic_memory_generator.py](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/experiments/motivation_memory_selection/synthetic_memory_generator.py)

## 15. 当前正式 `v3` 比较的 baseline

当前正式版本主要比较两种 baseline mode：

- `full_selection`
- `oracle_pruned_rerank`

### 15.1 `full_selection`

这是完整路径：

1. retrieve
2. filter
3. expand
4. aggregate
5. rerank
6. 取 top-k

它代表“正常做完整 selection”的成本。

### 15.2 `oracle_pruned_rerank`

它保留前面的 selection 流程，但在 rerank 前先做 oracle 剪枝。

它代表“如果未来我们能把 rerank 前候选集压小很多，理论上能省多少”的分析型参考。

## 16. 日志里的时间到底怎么定义

这个部分最容易混淆。

### 16.1 `selection_total_latency_ms`

它只表示：

- 第一轮 selection pass 的时间

在 `phase_aware` 模式下，它不是整个 agent 真正支付的总 selection 成本。

### 16.2 `phase_selection_total_latency_ms`

它表示：

- `planning` selection
- `reasoning` selection
- `output` selection

三者的总和。

所以在当前正式 `phase_aware` 运行里，真正应该用于论文表达的“selection 总成本”是：

- `phase_selection_total_latency_ms`

同理：

- `phase_retrieval_total_latency_ms` = 三阶段 retrieval-like 部分总和
- `phase_rerank_total_latency_ms` = 三阶段 rerank 总和

### 16.3 `total_response_latency_ms`

代码里定义为：

```text
total_response_latency_ms
= phase_selection_total_latency_ms
+ agent_execution_total_latency_ms
```

所以时间占比也是围绕这个总量计算的。

例如：

- `phase_selection_to_total_response_ratio`
- `phase_retrieval_to_total_response_ratio`
- `phase_rerank_to_total_response_ratio`

这些才是你在论文里讨论“selection / retrieval / rerank 占总响应时间多少”的正确口径。

## 17. 为什么 cross-encoder 下热点会转向 rerank

因为在 `phase_aware` 模式下，selection 不再只执行一次，而是三次。

于是总成本近似变成：

```text
总 selection 成本
~ 3 * (retrieval-like cost + rerank cost + context packaging)
```

当 rerank 很轻时，比如 `heuristic`：

- selection 总成本在小数据集上还是比较轻
- 但在更大数据集上，retrieval / filter / aggregation 也会开始可见

当 rerank 很重时，比如 `cross_encoder`：

- 每个 phase 的 rerank 都很贵
- 三个 phase 再叠加一次
- 最后 selection 总成本就会被放大到和 agent LLM 同量级

这正是你这次实验和之前“单次 selection + agent”的关键区别。

## 18. 当前正式 `v3` 六组运行

当前版本一共保留了六组正式实验：

| 数据集 | Rerank | 输出目录 |
|:--|:--|:--|
| LoCoMo | heuristic | `runs/motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615` |
| LoCoMo | cross_encoder | `runs/motivation_memory_selection_phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615` |
| LongMemEval | heuristic | `runs/motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_heuristic_rerank_20260616` |
| LongMemEval | cross_encoder | `runs/motivation_memory_selection_phase_aware_real_agent_v3_longmemeval_full_cross_encoder_rerank_20260616` |
| MSC | heuristic | `runs/motivation_memory_selection_phase_aware_real_agent_v3_msc_full_heuristic_rerank_20260616` |
| MSC | cross_encoder | `runs/motivation_memory_selection_phase_aware_real_agent_v3_msc_full_cross_encoder_rerank_20260616` |

## 19. 当前结果应该怎么理解

这版 `v3` 给出的核心结论不是“所有 memory 优化都该去做 retrieval”。

更准确的结论是：

- 当你保留 phase-aware 三阶段 memory 调用后，完整 selection path 会变成真正的一阶成本项
- 在这种设置下，真正最重的热点常常不是 coarse retrieval，而是 rerank
- 如果能在 rerank 前做更聪明的结构化 candidate pruning，收益会明显大于只优化 ANN lookup

这对你的论文是有价值的，因为它把优化目标从“泛泛的 memory retrieval”收缩成了更具体的：

- phase-specific locality
- candidate frontier shaping
- rerank-adjacent primitives
- host-side data movement 和 layout 重排

## 20. 这版实验的局限性

这版实验足够支撑动机，但还不是最终架构论文的完整系统结果。

当前局限包括：

- phase query rewriting 还是模板驱动，不是学习式
- tool stage 被简化成 deterministic memory inspection
- `oracle_pruned_rerank` 是分析 baseline，不是部署算法
- 检索后端是仓库内 ANN 实现，不是生产级向量库对比

这些局限是可接受的，因为你当前主要目标是：

- 证明 phase-aware memory trace 有意义
- 证明热点主要落在 selection / rerank，而不是单纯 coarse retrieval

## 21. 推荐优先查看的文件

如果后面你要快速引用当前版本，优先看这几个：

- [README_zh.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/README_zh.md)
- [phase_aware_real_agent_v3_experiment_report_20260616.md](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/phase_aware_real_agent_v3_experiment_report_20260616.md)
- [phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615.yaml](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_locomo_full_cross_encoder_rerank_20260615.yaml)
- [phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615.yaml](/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/20260616_phase_aware_real_agent_v3_formal/configs_snapshot/phase_aware_real_agent_v3_locomo_full_heuristic_rerank_20260615.yaml)

