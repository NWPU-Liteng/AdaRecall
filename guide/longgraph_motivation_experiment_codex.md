# Codex 执行文档：LongGraph 中的 Agent Memory Selection 动机实验

## 0. 执行前要求

在开始实现前，请先阅读当前项目中的已有说明和配置，尽量复用现有环境、脚本和目录结构，不要重复创建环境。

请优先检查并阅读以下文件，若某些文件不存在则跳过：

```bash
README.md
codex/README.md
codex/*.md
docs/*.md
requirements.txt
pyproject.toml
environment.yml
.env.example
config/*.yaml
configs/*.yaml
```

本任务只做“动机实验”，不要实现完整 PIM 系统、不要实现超图布局算法、不要修改 LongGraph 主流程。目标是用实验回答一个问题：

> 在长期 Agent Memory 场景下，memory selection 是否真的会产生足够明显的系统开销，并且这种开销是否主要表现为候选膨胀、rerank 压力和主机侧数据访问，而不是简单的 embedding retrieval 延迟？

## 1. 实验目标

请在 LongGraph 框架中实现一个 profiling-only 的实验模块，用于拆解 Agent Memory Selection 的各阶段开销。

需要验证的核心假设是：

1. 随着长期记忆规模增长，单纯 retrieval 阶段可能不是主要瓶颈；
2. 真正值得优化的是 retrieval 之后的 candidate expansion、candidate aggregation、rerank input size 和 rerank latency；
3. 如果 memory selection 产生大量候选并显著放大 reranker 输入规模，那么后续引入 PIM-side structural pruning 才有动机；
4. 因此本实验的重点不是证明 “retrieval 很慢”，而是证明 “长期 Agent Memory 会放大 semantic selection / reranking workload”。

## 2. 推荐目录结构

请尽量按项目已有结构放置文件。若没有现成实验目录，可以创建：

```bash
experiments/motivation_memory_selection/
├── run_motivation_profile.py
├── profile_longgraph_memory_selection.py
├── synthetic_memory_generator.py
├── trace_logger.py
├── metrics.py
├── plot_results.py
├── configs/
│   └── motivation_default.yaml
└── README.md
```

如果 LongGraph 已经有 experiments、benchmarks、scripts 或 evals 目录，请优先沿用已有目录风格。

## 3. Profiling 对象

请不要只记录 end-to-end latency。需要把一次 memory selection 拆成以下阶段，并分别计时：

```text
1. query preprocessing / query encoding
2. coarse retrieval
3. metadata filtering
4. relation expansion / graph traversal
5. candidate aggregation / deduplication
6. rerank
7. context construction
8. downstream generation, optional
```

其中 downstream generation 可以默认关闭，因为本动机实验重点是 memory selection 与 rerank 压力。如果 LongGraph 原流程中生成阶段难以关闭，则需要单独记录 generation latency，避免它掩盖 selection 阶段。

## 4. 必须记录的指标

每个 query 至少记录以下字段到 JSONL 或 CSV：

```text
query_id
query_type
memory_size
num_raw_retrieved
num_after_metadata_filter
num_after_relation_expansion
num_after_dedup
num_before_rerank
num_after_rerank
query_encoding_latency_ms
coarse_retrieval_latency_ms
metadata_filter_latency_ms
relation_expansion_latency_ms
candidate_aggregation_latency_ms
rerank_latency_ms
context_construction_latency_ms
generation_latency_ms, optional
selection_total_latency_ms
end_to_end_latency_ms
selection_ratio
rerank_ratio
estimated_host_memory_traffic_bytes
pim_optimization_potential_bytes
```

其中：

```text
selection_total_latency_ms =
    query_encoding_latency_ms
  + coarse_retrieval_latency_ms
  + metadata_filter_latency_ms
  + relation_expansion_latency_ms
  + candidate_aggregation_latency_ms
  + rerank_latency_ms
  + context_construction_latency_ms
```

```text
selection_ratio = selection_total_latency_ms / end_to_end_latency_ms
rerank_ratio = rerank_latency_ms / selection_total_latency_ms
```

如果 end-to-end 不包含 generation，则需要在结果文件中显式标注。

## 5. 关键实验变量

请至少支持以下变量：

```text
memory_size: 1K, 5K, 10K, 50K, 100K, 500K, 1M
query_type: fact, temporal, relation, plan_continuity, constraint
retrieval_topk: 20, 50, 100, 200, 500
relation_expansion_depth: 0, 1, 2, 3
rerank_topk_input: all, 50, 100, 200, 500
num_queries_per_setting: default 100
```

如果完整 1M memory 运行成本过高，可以先保证 1K–100K 可运行，再将 500K/1M 设置为可选 large-scale 模式。

## 6. 查询类型设计

请构造或复用 LongGraph 中的 Agent Memory 查询，至少覆盖以下五类：

### 6.1 fact

静态事实查询，例如用户偏好、长期目标、个人设置。

系统压力：较低，通常只需要少量 memory。

### 6.2 temporal

时序状态查询，例如“最近一次计划执行到哪里”“上周到现在状态如何变化”。

系统压力：需要时间窗口过滤和连续记录扫描。

### 6.3 relation

关系型查询，例如“哪些因素导致最近训练状态下降”。

系统压力：需要从一个或多个关键 memory 扩展到相关事件、状态、原因和结果。

### 6.4 plan_continuity

计划连续性查询，例如“中断训练后如何恢复原计划”。

系统压力：需要同时召回原始计划、执行进度、中断原因、当前状态和约束。

### 6.5 constraint

风险约束查询，例如“血压偏高时是否适合加量”。

系统压力：需要检索安全约束、健康状态、历史异常和当前任务目标。

## 7. Synthetic Memory Pool 构造要求

如果项目中没有足够规模的真实 agent memory，请实现 synthetic memory generator。每条 memory 至少包含：

```json
{
  "memory_id": "...",
  "text": "...",
  "summary": "...",
  "timestamp": "...",
  "memory_type": "episodic|semantic|procedural|profile|constraint",
  "task_tags": ["..."],
  "entity_tags": ["..."],
  "importance": 0.0,
  "recency": 0.0,
  "relations": ["memory_id_1", "memory_id_2"],
  "gold_group_id": "..."
}
```

Synthetic 数据不需要完美真实，但必须能体现长期 agent memory 的结构特点：

1. 有时间链条；
2. 有任务标签；
3. 有实体/状态标签；
4. 有关系边；
5. 有共现 group；
6. 有冷/热 memory 分布。

## 8. Host-memory traffic 估算

请实现一个轻量估算函数，不要求硬件级绝对准确，但需要相对可比较。

建议估算：

```text
traffic = scanned_memory_count * header_size
        + relation_edge_count * edge_entry_size
        + num_before_rerank * candidate_payload_size
```

默认参数可以放到 config：

```yaml
header_size_bytes: 128
edge_entry_size_bytes: 8
candidate_payload_size_bytes: 512
embedding_size_bytes: 1536
```

如果当前流程会搬运 embedding，也可以单独记录：

```text
embedding_traffic = num_embedding_loaded * embedding_size_bytes
```

最终需要输出：

```text
estimated_host_memory_traffic_bytes
estimated_embedding_traffic_bytes
estimated_structural_traffic_bytes
```

## 9. 本实验要输出的图

请实现 `plot_results.py`，至少生成以下图：

### Figure 1: Latency Breakdown vs Memory Size

横轴为 memory size，纵轴为 latency，堆叠显示：

```text
coarse retrieval
metadata filtering
relation expansion
candidate aggregation
rerank
context construction
```

目的：证明 retrieval 本身可能很小，但 relation expansion 和 rerank 会随着候选膨胀变重。

### Figure 2: Candidate Count Breakdown vs Memory Size

显示：

```text
num_raw_retrieved
num_after_metadata_filter
num_after_relation_expansion
num_after_dedup
num_before_rerank
```

目的：证明长期 memory 下候选集合如何被放大。

### Figure 3: Rerank Latency vs Candidates Before Rerank

横轴为进入 reranker 的候选数，纵轴为 rerank latency。

目的：证明减少 reranker 输入规模有直接收益。

### Figure 4: Host-memory Traffic vs Memory Size

显示 estimated host-memory traffic 随 memory size 的变化。

目的：为后续 PIM-side pruning 提供动机。

### Figure 5: Query Type Breakdown

按 query_type 展示不同阶段耗时。

目的：证明 relation、plan_continuity、constraint 类查询比简单 fact 查询更依赖结构扩展和 rerank。

## 10. 需要生成的汇总表

请生成 `summary.csv` 和 `summary.md`，至少包含：

```text
memory_size
query_type
avg_selection_latency_ms
p50_selection_latency_ms
p95_selection_latency_ms
p99_selection_latency_ms
avg_rerank_latency_ms
avg_rerank_ratio
avg_num_before_rerank
avg_host_memory_traffic_mb
```

并在 `summary.md` 中自动给出几句结论模板，例如：

```text
At 100K memories, rerank accounts for XX% of memory selection latency.
Relation-heavy queries expand candidates by XX times compared with fact queries.
Host-memory traffic grows from XX MB to XX MB as memory size increases.
These results suggest that reducing candidates before reranking is more important than only optimizing coarse retrieval.
```

## 11. Baseline 设置

动机实验至少需要包含以下 baseline 或模式：

### 11.1 retrieval_only

只执行 coarse retrieval，不执行 relation expansion 和 rerank。

目的：复现“retrieval 阶段可能很小”。

### 11.2 retrieval_plus_expansion

执行 retrieval + metadata filter + relation expansion + aggregation，不执行 rerank。

目的：观察结构扩展带来的候选膨胀。

### 11.3 full_selection

执行 retrieval + expansion + rerank + context construction。

目的：展示完整 memory selection 的真实瓶颈。

### 11.4 oracle_pruned_rerank, optional

用 gold memory group 或 synthetic ground truth 模拟理想剪枝，把进入 rerank 的候选数限制到较小范围，例如 50/100。

目的：估计如果未来 PIM 能提前压缩候选，理论上能减少多少 rerank latency。

注意：oracle_pruned_rerank 只是动机实验中的上界分析，不代表已经实现 PIM。

## 12. 不要做的事情

本任务只针对动机实验，请不要实现以下内容：

```text
不要实现完整 PIM simulator
不要实现真实 PIM kernel
不要实现最终 hypergraph layout optimizer
不要替换 LongGraph 的 reranker
不要声称 PIM 可以直接输出最终最佳候选
不要把 PIM 写成复杂语义推理模块
```

本实验的正确结论应该是：

> 如果长期 Agent Memory 导致 relation expansion 和 rerank input size 变大，那么后续工作可以引入 PIM-side structural pruning 来提前减少候选规模，从而降低 expensive reranker 的压力。

## 13. 运行命令要求

请提供统一运行入口，例如：

```bash
python experiments/motivation_memory_selection/run_motivation_profile.py \
  --config experiments/motivation_memory_selection/configs/motivation_default.yaml
```

并支持快速测试模式：

```bash
python experiments/motivation_memory_selection/run_motivation_profile.py \
  --config experiments/motivation_memory_selection/configs/motivation_default.yaml \
  --quick
```

quick 模式可以只运行：

```text
memory_size: 1K, 5K
num_queries_per_setting: 5
query_type: fact, relation
```

## 14. 输出文件要求

实验结束后请输出到：

```bash
outputs/motivation_memory_selection/
├── raw_profile.jsonl
├── summary.csv
├── summary.md
├── figures/
│   ├── latency_breakdown_vs_memory_size.png
│   ├── candidate_count_breakdown.png
│   ├── rerank_latency_vs_candidates.png
│   ├── host_memory_traffic.png
│   └── query_type_breakdown.png
└── logs/
```

## 15. 判断实验是否成功

实验成功的最低标准：

1. 能在 quick 模式下完整跑通；
2. 能输出 raw_profile.jsonl；
3. 能输出 summary.csv 和 summary.md；
4. 能生成至少 3 张核心图；
5. latency breakdown 中能区分 retrieval、expansion、rerank；
6. 能报告 num_before_rerank 和 rerank_latency 的关系。

理想结果是：

1. retrieval_only 的耗时相对较小；
2. relation / plan_continuity / constraint 查询产生更高的 candidate expansion；
3. rerank latency 与 num_before_rerank 正相关；
4. full_selection 中 rerank 或 expansion+rerank 占据主要开销；
5. oracle_pruned_rerank 显示：如果能在 rerank 前减少候选，整体 selection latency 有明显下降空间。

## 16. 结果解释边界

如果实验发现 retrieval 阶段占比很小，不要解释为研究动机失败。正确解释是：

> 本工作不应定位为单纯加速 coarse retrieval，而应定位为在 reranker 之前进行结构感知候选压缩，降低昂贵 semantic selection 的输入规模。

如果实验发现 rerank 占比很大，这是支持后续设计的证据：

> PIM 的目标不是替代 reranker，而是减少进入 reranker 的候选数量。

如果实验发现 selection 在包含 generation 的端到端流程中占比不高，则需要额外报告：

```text
selection-only latency
selection ratio without generation
p95/p99 latency
multi-agent concurrency setting
```

并解释：

> 在长期 agent 服务中，selection 是高频前置阶段，会影响 tail latency 和并发吞吐；即使单次端到端比例不总是最高，它仍然是可被系统优化的关键路径。
