# MemoryCD Candidate Frontier V5.1

日期：2026-06-23

本目录对应 `codex_memorycd_v5_1_candidate_frontier_guide.md` 的正式 V5.1 实验版本，目标是在真实 MemoryCD 数据、真实本地模型和严格 runtime/evaluator 分离条件下，评估 candidate frontier 的质量-代价关系，并为后续超图/PIM 加速 `select + rerank` 路径提供依据。

## 1. 实验目标

这版实验要回答的不是“MemoryCD 能不能做任务”，而是更具体的两个问题：

1. `R` 增大时，候选前沿 `candidate frontier` 的覆盖质量是否提升。
2. `B` 增大时，昂贵 rerank 预算是否真的换来下游任务质量提升。

这里：

- `R` 表示 dense retrieval 先召回多少条 memory 进入候选前沿。
- `B` 表示 cross-encoder reranker 只重排其中多少条候选。
- `K=8` 表示最终送入 Qwen prompt 的 memory 条数固定为 8。

因此，方法名 `dense_r32_b8_k8` 的含义是：

- BGE dense retrieval 先取 top-32
- MiniLM rerank 只处理其中 8 条
- 最终选 8 条 memory 进入 Qwen

这个设计直接对应论文里的系统优化目标：如果后续用超图或 PIM 去加速 `R -> B -> K` 这段 preselect/rerank 路径，那么收益应该能同时体现在：

- selection latency 降低
- 在部分任务上维持接近高预算配置的任务质量

## 2. 数据与协议

### 2.1 数据路径

- 数据集根目录：`data/public_datasets/memorycd/`
- 元数据目录：`data/public_datasets/memorycd/meta/`
- 用户目录：`data/public_datasets/memorycd/users/`

### 2.2 三类 protocol

- `official_single`
- `official_cross`
- `causal_cross`

### 2.3 本轮正式评估任务

- `rating_prediction`
- `item_ranking`

本轮没有在正式 subset 上运行：

- `review_summarization`
- `review_generation`

### 2.4 runtime / evaluator 分离

按 guide 要求，运行时只读取：

- runtime manifest
- memory bank
- metadata

运行时不读取 evaluator labels。  
评估阶段再离线读取 `evaluation_labels.jsonl`，计算 RMSE、NDCG、Recall、bootstrap CI 和 Pareto 结果。

## 3. 模型与实现约束

### 3.1 本地模型

- Dense embedding：`bge-base-en-v1.5`
- Reranker：`cross-encoder-ms-marco-MiniLM-L6-v2`
- Task LLM：本地 `Qwen`

### 3.2 检索/选择约束

- 使用真实 MemoryCD memory bank
- 使用真实 BGE dense retrieval
- 使用真实 MiniLM rerank
- 不使用 phase/controller path
- 不做 relation expansion
- 不引入 Qwen retrieval controller
- 不使用 synthetic data
- 不混入 LoCoMo / MSC / LongMemEval 路径

### 3.3 检索正确性

前序 selection sweep 已完成：

- FAISS flat 与 brute-force Top-R score-boundary tolerance `1e-6` 校验

对应报告：

- `reports/candidate_frontier_sweep_report.md`

## 4. 当前已完成的实验

### 4.1 Selection-only cached sweep

已完成每个 protocol 前 `1000` 个 eligible `rating/ranking` event 的 selection-only cached sweep：

- 每个 protocol `11000` 条 trace
- 总计 `33000` 条 trace

这一部分只证明 selection latency 与预算关系，不等同于完整 task quality。

### 4.2 正式 task-quality subset curve

本轮正式完成的是固定 subset Qwen task-quality curve。

实验矩阵为：

- `3 protocols`
- `2 tasks`
- `4 settings` / protocol-task
- `11 methods`

每个 `protocol × task × setting` 最多取 `200` 个真实 eligible event。  
前 `20` 个 event 只作为 warmup，不进入延迟聚合与最终统计，因此：

- 常规 setting 的有效样本数为 `200`
- `Beauty_and_Personal_Care` 相关 setting 为 `180`

总 trace 数：

- 每个 `protocol × task`：`4 × 200 × 11 = 8800`
- 6 个正式 trace 文件总计：`52800`

正式 trace 文件：

- `runs/task_curve_subset/official_single_rating_prediction_task_curve_subset.jsonl`
- `runs/task_curve_subset/official_cross_rating_prediction_task_curve_subset.jsonl`
- `runs/task_curve_subset/causal_cross_rating_prediction_task_curve_subset.jsonl`
- `runs/task_curve_subset/official_single_item_ranking_task_curve_subset.jsonl`
- `runs/task_curve_subset/official_cross_item_ranking_task_curve_subset.jsonl`
- `runs/task_curve_subset/causal_cross_item_ranking_task_curve_subset.jsonl`

### 4.3 旧 smoke 文件处理

早期 smoke 产物：

- `runs/task_curve_subset/_legacy_smoke/official_single_task_curve_subset.jsonl`

该文件已移出正式目录，summary 脚本也只汇总带正式 task 后缀的 trace 文件，因此不会再污染当前版本结果。

## 5. 方法矩阵

本轮固定比较 11 个方法：

- `recency_k8`
- `dense_k8`
- `dense_r32_b8_k8`
- `dense_r32_b16_k8`
- `dense_r32_b32_k8`
- `dense_r64_b16_k8`
- `dense_r64_b32_k8`
- `dense_r64_b64_k8`
- `dense_r128_b16_k8`
- `dense_r128_b32_k8`
- `dense_r128_b64_k8`

解释：

- `recency_k8`：只按时间最近选 8 条
- `dense_k8`：dense retrieval 后直接取 top-8，不走 rerank
- `dense_rR_bB_k8`：先取 top-R frontier，再对其中 B 条做 rerank

## 6. 指标与统计口径

### 6.1 任务质量

- `rating_prediction`：`rmse`
- `item_ranking`：主指标 `ndcg@5`

内部 evaluator 同时计算：

- `recall@1`
- `recall@3`
- `recall@5`

但当前主汇总表 `task_quality_curve_summary.csv` 记录的是主指标 `ndcg@5`。

### 6.2 时延

记录：

- `p50_selection_ms`
- `p95_selection_ms`
- `p99_selection_ms`
- `p50_task_ms`
- `p95_task_ms`
- `p99_task_ms`

其中：

- `selection` 指 memory selection / rerank 路径
- `task` 指 `selection + Qwen generation` 的总任务路径

### 6.3 Bootstrap CI 与 Pareto

当前 summary 已使用 bootstrap CI。

Pareto 规则：

- `rating_prediction`：越小越好，用 RMSE 阈值 `reference + delta`
- `item_ranking`：越大越好，用 NDCG 阈值 `reference - delta`

当前 `delta` 定义：

- `rating_prediction`：`max(0.01, 1% * reference_metric)`
- `item_ranking`：`0.01`

## 7. 当前正式结果

正式汇总文件：

- `reports/task_quality_curve_summary.csv`
- `reports/pareto_budget_report.json`

当前 summary 共 `264` 行，对应：

- `3 protocols × 2 tasks × 4 settings × 11 methods`

### 7.1 rating_prediction

结论非常稳定：`R/B` 预算增大几乎不带来可见质量收益。

按 protocol 聚合后的代表结果：

- `official_single`
  - 最优：`recency_k8`，RMSE `1.6197`
  - `dense_k8`：`1.6209`
  - 大多数 rerank 配置也约为 `1.6209`
- `official_cross`
  - 所有方法几乎完全打平在 `1.6476`
- `causal_cross`
  - 所有方法几乎完全打平在 `1.6325`

这说明：

- 对 rating 任务，candidate frontier 改善对下游 Qwen 预测帮助非常有限
- 增加 rerank 预算带来的额外 latency 很难被质量收益证明是值得的

对应时延代价：

- `recency_k8`：平均 `p50 selection ~= 0 ms`
- `dense_k8`：平均 `p50 selection ~= 12-13 ms`
- `dense_r128_b64_k8`：平均 `p50 selection ~= 76-80 ms`

也就是说，rating 任务上预算拉大时：

- 成本明显增加
- 质量基本不变

这是论文里“预算压缩合理性”的直接证据。

### 7.2 item_ranking

`item_ranking` 与 `rating_prediction` 完全不同，对 candidate frontier 显著更敏感。

按 protocol 聚合后的代表结果：

- `official_single`
  - `recency_k8 = 0.9598`
  - `dense_k8 = 0.9656`
  - 最优 `dense_r128_b64_k8 = 0.9732`
  - 相对 recency 提升 `+0.0134`
- `official_cross`
  - `recency_k8 = 0.9689`
  - `dense_k8 = 0.9754`
  - 最优 `dense_r128_b16_k8 = 0.9854`
  - 相对 recency 提升 `+0.0165`
- `causal_cross`
  - `recency_k8 = 0.9778`
  - `dense_k8 = 0.9845`
  - 最优 `dense_r32_b8_k8 = 0.9866`
  - 相对 recency 提升 `+0.0089`

这说明：

- ranking 任务确实依赖更好的候选前沿
- dense retrieval 相比 recency 已经有收益
- 在部分 setting 中，再加适度 rerank 可以继续提升

但同时也说明：

- “更大预算一定更好”并不成立
- 一些低预算配置已经接近或达到高预算效果

### 7.3 setting 级例子

几个最有代表性的 setting：

- `official_cross / cross_to_beauty_and_personal_care`
  - `recency_k8 = 0.9410`
  - `dense_r32_b8_k8 = 0.9927`
  - 提升 `+0.0517`
  - 说明 candidate frontier 质量对 ranking 结果有实质影响

- `causal_cross / cross_to_books`
  - 高预算最优 `dense_r128_b64_k8 = 1.0000`
  - `recency_k8` 也达到 `1.0000`
  - 说明有些 setting 很早就已经饱和，不需要昂贵 rerank

- `official_single / single_beauty_and_personal_care`
  - 最优 `dense_r128_b16_k8 = 0.9503`
  - 相对 `recency_k8` 提升 `+0.0283`
  - 相对 `dense_k8` 提升 `+0.0155`
  - 说明单域 setting 中也存在明显的 frontier 质量效应

## 8. Pareto-min-budget 结果

当前 `24` 个 `protocol × task × setting` group 中：

- `rating_prediction`：`12 / 12` 没有选出 Pareto-min-budget
- `item_ranking`：`4 / 12` 选出 Pareto-min-budget

这 4 个 group 分别是：

- `causal_cross / item_ranking / cross_to_books`
  - reference：`dense_r128_b64_k8`
  - pareto：`recency_k8`
- `official_cross / item_ranking / cross_to_beauty_and_personal_care`
  - reference：`dense_r32_b8_k8`
  - pareto：`dense_r32_b8_k8`
- `official_cross / item_ranking / cross_to_books`
  - reference：`dense_r128_b16_k8`
  - pareto：`recency_k8`
- `official_single / item_ranking / single_books`
  - reference：`dense_r128_b64_k8`
  - pareto：`recency_k8`

解读：

- Pareto 现象已经出现，说明确实存在预算压缩空间
- 但它不是所有 setting 都成立
- 预算压缩是否可行，是 task-aware、setting-aware 的

这正好支持后续论文叙述：

- 不应盲目增大 `B`
- 应寻找“足够好但更便宜”的 frontier/rerank 配置

## 9. 对论文的支撑

### 9.1 对系统设计的支撑

这版实验支持以下系统结论：

1. `R` 和 `B` 不是任意超参，而是 candidate frontier 系统里的关键成本控制点。
2. selection 路径时延会随 `B` 上升显著增加。
3. 下游质量对 `R/B` 的敏感度是任务相关的。

因此，后续用超图或 PIM 加速 `R -> B` 之间的 preselect/rerank 路径是有明确实验依据的，不是抽象的工程优化。

### 9.2 对论文核心论点的支撑

这版结果支持如下主张：

- 对 `rating_prediction`，大预算 rerank 没有明显收益，因此可以优先压缩 selection/rerank 成本。
- 对 `item_ranking`，candidate frontier 质量确实重要，但收益通常在最大预算前就开始饱和。

因此论文可以合理主张：

- 目标不是“把 reranker 一路堆大”
- 而是在保证 frontier 质量的前提下，找到更优的预算点

### 9.3 与你前序 v4 结论的关系

如果你在 v4 中已经验证：

- `select`
- `rerank`

在整个 agent 执行流里占有显著时延比例，那么这版 V5.1 实验给了更强的任务级支撑：

- selection-only sweep 证明这两段路径本身确实随预算增长而变慢
- task-quality subset 证明变慢并不总能换来质量增益

所以你后续强调“通过更优 preselect 机制同时降低 `select + rerank` 总时延”是有数据基础的。

## 10. 当前尚未完成的部分

当前完成的是正式 `subset curve`，不是最终 full formal 结果。

尚未运行：

- `task_final_full`
- 以 Pareto 结果筛出的最终 full task runs
- 全量终表版本的 recall 汇总输出

因此当前结论应表述为：

- 已有较强 subset 证据
- 但最终论文主表仍应以 full formal run 为准

## 11. 关键产物索引

### 11.1 结果文件

- `reports/task_quality_curve_summary.csv`
- `reports/pareto_budget_report.json`
- `reports/task_quality_curve_report.md`
- `reports/candidate_frontier_sweep_summary.csv`
- `reports/candidate_frontier_sweep_report.md`

### 11.2 脚本

- `experiments/memorycd/run_task_quality_curve.py`
- `experiments/memorycd/summarize_task_quality_curve.py`
- `experiments/memorycd/run_task_final_full.py`

### 11.3 运行痕迹

- `runs/task_curve_subset/*_task_curve_subset.jsonl`
- `runs/task_curve_subset/*_task_curve_run_manifest.json`
- `runs/task_curve_subset/_legacy_smoke/official_single_task_curve_subset.jsonl`

## 12. 当前结论摘要

一句话概括当前版本：

- `rating_prediction` 上，大预算 rerank 基本不值。
- `item_ranking` 上，更好的 candidate frontier 确实有效，但常常无需拉到最大预算。

这正是后续论文中“通过更高效的 frontier/preselect 机制，加速 `select + rerank` 而不过度牺牲质量”的实验依据。
