# Motivation 实验总结文档

本文档用于对 `agent-memory-profiling/versions/20260703_method1_motivation` 这一整套 motivation 实验进行一次面向审查的汇总，目标是让后续 GPT 或人工 reviewer 可以直接基于这份文档判断：

1. 实验设置是否合理。
2. 实验安排是否完整。
3. 结果是否形成闭环。
4. 当前结果适合支撑什么结论，不适合支撑什么结论。

## 1. 实验目标

本次 motivation 实验不是为了直接证明最终方法设计有效，而是为了完成论文动机部分所需的四类经验事实验证：

- `M1`：下游 memory selection / rerank 成本会随着候选 frontier 扩大而显著上升。
- `M2`：任务质量与 frontier budget 之间存在值得研究的 trade-off，尤其在 harder queries 上更明显。
- `M3`：命中的 memory 在物理布局上存在 scattering，顺序布局利用率低，存在布局优化动机。
- `O1`：真实 trace 中存在 phase-conditioned locality，memory access 并非随机。

本轮工作的优先级是把 motivation 实验本身做成闭环，不要求当前就足以“支持最终设计”。

## 2. 实验包范围

当前实验包目录：

- `agent-memory-profiling/versions/20260703_method1_motivation`

主要输入来源：

- 历史 V3 formal real-GPU cross-encoder runs：
  `versions/20260616_phase_aware_real_agent_v3_formal/`
- 历史 V4 formal real-trace adaptive runs：
  `versions/20260621_real_trace_adaptive_phase_v4_formal/`

最终已通过输出校验：

- `python agent-memory-profiling/versions/20260703_method1_motivation/scripts/validate_motivation_outputs.py --root agent-memory-profiling/versions/20260703_method1_motivation`

## 3. 实验设置

### 3.1 环境与可追溯性

- 运行环境：`agent-memory-profiling` Conda 环境。
- GPU：2026-07-03 在本机 NVIDIA GeForce RTX 3090 上重新确认。
- 工作区限制：当前 workspace 的 `.git/` 目录为空，无法恢复有效 git commit；因此实验包中将 commit 记录为 `unavailable_empty_dot_git_directory`。
- 包级 manifest：
  `manifests/run_manifest.json`
- 历史来源映射：
  `manifests/source_runs.json`

### 3.2 数据集与任务

历史 M1/M2/M3 来源数据集：

- `locomo`
- `longmemeval`
- `msc`

Fresh V4 official-QA rerun 数据集：

- `locomo`
- `longmemeval_s`

Fresh V4 QA 任务类型：

- `official_qa`
- real-trace runtime manifest 驱动
- answer generation 打开，即 `selection_only=false`

### 3.3 Fresh V4 QA grid 的统一实验配置

本轮最终统一到同一条 answer path 的 formal/smoke QA grid 配置具有以下共同点：

- memory mode：`shared_once`
- `selection_only=false`
- `final_top_k=5`
- budget grid：`{8,16,32,64}`
- 每个 query 的平均 retrieval invocation 数约为 `1.0`
- runtime allowed fields 含：
  `text`, `time`, `session_id`, `speaker`, `tenant_id`

### 3.4 本轮修复后的 answer / scorer 路径

为避免 official QA 指标退化为全零，本轮对实验路径进行了审计可解释的修复，但没有修改任务定义本身：

- answer prompt 增加了 metadata awareness：
  使用 memory 的 `event_time`、`session_id`、`speaker`
- answer prompt 增加了 query 自身时间：
  `Current query timestamp`
- answer prompt 明确要求：
  只输出单行最终答案，不输出占位符，不继续对话，不自我修正
- generation 后增加 deterministic short-answer contraction：
  从 raw completion 中提取最终短答案
- scorer 增加 reference normalization：
  显式支持 `also acceptable` 类型参考答案
- scorer 增加 insufficiency label 对齐：
  将长句式 “信息不足” 参考答案统一识别为 `insufficient information`

这意味着当前 QA 结果是“实验管线修复后”的结果，而不是“完全不加后处理的裸生成结果”。

## 4. 实验安排

### 4.1 M1 / M2 / M3 / O1 的来源安排

#### M1

- 直接复用历史 V3 formal real-GPU cross-encoder run。
- 输出文件：
  `aggregate/M1_stage_latency.csv`

#### M2 历史代理版本

- 直接复用历史 V3 synthetic-query / proxy quality 结果。
- 输出文件：
  `aggregate/M2_pareto.csv`

#### M3

- 基于历史命中集合做 tile scattering 分析。
- 输出文件：
  `aggregate/M3_tile_scattering.csv`

#### O1

- 复用历史 V4 real-trace adaptive `selection_only` run 做 locality/null-model 对比。
- 输出文件：
  `aggregate/O1_phase_locality.csv`
  `aggregate/O1_phase_matrix.csv`

### 4.2 Fresh V4 official-QA 补充实验安排

目的：

- 用 benchmark-native QA 任务补 M2 的真实任务闭环。
- 不再只依赖历史 synthetic proxy。

执行顺序：

1. 先做 smoke rerun，验证代码路径与指标是否通。
2. 对 answer path 和 scorer 做修复。
3. 重新做 smoke，检查非零 QA 指标是否出现。
4. 做 formal 端点验证，例如 `longmemeval_s` 的 `8/64`。
5. 最终将 `locomo` 与 `longmemeval_s` 的 formal `{8,16,32,64}` 全部重新跑齐。
6. 统一刷新 aggregate、strata、文档，并通过 package validator。

最终状态：

- `locomo` formal `{8,16,32,64}`：已全部按统一 answer path 重新跑完。
- `longmemeval_s` formal `{8,16,32,64}`：已全部按统一 answer path 重新跑完。
- smoke / formal 输出均保留在：
  `runs/`

## 5. 核心结果

### 5.1 M1：memory-scale profiling

来源：

- `aggregate/M1_stage_latency.csv`

代表性观察：

- `msc`，memory size `395934`，frontier budget `100`：
  `t_rerank_mean_ms = 654.7`
  `t_total_mean_ms = 3755.7`
- `msc`，budget 从 `50 -> 100`：
  `t_rerank_mean_ms` 从 `450.1 -> 654.7` ms
  `t_total_mean_ms` 从 `3130.9 -> 3755.7` ms

结论：

- rerank-heavy selection cost 确实存在。
- frontier 扩大时，下游代价增加明显。

### 5.2 M2 历史代理版本

来源：

- `aggregate/M2_pareto.csv`

代表性观察：

- `locomo multi_evidence`：
  downstream selection cost 从 `422.6 -> 721.0` ms
  task proxy group hit 从 `0.4000 -> 0.4833`
- `longmemeval long_range`：
  downstream selection cost 从 `1293.1 -> 2536.3` ms
  high-budget stability 从 `0.6063 -> 1.0000`

结论：

- 历史 proxy 版本的 M2 是成立的：
  更大的 candidate frontier 带来更高成本；
  harder strata 具有更强的 budget-sensitive 特征。

限制：

- 这部分不是 benchmark-native QA 指标，而是 synthetic/proxy quality。

### 5.3 M2 Fresh V4 official-QA：formal 汇总

来源：

- `aggregate/v4_official_qa_budget_grid_summary.csv`
- `aggregate/v4_official_qa_budget_grid_summary.md`

#### `locomo` formal

| budget | avg selection ms | evidence recall | EM | token F1 | insuff rate |
|---|---:|---:|---:|---:|---:|
| 8  | 201.3 | 0.3490 | 0.0313 | 0.2220 | 0.0313 |
| 16 | 328.5 | 0.3333 | 0.0469 | 0.2328 | 0.0313 |
| 32 | 543.7 | 0.3438 | 0.0156 | 0.1900 | 0.0313 |
| 64 | 990.9 | 0.3438 | 0.0313 | 0.2091 | 0.0156 |

观察：

- selection-path latency 随 budget 明显上升。
- evidence recall 基本持平，未随 budget 明显改善。
- official QA `EM` 虽然不再是全零，但仍然很低，且非单调。

解释：

- `locomo` 当前更像是 retrieval-limited 问题。
- answer path 修复消除了“指标全零”的退化，但没有把它变成干净的 budget-quality 曲线。

#### `longmemeval_s` formal

| budget | avg selection ms | evidence recall | EM | token F1 | insuff rate |
|---|---:|---:|---:|---:|---:|
| 8  | 87.3  | 0.3906 | 0.1563 | 0.2258 | 0.5000 |
| 16 | 167.6 | 0.4219 | 0.1406 | 0.2140 | 0.5000 |
| 32 | 272.5 | 0.4219 | 0.1406 | 0.2167 | 0.5000 |
| 64 | 466.7 | 0.4271 | 0.1250 | 0.1948 | 0.5000 |

观察：

- selection-path latency 随 budget 明显上升。
- evidence recall 有小幅提升。
- `EM` 和 `token F1` 为非零，说明 end-to-end QA 闭环已打通。
- 但质量曲线依旧不单调，且没有随 budget 稳定提升。

解释：

- `longmemeval_s` 比 `locomo` 更适合观察 answer path 修复后的收益，因为它的 evidence recall 更高。
- 但 formal 规模下仍然没有出现“budget 越大，任务质量越高”的稳定规律。

### 5.4 M2 Fresh V4 official-QA：strata 分析

来源：

- `aggregate/v4_official_qa_query_strata.csv`
- `aggregate/v4_official_qa_strata_population.csv`
- `aggregate/v4_official_qa_strata_summary.csv`

strata 构造：

- `single_evidence`
- `multi_evidence`
- `long_range`

#### `locomo`

代表性观察：

- `single_evidence` 在 budget 64 的 `mean_evidence_recall = 0.5385`
- `multi_evidence` 在 budget 64 的 `mean_evidence_recall = 0.2105`
- `long_range` 在 budget 64 的 `mean_evidence_recall = 0.2308`
- `multi_evidence` 的 selection cost 从 `362.6 -> 1940.3` ms
- `long_range` 的 selection cost 从 `397.7 -> 1874.8` ms

结论：

- harder strata 明显更难，且 cost 增长更明显。
- retrieval-side 的 budget sensitivity 是成立的。

#### `longmemeval_s`

代表性观察：

- `long_range` 在所有 budget 下 `mean_selected_memory_count ≈ 1.15`
- `multi_evidence` 在所有 budget 下 `mean_selected_memory_count ≈ 2.26`
- `single_evidence` 在所有 budget 下 `mean_selected_memory_count ≈ 2.73`
- 即使 budget=64，很多 query 也没填满 `final_top_k=5`

结论：

- 该数据集的 harder subsets 具有明显 frontier sparsity 问题。
- 这支持“检索难度和 frontier budget 的关系值得研究”，但不支持“budget 增大一定改善最终任务分数”。

### 5.5 M3：candidate scattering

来源：

- `aggregate/M3_tile_scattering.csv`

代表性观察：

- chronological layout 下，tile utilization 远低于 ideal layout。
- 例如 `locomo long_range`：
  - capacity `16` 时 chronological `mean_tile_utilization = 0.0652`
  - ideal `mean_tile_utilization = 0.5000`

结论：

- 命中 memory 的物理分布是散的，不适合简单顺序布局。
- 这为后续 layout / placement 优化提供动机。

### 5.6 O1：phase-conditioned locality

来源：

- `aggregate/O1_phase_locality.csv`

代表性观察：

- `real`，window=1：
  `adjacent_overlap = 0.7094`
- `memory_shuffle`，window=1：
  `adjacent_overlap = 0.0843`
- `phase_shuffle` 不改变 access set，因此 overlap 基本不变。

结论：

- 真实 trace 中的 locality 明显强于 memory-shuffled null model。
- access 不是随机的，phase-conditioned locality 动机成立。

## 6. 本次实验闭环完成到了什么程度

### 6.1 已完成的闭环

- `M1`：完成。
- `M2` 历史 proxy 版本：完成。
- `M2` fresh official-QA end-to-end 版本：完成。
- `M3`：完成。
- `O1`：完成。
- 整个 motivation package：已校验通过。

### 6.2 “完成” 的准确含义

这里的“完成”指：

- 所有 motivation 子问题都已有对应实验产物。
- fresh official-QA 已经不是退化的全零回答指标。
- formal QA grid 已统一在一条 answer path 上重跑完成。
- 结果、限制和边界已经在包内显式记录。

这里的“完成”不等于：

- 已经得到足以强力支撑方法设计的 task-quality 曲线。
- 已经证明更大 budget 会稳定提高 final task score。

## 7. 当前结果最适合支撑的结论

适合支撑：

- 下游 selection / rerank 成本对 frontier budget 高敏感。
- harder queries 在 retrieval-side 更难，且更容易出现 sparse frontier。
- memory 命中在物理布局上存在 scattering。
- real trace 中存在显著 locality，而不是随机访问。
- official QA 闭环已经打通，且不是全零退化。

不适合支撑：

- “更大 budget 会稳定提升官方 QA 指标”
- “当前动机实验已经直接证明你的方法设计有效”
- “当前结果足以单独证明 hypergraph / PIM / placement 方案收益”

## 8. 当前实验的合理性检查点

建议 GPT 或 reviewer 重点检查以下问题：

### 8.1 实验设置合理性

- 历史 M1/M2/M3 与 fresh M2 QA supplement 混合使用，是否在论文中说明得足够清楚？
- answer path 中的 deterministic contraction 和 scorer normalization 是否属于合理的实验管线修复，而不是引入了不可接受的 benchmark 偏移？
- `shared_once`、`final_top_k=5`、budget `{8,16,32,64}` 是否与论文目标和 guide 保持一致？

### 8.2 实验完整性

- 是否每个 motivation claim 都至少有一类直接证据？
- 是否所有 fresh formal QA runs 都在同一条 answer path 上重新跑齐？
- 是否结果与限制都显式记录，没有只汇报“好看”的部分？

### 8.3 结果解释合理性

- 在 `locomo` 上应否将主要瓶颈解释为 retrieval-limited，而不是 answer-formatting-limited？
- 在 `longmemeval_s` 上是否应承认 official QA 已非零，但 budget-quality 关系仍弱且非单调？
- strata 分析是否足以说明 harder subsets 的 retrieval-side 困难度，而不夸大为 task-quality law？

## 9. 当前实验的完整性结论

如果问题是“这次 motivation 实验是否已经形成闭环”：

- 答案是：`是，已经形成实验闭环。`

如果问题是“这次 motivation 实验是否已经足以强支撑最终设计或方法收益”：

- 答案是：`否，当前还不够。`

更准确的结论是：

- 这套实验已经完成了 motivation 层面的观测闭环和 QA 闭环。
- 但它更适合支撑“为什么这个问题值得做”，而不是“为什么你的具体方案一定有效”。

## 10. 建议 GPT 后续审查时重点回答的问题

你可以让 GPT 重点回答下面几个问题：

1. 这套 motivation 实验是否覆盖了论文动机所需的关键现象，是否存在缺失环节？
2. 历史实验与 fresh official-QA supplement 的组合是否方法学上合理？
3. answer-path 修复和 scorer 修复是否足够透明、足够克制？
4. 当前结果最稳妥的论文表述应该是什么？
5. 如果要让这套 motivation 进一步支撑最终设计，下一步最值得补的实验是什么？

## 11. 相关文件索引

核心说明：

- `README.md`
- `RESULTS.md`
- `LIMITATIONS.md`

核心聚合：

- `aggregate/M1_stage_latency.csv`
- `aggregate/M2_pareto.csv`
- `aggregate/M3_tile_scattering.csv`
- `aggregate/O1_phase_locality.csv`
- `aggregate/v4_official_qa_budget_grid_summary.csv`
- `aggregate/v4_official_qa_strata_summary.csv`

fresh QA runs：

- `runs/v4_formal_locomo_budget8/`
- `runs/v4_formal_locomo_budget16/`
- `runs/v4_formal_locomo_budget32/`
- `runs/v4_formal_locomo_budget64/`
- `runs/v4_formal_longmemeval_s_budget8/`
- `runs/v4_formal_longmemeval_s_budget16/`
- `runs/v4_formal_longmemeval_s_budget32/`
- `runs/v4_formal_longmemeval_s_budget64/`

脚本：

- `scripts/run_motivation_suite.py`
- `scripts/run_v4_official_qa_budget_grid.py`
- `scripts/backfill_v4_official_qa_metrics.py`
- `scripts/summarize_v4_official_qa_budget_grid.py`
- `scripts/analyze_v4_official_qa_strata.py`
- `scripts/validate_motivation_outputs.py`
