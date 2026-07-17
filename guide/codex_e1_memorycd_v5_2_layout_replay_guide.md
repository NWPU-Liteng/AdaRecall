# Codex 执行指南：E1 — 基于 MemoryCD V5.1 真实选择轨迹的逻辑布局回放

## 0. 这轮只做什么

本轮只验证一个问题：**历史上共同进入同一参考选择结果的 memory，是否会在未来真实事件中再次共同出现；若会，超图布局能否将它们集中到更少的固定容量逻辑 Tile 中。**

V5.1 已定义真实选择路径：

```text
query → dense top-R → rerank top-B → final top-K (K=8)
```

对每个事件，取真实选择路径输出的 final top-8 memory IDs 作为“参考选择集合”。使用早期事件学习不同布局，再用后续事件检查：该事件的参考 top-8 分散在多少 Tile 中。

本轮是**真实数据驱动的离线逻辑布局回放**，不是 PIM 仿真。

### 本轮不做

- 不做 PIM simulator、DRAM Row/Bank 映射、row-buffer、带宽、能耗或端到端 PIM latency。
- 不实现在线超图扩展、渐进剪枝、候选替换或完整 AdaRecall。
- 不重跑 Qwen task quality。
- 不读取 `evaluation_labels.jsonl`、NDCG、RMSE、Recall 或 Pareto 结果来构图、选样本、选 selector、选 Tile 容量或调参。
- 不覆盖 V5.1 的任何正式结果。

### 本轮结果能说明什么

若超图布局在未来事件上让参考 top-8 更集中于较少 Tile，则可说明：**真实选择轨迹中存在可利用的逻辑共置潜力**。

### 本轮结果不能说明什么

不能说明 PIM 已经更快，不能说明 DRAM Row/Bank 指标更优，也不能说明最终任务质量已提升。

---

## 1. 必须继承的 V5.1 基础

以下内容已经在 `README_v5.1.md` 中确认，必须复用或核实，不得替换：

| 项目 | 设置 |
|---|---|
| 数据集 | MemoryCD |
| Protocol | `official_single`、`official_cross`、`causal_cross` |
| Task | `rating_prediction`、`item_ranking` |
| Dense embedding | `bge-base-en-v1.5` |
| Reranker | `cross-encoder-ms-marco-MiniLM-L6-v2` |
| Task model | 本地 Qwen；E1 通常不调用 |
| Final K | 8 |
| 数据隔离 | runtime / evaluator 分离 |
| V5.1 subset | 每个 `protocol × task × setting` 最多 200 个真实 eligible event；前 20 个为 warmup，不进入正式统计 |

所有新输出必须写入独立版本目录，例如：

```text
runs/e1_memorycd_v5_1_layout_replay_v1/
reports/e1_memorycd_v5_1_layout_replay_v1/
configs_snapshot/e1_memorycd_v5_1_layout_replay_v1/
```

若项目实际目录结构不同，先发现 V5.1 根目录和现有命名规范，再创建等价目录；禁止假设绝对路径。

---

## 2. 先做数据审计：未通过就停止

E1 依赖“同一 memory bank 上有多个按时间/因果顺序出现的真实选择事件”。若 V5.1 没有这种数据条件，不能通过跨 bank 拼接或伪造序列来继续。

### 2.1 交付 `e1_input_audit.md` 和 `e1_input_audit.json`

Codex 必须先审计并输出真实结果：

1. V5.1 正式 manifest、runtime memory bank、task trace、run manifest、配置快照的实际路径。
2. 每条事件是否有：
   - stable event ID；
   - stable runtime bank ID；
   - stable memory item ID；
   - 事件顺序或时间戳；
   - final top-K memory IDs，或能否通过真实 selection-only 路径重新 materialize；
   - event 发生前的可见 memory universe，或能否证明 memory bank 为固定静态 bank。
3. 每个 bank 的：
   - eligible event 数、warmup 后 event 数；
   - 可用于因果切分的事件数；
   - memory ID 重用率；
   - final top-K 集合的 Jaccard 分布；
   - 每个 memory 被 final-selected 的次数分布。
4. 是否存在真实 memory write/insertion order、memory timestamp 或 per-event snapshot。
5. 若同时有 write index 与 timestamp，检查两者是否几乎相同；若相同，后续只保留一个 `write_time_order` 基线。

### 2.2 Go / No-Go 规则

只有同时满足以下条件，才进入主实验：

- 同一 bank 存在多个按因果顺序排列的 eligible event；
- final top-K 能可靠映射回真实 memory ID；
- memory ID 能和该 bank 中的 item 一一对应；
- 训练期与测试期可以按时间分开；
- 不需要 evaluator label 才能得到 final top-K。

以下任一情况出现时，必须生成 blocker report 并停止：

- 每个 bank 只有一个可用事件；
- final top-K memory ID 无法恢复；
- 没有可证明的事件顺序；
- memory ID 跨事件不稳定；
- 无法隔离 runtime bank / tenant；
- 必须读取 evaluator label 才能获取参考选择集合。

停止时的正确结论是：

> 当前 V5.1 能验证 candidate-frontier 的质量—成本关系，但不足以验证同一 memory bank 上的时间复现共访问与逻辑布局局部性。

不得把不同 bank 的 memory ID 合并成一个全局 bank。

---

## 3. 参考选择集合：固定、无标签、可复现

### 3.1 主参考选择器

主参考选择器固定为：

```text
reference_selector_primary = dense_r128_b64_k8
```

这是 V5.1 已经定义的一个高预算 dense + rerank 选择路径。它仅用于生成一致的“参考选择集合”，不是 ground truth，不是所有 setting 的最好方法，也不是 AdaRecall 的固定线上入口。

对事件 `q`：

```text
A(q) = dense_r128_b64_k8 输出的 final selected memory IDs
|A(q)| ≤ 8
```

`A(q)` 只用于：

- 在训练期构建历史共访问结构；
- 在测试期离线测量不同布局下的集中程度。

### 3.2 无标签敏感性检查

主实验完成后，若可以获得完整 final IDs，再对以下选择路径重复同一回放：

```text
reference_selector_sensitivity = dense_r64_b32_k8
```

目的：检验结果是否完全依赖一个参考选择器。不得因为任务分数、Pareto 或其他 evaluator 结果更好而替换该方法。

### 3.3 若既有 trace 缺 final IDs

允许补跑 selection-only materialization，但必须：

- 复用 V5.1 runtime manifest、memory bank、真实 BGE、真实 MiniLM 与原始 `R/B/K`；
- 不调用 Qwen；
- 不读取 evaluator labels；
- 不改变 V5.1 eligible-event 顺序、warmup 边界或缓存语义；
- 新产物写入独立 E1 目录；
- 明确标记为“补齐 stable memory IDs 的 selection-only materialization”。

---

## 4. 因果切分与 memory 可见性

### 4.1 切分

对每个通过审计的 `runtime_bank_id`，按真实因果顺序切分：

```text
最早 70% 的非-warmup eligible event → train
最晚 30% 的非-warmup eligible event → test
```

禁止随机切分。小 bank 的实际 train/test 数必须完整写入 report；不能跨 bank 合并 memory ID 来凑样本。

### 4.2 可见 memory

- 若能恢复每个 event 的 memory snapshot：训练布局只能使用训练期可见 memory；测试中新写入的 memory 只能按真实写入顺序进入独立 append-only delta Tile，且所有方法使用完全相同的 delta policy。delta 不计为超图收益，需单独报告。
- 若只能确认固定静态 bank：允许做静态 layout replay，但报告必须明确“不验证持续写入与增量布局稳定性”。

---

## 5. Tile 的定义

Tile 是固定容量的**逻辑存储块**：

```text
memory_id → logical_tile_id
```

主实验设置：

```text
Tile capacity C ∈ {8, 16, 32} memory items
```

这些只是逻辑粒度敏感性参数，不是 DRAM row size、PIM bank capacity 或任何特定硬件参数。

约束：

```text
每个 memory 在一个 layout view 中恰好属于一个 Tile
每个 Tile 的 item 数 ≤ C
```

禁止将任何结果称为 row activation、bank conflict、PIM latency、bandwidth 或 energy。

---

## 6. 必须比较的布局

所有布局必须使用同一 bank、同一 train event、同一可见 memory universe、同一 `C`、同一 test event 和同一参考集合 `A(q)`。

| Layout | 真实输入 | 规则 | 注意 |
|---|---|---|---|
| `random_balanced` | 真实 memory ID | 固定 seed 的稳定伪随机排列，每 C 个填一 Tile | 仅是算法下界，不是生产布局 |
| `write_time_order` | 真实 write order / timestamp | 按真实写入顺序或时间顺序填 Tile | 字段缺失则 `not_evaluable`；若二者相同只保留一个 |
| `semantic_balanced_cluster` | 真实 BGE embedding | 先聚类，再做容量约束分配 | 不能忽略超容量 cluster，需报告 spill/overflow |
| `pairwise_graph` | train 期真实 top-K | 统计两两共访问权重，在容量约束下共置高权 pair | 必须报告图规模和启发式 |
| `global_hypergraph` | train 期真实 top-K | 每个 event top-K 为一条超边，减少高权超边跨 Tile | 必须说明算法为启发式，不是最优超图分区 |
| `context_hypergraph` | 真实 runtime context + train top-K | 仅在同一 bank 有多个有效 context 时构建多个 logical view | 不满足条件必须 `not_evaluable` |

### 6.1 `random_balanced`

使用固定全局 seed 对稳定 memory ID 排序或伪随机排列，每 `C` 个填入 Tile。报告 seed。

### 6.2 `write_time_order`

只在存在真实 insertion order 或 timestamp 时使用。若没有该字段，不能用 event ID 或文件顺序冒充写入时间。

### 6.3 `semantic_balanced_cluster`

```text
num_tiles = ceil(memory_count / C)
```

使用真实 BGE embedding 和固定 seed 聚类。随后必须执行容量约束分配，保证每个 Tile 不超过 `C`。可用确定性的容量贪心修复、最小成本匹配或项目已有实现，但必须报告算法、overflow、spill 和最终 Tile size 分布。

### 6.4 `pairwise_graph`

训练期构图：

```text
vertex = memory item
edge(u, v) = train events 中 u 与 v 同时出现于 A(q) 的次数
```

若没有成熟图分区依赖，可实现以下透明贪心：

```text
1. 选择未分配节点中 weighted degree 最大者作为 Tile seed；
2. 反复加入与当前 Tile 累计边权最大的未分配节点；
3. Tile 满后开始下一个 Tile；
4. tie-break 按稳定 memory ID；无边节点按稳定顺序填充。
```

报告节点数、边数、孤立节点数和布局构造时间。不得称为最优图分区。

### 6.5 `global_hypergraph`

训练期构图：

```text
vertex = memory item
hyperedge = 一个 train event 的 A(q)
hyperedge weight = 相同 canonical member set 的出现次数
```

逻辑目标是减少高权超边跨越 Tile 的数量：

\[
HGCost(\pi)=\sum_{e\in E}w(e)\cdot(\lambda_\pi(e)-1)
\]

其中 `λπ(e)` 是超边 `e` 覆盖的 Tile 数。

若没有成熟超图分区工具，可实现透明的容量约束贪心：以 incident hyperedge 总权重最大节点作为 Tile seed；每次加入使下式边际增益最大的未分配节点：

\[
F(\pi)=\sum_{e\in E}w(e)\sum_t\left(\frac{|e\cap t|}{|e|}\right)^2
\]

该实现必须命名为：

```text
capacity-constrained greedy hypergraph layout heuristic
```

不是精确最优分区。

### 6.6 `context_hypergraph`：仅在可验证时实现

只有同时满足以下条件才允许实现：

- 同一 `runtime_bank_id` 在 train/test 中具有至少两个真实、运行时可见的 context 值；
- 各 context 使用可比较的同一 memory universe 或可追溯的共享 memory ID；
- context 来自 runtime manifest，例如 `protocol`、`task`、`setting` 或无标签可得的 history-size bucket；
- context 不是 evaluator label、任务指标、future event 或 selector score。

若这些条件不满足，标记：

```text
not_evaluable_on_current_v5_1_trace
```

绝不跨 bank 合并 memory 来制造 context view。

若满足条件：每个 context 训练一个 logical view；test event 用自身可见 context 的 view；样本不足时回退到 global view，并报告 fallback rate、view count、view multiplicity。不得把多视图写成零复制成本的物理布局。

---

## 7. 主评估：只测布局集中度

本轮不模拟在线检索。对于 held-out test event `q`，已知离线参考集合 `A(q)`，在每个 layout 下计算：

| 指标 | 定义 | 方向 | 性质 |
|---|---|---:|---|
| `tile_span@K` | `A(q)` 覆盖的不同 Tile 数 | ↓ | 逻辑碎片化 |
| `min_tiles_for_coverage@0.75` | 覆盖至少 75% 参考成员的最少 Tile 数 | ↓ | oracle layout-concentration |
| `min_tiles_for_coverage@1.00` | 覆盖全部参考成员的最少 Tile 数 | ↓ | oracle layout-concentration |
| `best_1/2/4_tile_coverage` | 包含参考成员最多的 1/2/4 个 Tile 可覆盖的参考成员比例 | ↑ | oracle layout-concentration |
| `reference_density@1/2/4` | 上述 Tile 中参考成员数 / Tile 内全部 memory 数 | ↑ | 逻辑纯度代理 |

`oracle` 的含义：评估时知道 `A(q)`，只用于衡量布局潜力；绝不表示在线系统提前知道该 Tile。

若部分参考 memory 无法映射，必须逐条记录原因：

```text
not_in_visible_bank
new_item_after_train_cutoff
missing_stable_id
layout_failure
```

不得静默丢弃。

---

## 8. 统计与正式范围

### 8.1 主范围

主实验优先覆盖 V5.1 中**全部**：

```text
protocol × item_ranking × setting
```

这样避免根据 NDCG、RMSE 或 Pareto 结果后验挑选“容易成功”的 setting。

若资源不足，按无标签规则缩减：

1. 每个 protocol 选择 eligible event 数最多的 `item_ranking` setting；
2. 并列时按 setting 名稳定字典序；
3. 在 `slice_resolution_report.md` 中报告所有计数和选择过程。

`rating_prediction` 可作为辅助对照，不强制要求出现超图优势。

### 8.2 聚合与 CI

同一 bank 内 event 相关，主统计应先在 bank 内求均值，再以 bank 为 paired bootstrap 单位：

```text
bootstrap iterations = 10,000
CI = 95%
```

至少比较：

```text
global_hypergraph vs semantic_balanced_cluster
global_hypergraph vs pairwise_graph
write_time_order vs semantic_balanced_cluster
context_hypergraph vs global_hypergraph（仅可评估时）
```

若可用 bank 数太少，必须如实说明。可补充 event-level paired bootstrap，但只能标为探索性结果，不能当作强独立性证据。

### 8.3 预注册敏感性

必须完整报告：

```text
C ∈ {8, 16, 32}
reference selector ∈ {dense_r128_b64_k8, dense_r64_b32_k8（若可获得）}
```

不能只报告最好看的 capacity 或 selector。

---

## 9. Codex 实现顺序与交付物

### Step 0 — 输入审计

输出：

```text
reports/.../e1_input_audit.md
reports/.../e1_input_audit.json
reports/.../limitations_and_blockers.md
```

未通过 Go 条件时停止。

### Step 1 — 仅必要时补齐 selection trace

输出：

```text
runs/.../reference_selection_trace_primary.jsonl
runs/.../reference_selection_trace_sensitivity.jsonl
configs_snapshot/.../reference_selection_config.json
```

每条记录必须包括 event ID、bank ID、event order、final IDs、可追溯的 runtime snapshot reference 和 `used_evaluator_labels=false`。

### Step 2 — 因果切分

输出：

```text
manifests/.../bank_causal_split.jsonl
reports/.../split_audit.md
```

### Step 3 — 构建布局

每个 bank、layout、capacity 输出：

```text
layouts/.../<bank>/<layout>/capacity_<C>/memory_to_tile.jsonl
layouts/.../<bank>/<layout>/capacity_<C>/layout_metadata.json
```

metadata 至少包含：layout 名、C、训练 event 数、memory 数、tile 数、tile size 分布、随机 seed、是否使用 test event、是否使用 evaluator label、算法描述、context view/fallback 信息。

### Step 4 — held-out 回放

输出：

```text
runs/.../layout_replay_per_event.jsonl
```

每条记录至少包括：event ID、bank ID、split、layout、C、reference selector、reference set size、mapped count、所有第 7 节指标、`is_oracle_layout_metric=true`、`used_evaluator_labels=false`。

### Step 5 — 汇总

输出：

```text
reports/.../layout_replay_summary.csv
reports/.../layout_replay_summary.md
reports/.../layout_replay_bootstrap.json
reports/.../slice_resolution_report.md
reports/.../limitations_and_blockers.md
README_e1.md
```

`README_e1.md` 必须明确写出：本实验是逻辑布局回放，不是 PIM 性能实验。

---

## 10. Smoke run 和正式运行

### Smoke run

先在一个通过审计的 bank 和少量真实 event 上走通：

- 选择 trace materialization（若需要）；
- train/test 因果切分；
- `random_balanced`、`semantic_balanced_cluster`、`pairwise_graph`、`global_hypergraph`；
- Tile 容量检查；
- test event 未参与训练检查；
- `used_evaluator_labels=false` 检查；
- reference ID 与 runtime bank 对齐检查；
- 一份人工可读的 event-level 示例。

Smoke run 只证明管线正确，不产生论文结论。

### 正式运行

正式运行必须固定 config、seed、代码版本（若项目有 git commit）、环境和依赖版本；保存所有 per-event 原始输出；不能只输出最佳结果。

---

## 11. 什么结果算好

### 支撑超图逻辑布局的结果

在 held-out test 上，若 `global_hypergraph` 相比 `semantic_balanced_cluster` 和 `pairwise_graph`：

- `tile_span@K` 更低；
- `best_L_tile_coverage` 更高；
- `reference_density` 更高或至少不下降；
- 差异在 bank-level paired bootstrap 中方向稳定；
- 在多个 C 下和 sensitivity selector 下不完全消失；

则可以谨慎表述：

> 在真实 V5.1 历史选择轨迹的 held-out 回放中，高阶共访问布局使参考选择集合更集中于较少逻辑 Tile，为后续 PIM-aware physical mapping 提供了逻辑局部性证据。

### 只能保留全局超图、不能强调多视图的结果

若：

```text
global_hypergraph 优于 pairwise / semantic
context_hypergraph 与 global_hypergraph 无稳定区别，或不可评估
```

则论文暂时应采用“execution-trace hypergraph with context-aware weighting”，不要强称已验证多视图物理布局。

### 不应进入 PIM 仿真的结果

若发生任一情况：

- semantic layout 持续优于超图；
- hypergraph 与 pairwise 无 held-out 稳定差异；
- 结果只在 train 成立；
- 对 selector 或 C 极端敏感；
- 大多数 bank 无法做因果切分；
- context view 只能通过跨 bank 拼接才成立；
- ID / visibility 无法可靠追溯；

则暂停 PIM 仿真。应先重新检查 hyperedge 定义、bank 粒度、trace 来源或上下文定义。

---

## 12. 对 Codex 的硬性约束

- 先审计，后实现；不要猜测字段、路径、bank 语义或写入顺序。
- 不得跨 bank 合并 memory ID，不得读取 evaluator label，不得使用 future event，不得静默丢弃无法映射的 memory。
- 不得根据任务分数选择 setting、selector 或 Tile 容量。
- 不得把逻辑 Tile 指标转换为任何 PIM、DRAM、Row、Bank、带宽、能耗或端到端 latency 结论。
- V5.1 trace 若不支持重复 bank 的因果序列，必须明确 No-Go；这是一项有效结论，不是实现失败。
