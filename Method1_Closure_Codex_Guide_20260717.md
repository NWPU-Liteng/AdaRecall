# 方法一完整实验闭环实施指南（交由 Codex 执行）

> 版本：2026-07-17 v2.1  
> 当前参考版本：`20260714_method1_fsa_dynamic_hypergraph_v21`  
> 本轮唯一目标：完成方法一的算法、实现、增量维护与实验指标闭环。  
> 方法二/PIM 仿真暂不实施；SOTA 全量对比与正式消融放在方法一闭环之后。

---

## 0. Codex 总任务

请不要只输出分析、计划或伪代码。应在服务器正式环境中审计 v21，创建独立 v22，实现本指南定义的方法一，完成测试、真实数据运行、指标汇总和结论审计，直到方法一达到“功能闭环 + 指标闭环 + 初步效果闭环”，或遇到必须由用户提供新权限/数据/模型的真实阻塞。

本轮优先级固定为：

1. 修复数据、Agent 生成和评测异常；
2. 将数据集专用逻辑改造成统一、可迁移的数据接口；
3. 实现具有明确算法贡献的方法一，而不是简单 ANN 包装；
4. 在方法一内部完成准确率、候选开销、选择时延、逻辑 Tile 激活、over-fetch、流量与增量维护指标；
5. 得到至少一个可解释的质量—开销非支配点或明确的失败诊断；
6. 闭环完成后再做 SOTA、完整消融和论文级持续调参。

### 0.1 运行环境

1. 当前聊天沙箱不作为正式实验环境。正式数据处理、模型推理和性能实验必须在沙箱外的服务器上执行。
2. 必须使用服务器上已有 Conda 环境：

   ```text
   agent-memory-profiling
   ```

3. 项目根目录以服务器实际目录为准。当前已知根目录为：

   ```text
   /data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling
   ```

4. 若 Codex 当前终端已经位于服务器并且 `conda env list` 能找到该环境，可直接执行。否则必须通过项目已经配置的远程服务器入口执行；若服务器地址/远程入口不可发现，停止正式任务并只向用户询问远程入口，不得在沙箱内用 CPU 结果冒充正式结果。
5. 不得使用 `base` 环境，不得创建替代环境，不得静默升级已有核心依赖。新增依赖只能安装到 `agent-memory-profiling`，安装前后都要导出环境快照。
6. GPU 只用于适合 GPU 的阶段：embedding、ANN 批量查询、cross-encoder reranking，以及经过正确性对照后确有收益的张量操作。Boolean 因子搜索的稀疏位图运算和多层分区可以保留在 CPU；不得为了“用了 GPU”而将低效控制流强行迁移到 GPU。

### 0.2 方法一与方法二的边界

本轮必须采用以下边界，不能把方法一应有的指标推迟到方法二：

#### 方法一本身必须完成

- 记忆的模糊语义与访问链超图构建；
- 查询条件化的候选前沿构建；
- logical Tile 布局与分层 Tile-opening 候选选择；
- dense/cross-encoder rerank 和最终 Top-K；
- 在线插入、删除、超边更新和局部布局维护；
- retrieval/evidence/answer quality；
- candidate construction、selection 和 rerank latency；
- logical Tile activations、over-fetch、useful/fetched bytes；
- index build、storage、update 和 compaction cost；
- 完整 Agent 的阶段时延和答案指标。

#### 方法二以后再完成

- 将方法一产生的位图、计数、剪枝与 Top-K 操作映射到具体 PIM 指令；
- PIM bank/row 时序、并行度、cycles、带宽和能耗仿真；
- 真实 PIM 或校准模拟器上的硬件 speedup/energy；
- Host–PIM 调度与硬件资源冲突。

因此，本轮的 Tile 指标是**方法一的逻辑访问指标**。方法二以后只负责证明这些操作在 PIM 上能否进一步加速。

### 0.3 本轮成功标准，不是 SOTA 标准

本轮不要求超过所有相关工作，但必须同时满足：

1. **功能闭环**：真实数据可完成“建图—布局—检索—选择—rerank—回答—增量更新”。
2. **算法闭环**：UOT 模糊关联、访问序列挖掘、双视图超图扩散、超图分区和分层 Tile-opening 预算选择都实际参与最终路径，不是只写接口或死代码。
3. **指标闭环**：准确率、证据召回、候选数、选择时延、逻辑 Tile、over-fetch、bytes、更新成本和 Agent 总时延均有机器可读结果。
4. **初步效果闭环**：相对公平 ANN 基线，至少出现以下任一现象：
   - 相同候选/Tile 预算下质量更高；
   - 相同质量下候选数、逻辑 Tile、bytes 或时延更低；
   - 质量略有下降，但系统开销明显下降并形成可解释 Pareto 点。

如果所有点均被公平基线支配，则功能闭环可以标记 PASS，但效果闭环必须标记 FAIL，并只允许在 train/validation 上进行有限诊断和修复；不得去 formal test 反复调参。

### 0.4 本轮暂不优先做

- 十余个外部 SOTA 的完整复现；
- 全量组件消融；
- 穷举超参数；
- 方法二/PIM 模拟器；
- 真实 energy/speedup；
- 最终论文出版级作图；
- 重新训练大型回答模型。

但代码必须为后续 SOTA、消融和 PIM trace 导出预留标准接口。

### 0.5 科学边界

1. `persona` 只能映射为通用 `namespace_id`。所有基线也必须在同一 namespace 内检索，不能把过滤其他患者造成的 20 倍扫描差异算成方法贡献。
2. `query_type` 只能用于结果分组，不得作为在线路由标签，除非真实部署也能获得且其预测误差被计入。
3. `evidence_group_id`、`acceptable_memory_ids`、gold answer 和 future access 不得进入在线候选构造、布局或 Tile 成本计算。
4. 没有原生 memory-read 日志时，可用冻结策略产生 `instrumented replay trace` 完成工程闭环，但不得称为真实 Agent co-access。
5. 没有绝对时间戳时可以用 `session/order` 作为逻辑顺序，但不能宣称真实墙钟衰减。
6. logical Tile 不是物理 PIM Tile。只能报告 logical activation/over-fetch，不得报告 PIM latency/energy。
7. protected ANN seed 可以进入候选池，但不能强制占据最终 Top-K 后再把“Top-K 不下降”当成实验发现。
8. OOF seeds 不是新增独立样本。统计以 persona/namespace 或 session cluster 为单位。
9. 统计显著性和系统实际意义必须同时报告：绝对毫秒、相对比例、Agent 总时延占比。

### 0.6 工程纪律

1. 不覆盖 v21 的模型、配置、日志、aggregate 和报告。
2. 新建：

   ```text
   versions/20260717_method1_advanced_closure_v22/
   ```

3. 每次运行保存唯一 `run_id`、resolved config、Git commit、输入 hash、模型版本和环境快照。
4. 耗时任务先 smoke，再全量；支持 checkpoint/resume。
5. 失败 attempt 不删除，记录失败原因和变化因素。
6. 一次实验只修改一个主要因素。
7. 未经用户授权不提交、推送、覆盖用户 Git 改动或静默下载大型模型。

---

## 1. 方法一的科学定位

### 1.1 要解决的问题

方法一不是单纯让 Top-K 排序更快。真正问题是：

1. 长期记忆具有语义模糊性，一条记忆可能同时属于多个主题或证据簇；
2. Agent 执行中会出现多条记忆共同或连续被访问的高阶局部性；
3. flat ANN 只按独立向量相似度排序，容易构造较大且证据密度低的候选前沿；
4. 即使最终候选数不大，候选在逻辑 Tile 中分布离散，仍会产生 over-fetch；
5. 记忆持续写入后，语义关系、访问链和布局都需要局部更新，不能频繁全局重建。

### 1.2 方法假设

需要验证的不是“超图在无限读取时必然比普通图精度高”，而是：

> 模糊语义与执行访问链构成的双视图超图，能够在受限候选数和逻辑 Tile 预算下形成更高证据密度的候选前沿，并通过超图感知布局降低 Tile 级过度取数；增量局部更新能够维持该质量—开销优势而无需全局重建。

### 1.3 方法一应显现的算法贡献

方法一至少包含以下四个相互衔接的算法模块，而不是无关算法堆叠：

1. **Unbalanced Optimal Transport Fuzzy Association**：处理多归属、簇不均衡和离群记忆；
2. **Decay-aware Sequential Hyperedge Mining**：从执行访问序列提取可变阶、有方向的局部访问关系；
3. **Query-conditioned Multi-view Hypergraph Diffusion**：从 ANN seed 在两个超图视图中传播相关性，并按查询可靠性自适应融合；
4. **Hierarchical Tile-opening Submodular Selection + Multilevel Hypergraph Partitioning**：分层优化证据覆盖、冗余、候选预算和逻辑 Tile 成本。

它们分别对应“语义不确定性—执行局部性—查询扩展—质量/访存联合优化”，不得只为了显得复杂而引入。

必须严格区分“使用高级算法”和“提出论文创新”：UOT、PrefixSpan、PPR、超图分区和次模优化本身都是已有算法家族，单独接入它们不能宣称创新。v22 当前可验证的**创新候选**应集中为：

1. 将模糊语义关系与 Agent 执行访问链统一为查询条件化双视图超图；
2. 用逐查询可靠性估计控制两个视图的传播，而不是固定融合；
3. 把候选质量与 logical Tile 开启成本写成分层预算优化问题；
4. 用受影响超边驱动的局部布局修复支持在线增量。

只有 FULL 相对公平基线在质量—候选—Tile—时延上形成稳定非支配点，并在后续消融与外部方法对比中成立，才能把其中某项写成论文贡献；闭环阶段统一称为“设计”或“创新候选”。

### 1.4 相对已有工作的设计差异

- 相比 flat ANN：不把 memory 视为独立点；
- 相比普通 pairwise graph：把一次多记忆共同访问保留为一个原子超边，并直接优化超边跨 Tile 代价；
- 相比一般 HyperGraphRAG：重点不是从文档抽取 n-ary 事实，而是联合表示模糊语义与 Agent 执行访问链；
- 相比 STARC 式语义聚类共置：不仅按语义聚类，还利用执行共访问关系和查询预算进行动态前沿选择；
- 相比 MenSelect/PRISM：方法一减少进入昂贵 reranker 的候选数与分散度，属于 pre-rerank；后者优化已经构造好的候选池，二者以后可以组合。

这些差异目前只是设计定位，只有实验通过后才能写成论文结论。

### 1.5 算法依据与外部定位锚点

工作区论文只是起点。Codex 实现和撰写 related work 时至少复核以下原始论文/项目，不得从二手博客推导算法或 SOTA 结论：

- [UOT Sinkhorn 的目标与复杂度分析（ICML 2020）](https://proceedings.mlr.press/v119/pham20a.html)；
- [PrefixSpan 原始顺序模式挖掘方法](https://hanj.cs.illinois.edu/pdf/span01.pdf)；
- [KaHyPar 高质量多层超图分区](https://arxiv.org/abs/2106.08696)；
- [单调次模函数的 knapsack 约束算法](https://www.sciencedirect.com/science/article/abs/pii/S0167637703000622)；
- [有向超图表征与传播（AISTATS 2024）](https://proceedings.mlr.press/v238/ma24b.html)；
- [HyperGraphRAG](https://arxiv.org/abs/2503.21322) 与 [HGRAG](https://arxiv.org/abs/2508.11247)：用于区分“n-ary 文档事实超图”和本文“模糊语义 + 执行访问链超图”；
- [LongMemEval](https://arxiv.org/abs/2410.10813) 与 [MemoryAgentBench](https://arxiv.org/abs/2507.05257)：作为闭环后的静态长期记忆和增量交互迁移目标。

这些文献只证明算法来源与比较对象存在，不证明 v22 新颖或有效；新颖性必须由问题定义、组合方式、消融和实验结果共同支持。

---

## 2. v21 的保留、降级与修复

### 2.1 允许复用

- train-only provenance、causal eligibility 和泄漏审计；
- memory/query embedding 的读取、冻结和 hash；
- residual projection 与 log-domain UOT/Sinkhorn 数值实现；
- ANN seed、候选 cap 和 OOF runner；
- overlay、snapshot、atomic publish、tombstone、compaction；
- stage latency、bytes、logical Tile 和 query-level JSONL 统计；
- 现有测试框架与失败产物管理。

### 2.2 从主路径移除或降级

1. Persona sharding：改为所有方法共享的 namespace 层；不计为算法收益。
2. 旧 QESA：direct retrieval 为负，保留 legacy variant，不作为 v22 默认路由器。
3. 固定 arity-3 bundle：删除；用访问序列自然产生的可变阶有向超边替代。
4. 旧 learned QHDS：保留后续对照；v22 使用可审计的分层 Tile-opening submodular objective。
5. 强制 protected Top-5：默认关闭。
6. v21 的 `E2E latency` 改名为 `retrieval_pipeline_latency`。
7. SHA256 append 保留为 stress；新增 session/order causal replay。

### 2.3 Agent 生成链路必须先修复

v21 所有 handoff 准确率均为 `0.259041`，completion tokens 平均固定为 `48.0`。正式方法实验前必须运行：

```text
no-memory
random-memory
retrieved-memory
gold-evidence-only
```

逐项检查：

- prompt/context/output hash；
- B8/B16/B32 的实际 memory IDs 和 token 数；
- 1800-word cap 是否使不同 B 得到相同上下文；
- `max_new_tokens=48` 是否导致截断；
- EOS/stop token；
- 缓存 key 是否遗漏 context；
- exact match、归一化、LLM judge 是否读取正确字段；
- gold context 是否显著优于 random/no-memory。

若 gold context 仍不能提高准确率，先修复回答模型、prompt 或 scorer，禁止继续把检索质量当作最终答案结果。

---

## 3. 数据集无关接口

建立独立 schema：

```text
MemoryRecord
  memory_id: str
  namespace_id: str
  content: str
  embedding_ref: str | ndarray
  event_order: int | None
  session_id: str | None
  metadata: dict

QueryRecord
  query_id: str
  namespace_id: str
  text: str
  embedding_ref: str | ndarray
  cutoff_order: int | None
  metadata: dict

AccessEvent
  event_id: str
  namespace_id: str
  episode_id: str
  step_order: int
  query_id: str
  accessed_memory_ids: list[str]
  selected_memory_ids: list[str]
  task_success: float | None
  source: native_trace | instrumented_replay

EvaluationRecord
  query_id: str
  evidence_groups: list[list[str]]
  gold_answer: str | list[str]
```

要求：

- `EvaluationRecord` 与在线输入物理分离；
- `namespace_id` 只决定可见范围，不进入得分；
- `cutoff_order` 只执行因果过滤；
- query type 只做结果分组；
- MedMemoryBench 通过 adapter 转换；
- 未来 LoCoMo、LongMemEval、MemoryAgentBench 只增加 adapter，不修改方法核心。

---

## 4. 方法一算法设计

### 4.1 模糊语义视图：UOT-FSA

#### 4.1.1 表征适配

冻结基础 embedding，并保留低秩残差适配：

```text
z_i = normalize(x_i + W_up GELU(W_down x_i))
```

训练模式分为：

1. `generic_unsupervised`：不使用 evidence label，保证迁移能力；
2. `train_supervised_adapter`：只用 train evidence/hard negatives 训练 residual metric，用于 MedMemoryBench 质量增强。

默认先复现 v21 稳定 residual checkpoint，同时必须实现无监督 fallback。在线推理不得读取 supervision。

#### 4.1.2 Unbalanced Optimal Transport

设 memory 表征为 `z_i`，prototype 为 `p_k`：

```text
C_ik = 1 - z_i^T p_k
```

求解带熵正则和边缘 KL 松弛的 UOT：

```text
min_T <T, C>
      + eps * KL(T || a b^T)
      + tau_m * KL(T 1 || a)
      + tau_p * KL(T^T 1 || b)
```

要求：

- log-domain Sinkhorn；
- dustbin prototype 处理离群 memory；
- 一条 memory 允许属于多个 prototype；
- row-normalized transport 作为 fuzzy incidence `H_sem`；
- 检查 marginal residual、NaN/Inf、prototype collapse、effective prototype count；
- prototype 数、正则和 dustbin 写入配置。

UOT 的作用必须通过以下中间指标验证：

- fuzzy membership entropy；
- prototype effective count；
- outlier/dustbin ratio；
- evidence-group co-location；
- 相对 spherical k-means 的候选质量和存储开销。

闭环阶段只运行一个稳定默认配置和一个安全 fallback，不进行大规模 recipe 搜索。

### 4.2 执行局部性视图：衰减感知顺序超边挖掘

#### 4.2.1 轨迹来源

优先级：

1. 原生 Agent memory-read/access trace；
2. 冻结基线在过去训练查询上产生的 instrumented replay；
3. 仅为工程测试构造的 deterministic synthetic trace。

禁止使用测试 gold evidence 生成访问边。instrumented replay 必须冻结日志策略、按 session/order 执行，并与待评价方法分离。

如果 MedMemoryBench 不具备天然多步 episode，则：

- 用 persona/session/order 组织受控顺序 replay；
- 明确报告它不是 native co-access；
- 本轮完成算法和指标闭环；
- 后续再在 MemoryAgentBench/MemoryArena 验证真实执行局部性。

#### 4.2.2 Bounded PrefixSpan / Sequential Motif Mining

对每个 episode 的访问序列运行有界 PrefixSpan 或等价的投影数据库序列挖掘：

```text
min_length = 2
max_length = 4
max_step_gap = configurable
min_support = configurable
```

每个 Agent step 视为一个 itemset；同一步内多条 memory 是无序集合，不得按日志写入顺序伪造先后关系。对 motif：

```text
M = <I_1, I_2, ..., I_L>
tail(M) = union(I_1, ..., I_{L-1})
head(M) = I_L
e_access = tail(M) -> head(M)
```

motif 的 step 顺序单独保存在 edge metadata；用于无向布局视图时才把所有 itemset 取并集。若同一 memory 在 tail/head 重复，保留为可解释回访边，但单独统计 self-return ratio。

同时生成两个视图：

- directed incidence：用于 query-time 下一步候选传播；
- undirected co-access incidence：用于逻辑 Tile 共置。

#### 4.2.3 访问边权

避免热门 memory 仅凭频次主导，默认边权结合 support、lift/PMI、步距衰减和成功贡献：

```text
w_e = log(1 + support_e)
      * max(PMI_e, 0)
      * exp(-mean_step_gap_e / tau)
      * (1 + kappa * success_rate_e)
```

要求：

- support 做平滑，避免小样本极端权重；
- 限制单 memory 最大 incident edge 数；
- 对高频 hub 做 degree normalization；
- 输出 edge arity、support、PMI、age、success、source；
- 删除或失效 memory 后更新关联 edge；
- 训练、calibration、formal 的 edge stores 隔离。

### 4.3 查询条件化双视图超图扩散

#### 4.3.1 Seed initialization

先在 namespace 内运行真实 ANN，得到 seed score 向量 `s_0`。ANN seed 进入扩散起点，但不强制进入最终 Top-K。

#### 4.3.2 Semantic hypergraph propagation

使用稀疏归一化 incidence 构造传播算子：

```text
P_sem = D_v^-1 H_sem W_sem D_e^-1 H_sem^T
```

令所有分数为**行向量**，运行截断 Personalized PageRank / restart diffusion：

```text
r_sem^(t+1) = (1 - mu_sem) s_0 + mu_sem r_sem^t P_sem
```

#### 4.3.3 Directed access propagation

对 access tail/head incidence：

```text
D_head(e,e) = sum_v H_head(v,e)
D_out(v,v)  = sum_e H_tail(v,e) W_acc(e,e)
P_acc = D_out^-1 H_tail W_acc D_head^-1 H_head^T
r_acc^(t+1) = (1 - mu_acc) s_0 + mu_acc r_acc^t P_acc
```

`D_head^-1` 负责把一次超边转移的概率归一到多个 head memory；不能省略后再把非随机矩阵称为 random walk。无 outgoing access edge 的行保持为零，并由 restart 项回到 seed。实现必须检查非悬空行的 row sum 约等于 1，并与手算小图对照。

禁止构造 dense `N×N` 矩阵；使用稀疏 incidence、frontier truncation 和最多 2～4 轮传播。

#### 4.3.4 Reliability-adaptive view fusion

旧 QESA 失败后，不直接再堆一个黑盒 MLP。默认先实现可审计的可靠性门控：

```text
c_sem = ANN seed score mass incident to non-dustbin semantic edges
c_acc = ANN seed score mass incident to eligible access edges
h_sem = normalized fuzzy-membership entropy in [0, 1]
s_acc = clipped/log-scaled mean access support in [0, 1]
f_acc = mean access-edge freshness in [0, 1]

rho_sem = c_sem * (1 - h_sem)
rho_acc = c_acc * s_acc * f_acc
[g_sem, g_acc] = [rho_sem, rho_acc] / (rho_sem + rho_acc + eps)
beta_view = beta_max * clip(rho_sem + rho_acc, 0, 1)
```

先把 dense、semantic 和 access 分数分别做固定的非负 rank/robust normalization，再融合：

```text
r_i = (1 - beta_view) * dense_i
      + beta_view * (g_sem * r_sem_i + g_acc * r_acc_i)
```

要求：

- access coverage 低时自动退化为 semantic/dense；
- semantic entropy 高时允许提升 access view 权重；
- 两个 `rho` 都为 0 时直接退化为 dense，不得产生全零候选；
- `beta_max`、support clip 和 freshness decay 只在 train/calibration 冻结；
- 门控分量逐查询保存；
- 后续可加入 train-only learned gate，但只有超过确定性 gate 后才能进入主路径。

这一设计既保留 query conditioning，又避免复现旧 QESA 的近 query-independent collapse。

### 4.4 双视图超图的逻辑 Tile 布局

Tile 激活是方法一指标，因此方法一必须实际生成逻辑布局，而不是等方法二。

#### 4.4.0 Logical Tile 模型先冻结

logical Tile 是抽象访存单元，不是 PIM 硬件结论。正式评价前必须冻结：

- metadata、feature/vector、payload 三类 Tile 的容量与字节布局；
- memory 到 Tile 的打包顺序、对齐、跨 Tile 记录处理和缓存规则；
- 一个主容量，以及 `0.5×/1×/2×` 容量敏感性；
- 所有方法共享完全相同的容量和计数器实现。

若记录大于单 Tile，必须按确定性规则分片并计入全部 touched Tiles。主容量不得在看到 formal 结果后选择。方法一可以报告 logical Tile/bytes/over-fetch，但不能由此换算 PIM cycles 或 energy。

#### 4.4.1 布局目标

将 semantic 与无向化 access hyperedges 取并集，构造布局权重：

```text
E_layout = E_sem union E_access_undirected
w_layout(e) = eta_sem * 1[e in E_sem] * normalized_w_sem(e)
            + eta_acc * 1[e in E_access] * normalized_w_acc(e)
```

两类边权必须先在 train 上做稳健归一化，防止仅因数值尺度不同而让某一视图支配布局。

设 `lambda_e(P)` 为超边 `e` 跨越的 Tile 数，最小化：

```text
J(P) = sum_e w_layout(e) * (lambda_e(P) - 1)
       + rho * capacity_imbalance(P)
       + xi * remap_cost(P, P_previous)
```

约束：

```text
每条 memory 单副本
每个 Tile 的 vertex-byte weight <= tile_capacity
不同 namespace 不能混排
```

#### 4.4.2 Multilevel Hypergraph Partitioning

实现或接入标准多层超图分区流程：

1. heavy-edge matching/coarsening；
2. balanced initial partition；
3. uncoarsening；
4. FM-style local refinement；
5. incremental mode 加入 remap penalty。

依赖策略：

- 先检查环境是否已有 KaHyPar/hMETIS 或项目内分区器；
- 若存在，封装统一 backend，并固定版本；
- 若不存在，先实现 deterministic greedy + FM reference backend 完成闭环；
- 需要安装新库时先导出环境快照并请求用户确认；
- 不得用普通 k-means 冒充超图分区。

#### 4.4.3 必须输出的布局指标

- total logical Tiles；
- weighted hyperedge connectivity/cut；
- semantic/access hyperedge span；
- memory duplication factor，默认必须为 1；
- Tile occupancy distribution；
- layout build time 和 peak memory；
- incremental remapped memories/Tiles；
- layout storage bytes。

### 4.5 Hierarchical Tile-opening Submodular Frontier Selection

#### 4.5.1 Candidate pool

从 dense seed、semantic diffusion 和 access diffusion 的 union 中取 bounded pool `U_q`，再执行预算选择。必须保存候选来源和去重过程。

#### 4.5.2 Utility

定义单调次模效用，兼顾相关性、超边覆盖和多样性：

```text
F_q(S) = lambda_r * sum_{i in S} relevance_i
       + lambda_h * sum_e q_e log(1 + sum_{i in S} H_ei)
       + lambda_d * sum_c sqrt(sum_{i in S} membership_ic)
```

其中：

- `relevance_i` 来自 dense + dual-view diffusion；
- `q_e` 是查询对超边的激活权重；
- 对数/平方根使新增同类候选的边际收益递减，减少冗余。

所有 relevance、`q_e`、membership 和 lambda 必须非负；否则 `F_q` 不再保证单调，不能套用单调次模算法。

#### 4.5.3 分层求解，而不是错误复用 CELF

不能直接用：

```text
Delta F / (new unique Tiles + bytes)
```

做标准 CELF。原因是其他候选先打开某个 Tile 后，某候选的边际 Tile 成本会下降，旧 ratio 可能上升，不再是可安全复用的上界。

v22 改用两层求解。先令 `B_t` 为候选池中位于 logical Tile `t`、按不读取 payload 的 relevance 固定截断到 `bundle_cap` 后的候选 bundle。`bundle_cap` 在 train/calibration 冻结，防止大 Tile 仅靠候选多而获得虚高效用。

**Stage 1：Tile opening**

```text
G_q(A) = F_q(union_{t in A} B_t)
max G_q(A)
s.t. |A| <= tile_budget
```

固定大小 logical Tile 的开启成本为 1，因此 `G_q` 对 Tile 集合仍是单调次模函数，可使用 lazy greedy/CELF，并比较 best singleton。这里的性质只针对 bundle surrogate `G_q`，不代表两层联合问题已全局最优。若以后引入不同 Tile 成本，必须切换到经过验证的 knapsack 算法；不得沿用 unit-cost 近似保证。

**Stage 2：opened Tiles 内的 memory selection**

```text
c_i = c_score + c_payload * payload_bytes(i)
max F_q(S), S subset union_{t in A} B_t
```

约束：

```text
|S| <= frontier_cap
payload_bytes(S) <= byte_budget（启用时）
```

在固定 item cost 下使用 lazy marginal-gain queue，并按 `Delta F / c_i` 选择，同时比较 best singleton；可追加一次 Tile 1-swap 和 memory 1-swap 局部改进。由于同时存在 cardinality 与 byte 两个约束，除非实现严格对应的已证明算法，否则只称为 heuristic，不报告理论近似比。

无 Tile 预算时跳过 Stage 1，对全部候选执行 Stage 2；只有 Tile 预算时也必须保留 frontier cap，避免“打开少量 Tile 但把整 Tile 全送入 reranker”。

#### 4.5.4 正确性与 regret 审计

要求：

```text
deterministic tie-break
每步 marginal gain / opened Tile / accumulated bytes
小池对 Tile 与 memory 联合约束做精确枚举或 branch-and-bound
greedy regret = (oracle_utility - greedy_utility) / oracle_utility
```

- 支持只限候选数、只限 Tile、Tile + 候选 + byte 联合预算；
- exact oracle 只用于 synthetic/small-pool 审计，不进入正式在线路径；
- utility 和 bundle 构造不得读取 gold evidence；
- Stage 1 与 Stage 2 分别计时，防止用预计算隐藏在线成本。

#### 4.5.5 Rerank and final Top-K

两条路径：

1. dense reference：快速 correctness 和质量曲线；
2. cross-encoder：验证减少 frontier 是否转化为真实 rerank latency。

完整闭环必须至少使用一个服务器已有 cross-encoder。若无可用模型，先发现本地缓存；仍无模型时列出候选、大小和依赖并询问用户，不得静默下载大型模型。

### 4.6 增量更新算法

#### 4.6.1 新 memory

```text
new memory
→ residual projection
→ fixed prototype UOT/soft assignment
→ semantic overlay incidence
→ namespace-local ANN overlay
→ provisional Tile insertion
```

#### 4.6.2 新访问事件

```text
new AccessEvent
→ bounded motif count update
→ support/PMI/decay/success weight update
→ access incidence update
→ affected edge list
```

#### 4.6.3 局部布局修复

只对受影响 memory/Tile 执行 FM-style local refinement，并加入 remap penalty。达到以下任一条件才全局 compaction/repartition：

- overlay ratio 超阈值；
- weighted cut 恶化超阈值；
- Tile occupancy/imbalance 超阈值；
- 查询 over-fetch 连续恶化。

#### 4.6.4 Snapshot semantics

复用 immutable snapshot、atomic publish 和 tombstone。查询只能看到一个完整版本，不能混合发布前后的 edge/layout。

### 4.7 方法一最终数据流

```text
Offline / historical
memory embeddings
→ UOT fuzzy semantic hypergraph
access sequences
→ bounded PrefixSpan directed access hypergraph
two views
→ multilevel hypergraph partition
→ logical Tile layout

Online query
query embedding
→ namespace-local ANN seeds
→ semantic/access hypergraph diffusion
→ reliability-adaptive view fusion
→ hierarchical Tile-opening submodular frontier selection
→ cross-encoder rerank
→ final Top-K payload
→ Agent answer

Incremental
new memory/access event
→ local incidence/weight update
→ overlay publish
→ local layout refinement
→ threshold-triggered compaction
```

---

## 5. 分阶段实施计划

### Phase A：服务器、环境和输入预检

记录：

```text
pwd
git status --short
git rev-parse HEAD
conda env list
conda run -n agent-memory-profiling python -V
conda run -n agent-memory-profiling pip freeze
nvidia-smi
```

检查：

- v21 和原始数据存在；
- train/calibration/formal query 及 embedding 完整；
- memory embedding 数量、维度、dtype、NaN/Inf；
- ID、namespace、session/order 唯一性与可解析性；
- formal 启动前 `present == expected`；
- 磁盘/GPU/CPU/运行中旧任务；
- 当前依赖是否支持 ANN、稀疏矩阵、cross-encoder 和分区器。

输出：

```text
reports/PREFLIGHT_ENVIRONMENT.md
aggregate/preflight_environment.json
aggregate/input_integrity_manifest.json
environment/conda_before.yml
environment/pip_freeze_before.txt
```

**Gate A**：输入不完整时不得启动正式全量或读取 formal 指标。

### Phase B：评测修复与统一数据层

1. 创建 v22；
2. 实现统一 schema 与 MedMemoryBench adapter；
3. persona→namespace；query type→统计 metadata；
4. train/calibration/formal 物理隔离；
5. 修复 Agent sanity test；
6. 创建固定 16～32 查询 smoke subset；
7. 建立 namespace-local ANN 公平基线。

**Gate B**：gold context 应优于 random/no-memory，或给出可确认的模型能力上限；adapter 无 cross-namespace/future leakage；ANN baseline 完整运行。

### Phase C：高级算法模块正确性

按顺序实现并测试：

1. UOT-FSA；
2. access trace logger；
3. bounded PrefixSpan/motif miner；
4. semantic/access sparse diffusion；
5. reliability gate；
6. multilevel/greedy+FM logical Tile partition；
7. hierarchical Tile-opening submodular selector；
8. dense/cross-encoder rerank；
9. query-level trace 和指标输出。

对每个模块做小型可手算 synthetic test。例如：

- 一个 memory 属于两个 semantic prototypes；
- 一条三步访问链生成有向超边；
- access coverage 为 0 时 gate 退化到 semantic；
- directed access operator 的非悬空行和为 1；
- Tile-stage lazy greedy 与 eager greedy 在小例上选择一致；
- FM move 不得违反 Tile capacity；
- exact enumeration 与 greedy utility/cost 可核对。

**Gate C**：所有高级模块在 full path 中被调用，输出非退化且通过单元测试；不存在 dense `N×N` 隐式构造。

### Phase D：方法一 smoke 集成

运行：

```text
canonical data
→ semantic/access hypergraph
→ logical layout
→ query diffusion
→ hierarchical Tile-opening selection
→ rerank
→ Agent answer
```

至少比较：

```text
B0: namespace-local ANN + chronological/original layout
B1: ANN + UOT semantic view + semantic layout
FULL: dual-view diffusion + hypergraph layout + hierarchical Tile-opening selector
```

这三种是闭环基线，不等同于正式论文消融。

**Gate D**：所有 smoke queries 有 frontier、Top-K、Tile 和答案；0 cross-namespace/future/deleted memory；不同方法的 prompt/context 确实不同。

### Phase E：train-OOF 与新 held-out revision

1. 先运行 train-OOF；
2. 固定一个 closure 配置，不做大规模调参；
3. 通过完整性和泄漏审计后读取 calibration；
4. formal embedding 缺失问题修复后，新建独立 revision；
5. 只在参数冻结且 sentinel 通过后运行正式 held-out；
6. 不覆盖 v21 formal attempt。

**Gate E**：有 query-level paired 结果、clustered CI、质量—候选—Tile—时延曲线和 Agent 结果。

### Phase F：增量闭环

同时运行：

1. SHA256 80/20 hash stress；
2. session/order causal prefix replay；
3. growth 1%、5%、10%、20%；
4. 10% delete/update stress；
5. no-remap、local-refinement、compacted 三种维护状态。

每个点比较 overlay/snapshot、compacted 和 full rebuild reference。

**Gate F**：selection mismatch 为 0；deleted/future memory 为 0；每个增长点有质量、Tile、query/update/publish/compaction/remap 指标。

### Phase G：方法一闭环判定

只对 B0、B1、FULL 做完整汇总：

- 质量；
- 候选；
- selection/rerank/Agent latency；
- logical Tile/over-fetch/bytes；
- index/build/storage；
- incremental cost；
- 失败类型。

如果 FULL 没有正向点：

1. 先判断是 FSA、LACA、diffusion、layout、selector、reranker 还是 Agent scorer；
2. 只允许在 train/validation 上做 1～3 个有假设依据的 revision；
3. 每个 revision 只改一个因素；
4. 仍无非支配点则保留负结果，停止 formal 重跑，向用户汇报需要重构的模块。

---

## 6. 本轮最小实验矩阵

### 6.1 真实质量轨道

```text
dataset       = MedMemoryBench English
methods       = B0, B1, FULL
frontier cap  = 16, 32, 64
final K       = 5, 8
tile budget   = unconstrained, matched budgets
reranker      = dense reference, one cross-encoder
split         = train-OOF, eligible held-out revision
```

报告：

- candidate recall@C；
- evidence-member recall@K；
- group coverage/all-groups-covered；
- useful density；
- reranker precision/nDCG（如标签支持）；
- final answer accuracy/F1/LLM judge；
- abstention/insufficient-information；
- gold-context answer upper bound。

### 6.2 选择时延轨道

分阶段：

```text
query embedding
ANN seed
semantic diffusion
access diffusion
view fusion
submodular selection
payload gather
cross-encoder rerank
Agent generation
```

报告 mean/p50/p95/p99、95% CI、绝对差、相对差和 Agent 总时延占比。

### 6.3 Logical Tile 轨道

在方法一完成，不等待方法二：

```text
layouts = chronological/original, semantic, dual-view hypergraph
candidate sets = common fixed candidate set, native per-method candidate set
```

必须分开两种实验：

1. **固定候选布局比较**：同一候选集合下只比较布局，隔离 Tile placement 收益；
2. **完整方法比较**：各方法使用自己的候选和布局，报告最终系统效果。

指标：

- activated metadata/feature/payload logical Tiles；
- useful/fetched bytes；
- over-fetch ratio；
- Tile occupancy/utilization；
- hyperedge span/cut；
- layout build time/storage；
- frontier/payload 两阶段的 Tile 激活。

### 6.4 原生性能轨道

- native per-namespace memory；
- batch=1；
- warm/cold cache；
- 单次与连续查询；
- CPU 线程和 GPU device 固定；
- ANN 使用真实 HNSW/IVF/DiskANN/项目已有后端，不只用 Python flat scan。

### 6.5 受控规模压力轨道

为审查候选构建是否可能成为实际瓶颈，新增只用于性能、不用于准确率主张的 stress：

```text
memory per namespace = native, 10K, 100K, 1M（资源允许）
retrieval calls/task = 1, 10, 50
```

要求：

- 扩容方式、重复/扰动规则、dtype 和存储写入 manifest；
- synthetic items 不进入答案质量结论；
- 图中标记 `controlled scale stress`；
- 报告超图索引存储、扩散访问量和构建成本随规模增长；
- 测量 cross-encoder 候选减少带来的真实绝对时延，而不是只看百分比。

### 6.6 增量轨道

```text
growth       = 1%, 5%, 10%, 20%
update order = hash stress, causal session-prefix
layout       = no-remap, local refinement, compacted
```

报告 insert/update/delete、query during update、publish、compaction、remap、质量和 Tile 演化。

---

## 7. 指标定义

### 7.1 质量

1. `candidate_recall@C`：可接受 evidence 中进入前沿的比例；
2. `group_coverage@C`：证据组中至少命中一个成员的比例；
3. `all_groups_covered`：查询所需全部证据组是否覆盖；
4. `useful_density@C`：候选中有效证据比例；
5. `answer_accuracy/F1`：最终答案质量；
6. `oracle_context_accuracy`：gold context 上界，用于定位生成瓶颈。

group coverage 不能代替答案质量，两者必须同时报告。

### 7.2 候选与算法成本

- ANN entries visited；
- activated semantic/access hyperedges；
- diffusion iterations 与 nonzero visits；
- union candidates 与去重数量；
- greedy heap pops/recomputations；
- candidates sent to reranker；
- payload tokens/bytes；
- index build time 和 index storage。

### 7.3 Logical Tile

```text
tile_activation = number of unique logical tiles touched
overfetch_ratio = fetched_bytes / useful_bytes
tile_utilization = useful_slots / fetched_slots
```

分别统计 metadata、feature/frontier 和 selected payload，不能只给合计。

### 7.4 时延

GPU 阶段正确 warm-up 和 synchronize；CPU 稀疏操作固定线程。至少报告：

```text
mean, p50, p95, p99, std, count, clustered 95% CI
```

每项优化同时给：

```text
absolute_delta_ms
relative_delta_percent
fraction_of_agent_total
```

### 7.5 增量

- insert/update/delete p50/p95/p99；
- atomic publish pause；
- compaction latency；
- query latency during update；
- snapshot mismatch；
- stale/deleted return rate；
- affected/remapped memories and Tiles；
- hyperedge count/arity/storage growth；
- quality/Tile drift across prefixes。

### 7.6 Pareto 评价

至少形成三组比较：

1. 相同候选数下 quality–latency–Tile；
2. 相同或近似质量下 candidates/latency/Tile；
3. 相同 Tile budget 下 quality。

本轮不要求 FULL 在所有点领先，但至少需要一个未被 B0/B1 同时在质量和开销上支配的点，才能标记初步效果闭环 PASS。

---

## 8. 公平性与泄漏审计

1. 所有方法在相同 namespace 和 causal cutoff 下运行；
2. 相同 query/memory embedding、ANN backend、reranker、generator；
3. 相同 frontier cap、final K、context/payload budget；
4. 算法比较与布局比较分离；
5. query type、evidence group、gold answer、future access 不在在线输入；
6. train/calibration/formal 的模型、trace、cache、edge store 和目录隔离；
7. formal 前 `present == expected`；
8. 看到 formal 后不得在同一 revision 修改参数再运行；
9. warm/cold、CPU affinity、GPU、并发和模型版本写入 manifest；
10. 不把多个 seed 的重复 query 当独立样本；以 persona/namespace 或 session cluster 计算 CI。

输出：

```text
reports/DATA_LEAKAGE_AND_FAIRNESS_AUDIT.md
aggregate/fairness_audit.json
```

---

## 9. 建议目录与脚本

```text
versions/20260717_method1_advanced_closure_v22/
├── configs/
├── src/v22/
│   ├── schema/
│   ├── adapters/
│   ├── semantic_uot/
│   ├── access_motifs/
│   ├── hypergraph_diffusion/
│   ├── tile_partition/
│   ├── frontier_selection/
│   ├── incremental/
│   ├── evaluation/
│   └── audit/
├── scripts/
│   ├── 180_preflight_environment.py
│   ├── 181_audit_inputs.py
│   ├── 182_build_canonical_dataset.py
│   ├── 183_run_agent_sanity.py
│   ├── 184_build_uot_semantic_view.py
│   ├── 185_collect_access_traces.py
│   ├── 186_mine_access_hyperedges.py
│   ├── 187_build_logical_tile_layout.py
│   ├── 188_run_method1_frontier.py
│   ├── 189_run_method1_agent.py
│   ├── 190_run_incremental_replay.py
│   ├── 191_run_scale_stress.py
│   ├── 192_summarize_method1.py
│   └── 193_audit_claims.py
├── tests/
├── runs/
├── traces/
├── layouts/
├── aggregate/
├── reports/
├── figures_diagnostic/
├── environment/
└── negative_results/
```

若编号冲突可顺延。每个脚本支持：

```text
--config
--run-id
--resume
--smoke
--device
--output-dir
```

禁止硬编码服务器路径、GPU、split、模型和预算。

---

## 10. 建议主配置

```yaml
run:
  version: v22
  seed: 20260717
  deterministic: true
  smoke: false

dataset:
  adapter: medmemorybench
  language: en
  split: train_oof
  namespace_field: persona_id
  event_order_field: session_order
  forbid_online_query_type: true

semantic_uot:
  enabled: true
  representation_mode: train_supervised_adapter
  generic_fallback: true
  prototypes: 64
  dustbin_mass: 0.05
  log_domain: true

access_hypergraph:
  enabled: true
  trace_source: instrumented_replay
  mining: bounded_prefixspan
  min_length: 2
  max_length: 4
  max_step_gap: configurable
  use_pmi: true
  use_step_decay: true
  use_success_weight: true

diffusion:
  semantic_steps: 3
  access_steps: 3
  restart: true
  adaptive_reliability_gate: true
  beta_max: configurable
  support_clip: configurable
  freshness_decay: configurable
  learned_gate: false

tile_layout:
  capacity_source: frozen_logical_packing_profile
  metadata_tile_bytes: configurable
  feature_tile_bytes: configurable
  payload_tile_bytes: configurable
  capacity_scales: [0.5, 1.0, 2.0]
  primary_capacity_scale: 1.0
  backend: auto_discover_or_reference
  objective: connectivity_plus_remap
  single_copy: true
  local_refinement: true

frontier:
  method: hierarchical_tile_opening_submodular
  tile_solver: lazy_greedy_unit_cost_plus_singleton
  item_solver: lazy_ratio_plus_singleton_and_local_swap
  exact_oracle_max_pool: 20
  caps: [16, 32, 64]
  tile_budgets: [null, 8, 16, 32]
  final_k: [5, 8]
  force_protected_final: false

reranker:
  dense_reference: true
  cross_encoder: auto_discover

incremental:
  hash_growth: [0.01, 0.05, 0.10, 0.20]
  causal_prefix_replay: true
  delete_ratio: 0.10
  remap_penalty: configurable
  compaction_threshold: configurable

evaluation:
  query_rows: true
  clustered_ci: true
  latency_percentiles: [50, 95, 99]
  save_score_components: true
  save_prompts_and_hashes: true
  failure_taxonomy: true
```

`auto_discover` 的最终选择必须写入 resolved config，不能静默改变模型或后端。

---

## 11. 测试要求

### 11.1 单元测试

- schema round-trip；
- namespace/cutoff eligibility；
- UOT marginal、dustbin、row-normalized incidence；
- PrefixSpan support 和有向 tail/head；
- PMI/decay/success weight；
- sparse diffusion 与小型 dense reference 一致；
- zero access coverage 的 gate fallback；
- Tile capacity、单副本和地址唯一；
- FM move objective/capacity；
- Tile-level `G_q` 的单调性/次模性抽查；
- Tile-stage lazy queue 与 eager greedy 一致；
- item-stage marginal gain 和固定 item cost；
- exact enumeration/branch-and-bound vs greedy regret；
- deleted/future/cross-namespace rejection；
- overlay/snapshot/compaction equivalence；
- resume 不重复计数。

### 11.2 集成测试

一次命令完成：

```text
adapter
→ UOT/access hypergraph
→ logical layout
→ diffusion
→ hierarchical Tile-opening selection
→ rerank
→ Agent answer
→ metrics
```

失败返回非零退出码，不能吞异常后生成空 summary。

### 11.3 性能测试

- correctness 通过后才计时；
- 关闭 debug 日志；
- GPU warm-up/synchronize；
- 固定 CPU 线程和并发；
- 保存原始 query-level 时间；
- 同配置重复足够次数；
- 报告 warm/cold cache。

---

## 12. 必须生成的产物

```text
aggregate/v22_manifest.json
aggregate/input_integrity_manifest.json
aggregate/agent_sanity_summary.json
aggregate/uot_semantic_summary.json
aggregate/access_hypergraph_summary.json
aggregate/layout_summary.json
aggregate/method1_query_metrics.jsonl
aggregate/method1_summary.json
aggregate/incremental_events.jsonl
aggregate/incremental_summary.json
aggregate/scale_stress_summary.json
aggregate/fairness_audit.json
aggregate/claim_eligibility.json
reports/METHOD1_V22_CLOSURE_REPORT.md
```

每条 query metric 至少包含：

```text
run/query/namespace/split
visible memory count
ANN seeds
activated semantic/access edge counts
gate weights
diffusion nonzeros/iterations
frontier IDs/source/score components
final IDs
logical Tiles touched by stage
useful/fetched bytes
stage latency
evidence metrics
answer metrics
failure category
```

为未来方法二额外导出但本轮不消费：

```text
traces/method1_operation_trace.jsonl
```

只需记录 bitmap union、counter update、threshold prune、Top-K merge 和 Tile address 等抽象操作；不要在本轮实现 PIM。

---

## 13. 必须生成的诊断图

本轮不要求出版级样式，但数据必须真实、图意清晰：

1. `quality_vs_frontier_cap`；
2. `quality_vs_tile_budget`；
3. `selection_latency_breakdown`；
4. `rerank_latency_vs_candidates`；
5. `logical_tile_activation_by_layout`；
6. `overfetch_by_layout_and_budget`；
7. `agent_latency_breakdown`；
8. `incremental_growth_quality_tile_latency`；
9. `scale_stress_latency_storage`；
10. `failure_taxonomy`。

绘图只能读取 aggregate，不得手写实验数值。

---

## 14. 方法一闭环验收清单

### 14.1 数据与评测

- [ ] 正式运行使用 `agent-memory-profiling`；
- [ ] 环境、Git、GPU、依赖和输入 hash 已保存；
- [ ] 所有 embedding 完整；
- [ ] Agent gold/random/no-memory sanity test 已通过或给出明确模型上限；
- [ ] 0 cross-namespace、future、deleted memory；
- [ ] formal 前 sentinel 通过。

### 14.2 高级算法链

- [ ] UOT fuzzy semantic incidence 非 collapse；
- [ ] PrefixSpan/顺序 motif 生成可变阶有向访问超边；
- [ ] 两个视图均实际参与 sparse diffusion；
- [ ] reliability gate 可退化、可解释；
- [ ] 多层/greedy+FM 超图分区满足容量和单副本；
- [ ] 分层 Tile-opening selector 使用冻结的 logical Tile 成本；
- [ ] cross-encoder reranker 实际运行；
- [ ] 所有模块在 FULL 路径中有非零调用和 query-level trace。

### 14.3 效果与指标

- [ ] candidate/evidence/answer quality；
- [ ] selection/rerank/Agent latency；
- [ ] logical Tile activation；
- [ ] over-fetch/useful/fetched bytes；
- [ ] index build/storage；
- [ ] update/publish/compaction/remap；
- [ ] native 与 controlled scale stress；
- [ ] clustered CI 与绝对/相对变化；
- [ ] 至少一个非支配点，或效果闭环明确标记 FAIL。

### 14.4 增量

- [ ] hash stress 和 causal prefix replay；
- [ ] overlay/compacted/full reference mismatch 为 0；
- [ ] LACA 只用过去事件；
- [ ] 局部 layout refinement 可运行；
- [ ] 各 growth 点有质量、Tile 和时延。

### 14.5 最终状态格式

```text
Method-1 functional closure: PASS/FAIL
Advanced algorithm path: PASS/FAIL
Quality metric closure: PASS/FAIL
Latency metric closure: PASS/FAIL
Logical-Tile metric closure: PASS/FAIL
Incremental maintenance closure: PASS/FAIL
Initial Pareto-effect closure: PASS/FAIL
Independent held-out generalization: PASS/FAIL/PENDING
SOTA competitiveness: NOT YET EVALUATED
Full ablation: DEFERRED
PIM acceleration: OUT OF SCOPE FOR THIS VERSION
```

---

## 15. 闭环后的后续工作，本轮不要抢跑

只有上述方法一 Gate 完成后，才进入：

1. SOTA：Embedding RAG/HNSW、HippoRAG 2、LightRAG、A-MEM、SimpleMem、HyperGraphRAG/HGRAG/Hyper-RAG；
2. 正式消融：FSA only、LACA only、pairwise replacement、without layout、without Tile cost、different diffusion/gate；
3. 跨数据集：LoCoMo/LongMemEval、MemoryAgentBench/MemoryArena；
4. 方法二：把 `method1_operation_trace.jsonl` 映射到 PIM simulator/hardware；
5. 论文级参数优化和出版级作图。

成对图消融不要求超图在无限读取预算下精度更高，重点比较相同质量所需的候选、Tile 和访问成本。

---

## 16. Codex 最终交付格式

最终回复用户时必须简洁说明：

1. 实际完成到哪个 Gate；
2. 新增/修改的核心算法与文件；
3. 使用的数据、模型、硬件和规模；
4. UOT、访问超边、扩散、布局和选择是否实际生效；
5. B0、B1、FULL 的准确率/召回；
6. 选择、rerank 和 Agent 总时延；
7. logical Tile、over-fetch 和 bytes；
8. 增量正确性和维护成本；
9. 是否形成初步非支配点；
10. 负结果、异常和当前允许的论文主张；
11. 下一步应优化哪个算法模块，而不是笼统说“继续调参”。

附上：

```text
一键 smoke 命令
一键正式 closure 命令
主配置
method1_summary.json
METHOD1_V22_CLOSURE_REPORT.md
诊断图目录
```

若因远程入口、缺失模型、数据损坏或权限阻塞，应报告精确证据和用户需要补充的最小信息；不得用“建议用户自行运行”代替当前可以安全完成的工作。
