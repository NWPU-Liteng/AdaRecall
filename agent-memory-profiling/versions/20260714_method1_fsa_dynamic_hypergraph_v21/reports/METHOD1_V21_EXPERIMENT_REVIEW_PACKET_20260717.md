# Method 1 v21 实验审阅整合包

> 用途：将当前方法一（Method 1 / FSA-QEHG）在 v21 版本中的实验设置、方法设计、阶段性结果、负结果、数据边界、可复现入口和待优化问题集中到一个文件，供外部 GPT 或研究者审阅。本文只整合已有产物，不新增实验结论。
>
> 审计时间：2026-07-17（Asia/Shanghai）
>
> 主版本目录：`versions/20260714_method1_fsa_dynamic_hypergraph_v21/`

---

## 0. 一句话结论

**静态检索/index/Tile/Persona 分片/静态增量发布的工程闭环已经完成；Gate C 通过；但方法一尚未形成独立 held-out 泛化和最终 Agent 效果闭环。**

当前最重要的限制不是代码缺失，而是实验边界：

1. CRC/LTT 校准均未通过。
2. 正式测试的 290 个查询全部缺少冻结测试 embedding，因此没有有效 formal held-out 指标。
3. train-OOF 上所有 Agent handoff 变体的答案准确率都为 `0.259041`，没有显示最终答案质量改善。
4. 原始数据没有独立 Agent memory-read/co-access 事件，不能支持真实动态访问顺序或因果 update claim。
5. 当前不能声称 PIM 硬件加速、PIM latency、energy 或 access-order benefit。

因此建议外部审阅者把当前状态标为：

> **Static Method-1 engineering closure; generalization and Agent-effect closure blocked.**

---

## 1. 版本、范围与主张边界

### 1.1 当前版本

| 项目 | 内容 |
|---|---|
| 方法版本 | `20260714_method1_fsa_dynamic_hypergraph_v21` |
| 方法正式名称 | FSA-QEHG：Fuzzy Semantic Association and Query-conditioned Evidence Hypergraph |
| 运行环境 | Conda environment `agent-memory-profiling` |
| 主要代码 | `src/v21/`、`scripts/81` 至 `scripts/175` 系列脚本 |
| 主数据来源 | MedMemoryBench-derived memory corpus，以及继承的 v21 train/calibration/formal query plans |
| 最后产物写入 | 2026-07-15 22:52（final design paper） |
| 当前运行状态 | 审计时没有发现 v21/Method 1 训练、评估或 launcher 进程 |

### 1.2 v21 允许主张

当前证据支持以下较窄的主张：

- 训练 OOF 上，模糊 UOT 语义关联能够提高证据组 co-location 指标。
- 在固定候选、预算和对照条件下，FSA-QEHG 稀疏路径可以形成质量/逻辑 Tile/host latency 的 non-dominated OOF 点。
- Persona x Prototype 分片可以在不改变返回成员和最终选择的情况下减少无效 rank-prefix 扫描。
- 采用确定性静态 base/new-memory 分区时，overlay、snapshot、atomic publish、compaction 可以保持选择等价。

### 1.3 明确不允许的主张

- 不把序列化 payload 顺序解释成 Agent 的真实 memory access demand。
- 不把 co-evidence/evidence group 解释成观测到的 co-access。
- 不把静态 append stress 解释成因果 dynamic-memory behavior。
- 不把 host/logical Tile 计数解释成 PIM 硬件 Tile、PIM latency、energy 或 speedup。
- 不宣称 Agent answer quality 已提升或已经有独立泛化保证。
- 不宣称 LACA/OACH、Hawkes、ordered access chain 已被验证。

原始 scope 修正见 [`METHOD1_V21_PIVOT_AND_DATASET_DECISION.md`](METHOD1_V21_PIVOT_AND_DATASET_DECISION.md)，动态可行性见 [`METHOD1_V21_DYNAMIC_FEASIBILITY_AUDIT.md`](METHOD1_V21_DYNAMIC_FEASIBILITY_AUDIT.md)。

---

## 2. 实验设置

### 2.1 数据与切分

| 对象 | 数量/定义 |
|---|---:|
| Memory items | `42,034` |
| Memory embedding dimension | `768` |
| Train queries | `1,355` |
| Calibration queries | `290` |
| Formal test queries | `290` |
| OOF 外层结构 | 5 fixed seeds x 5 session-blocked outer folds = 25 units |
| Fixed seeds | `20260712`, `20260713`, `20260714`, `20260715`, `20260716` |
| Session split | query/session blocked；训练时不读 calibration/formal labels |
| 训练证据关系 | `query -> evidence_group -> acceptable_memory_ids`，每个 group 是 OR set |
| 保护 ANN seed | 每个查询固定保护合法 ANN Top-5 |

主要输入配置为 [`configs/v21_data.json`](../configs/v21_data.json)。其中声明了：

- train-only 训练信号只能来自 train query embedding、memory embedding、train evidence groups 和 train hard negatives；
- 不允许使用 stage、workflow、future access、calibration/formal labels、payload serialization order、手工 semantic/recency score；
- eligibility 只能由 persona/session/type/order metadata 计算，不能用答案或 gold memory 生成。

### 2.2 Eligibility 与泄漏控制

train-only provenance audit 的结果：

| 检查 | 结果 |
|---|---:|
| train queries audited | 1,355 |
| eligible incidences | 1,138,785 |
| exact-list mismatches | 0 |
| cross-persona incidences | 0 |
| future-session incidences | 0 |
| current-session incidences | 27,523 |
| calibration accesses（该阶段） | 0 |
| formal-test accesses（该阶段） | 0 |

可复核报告：[`METHOD1_V21_ELIGIBLE_MEMORY_PROVENANCE_AUDIT.md`](METHOD1_V21_ELIGIBLE_MEMORY_PROVENANCE_AUDIT.md) 和 [`METHOD1_V21_DATA_LEAKAGE_AUDIT.md`](METHOD1_V21_DATA_LEAKAGE_AUDIT.md)。

### 2.3 评价原则

评价以 original query/session-cluster 为统计单位，主要报告：

- evidence-group co-location / group coverage；
- all-groups-covered rate；
- evidence-member recall；
- exact task success；
- fixed-count、fixed-byte、iso-quality、iso-cost；
- protected-top-5 non-degradation；
- rank entries visited、metadata bytes、payload bytes；
- metadata/payload/total logical Tile activations；
- stage latency mean/p50/p95/p99；
- Agent answer accuracy、prompt/completion tokens、retrieval/generation/total latency。

逻辑 Tile 定义为 host 访问单位：rank entry 为 4 bytes，metadata Tile 为 4 KiB，feature/payload 为 capacity-16 的 neutral single-copy memory Tile。实验没有物理 PIM 设备读写测量。

### 2.4 预注册与配置入口

| 阶段 | 配置 |
|---|---|
| UOT/Phase 1 | [`configs/v21_phase1_oof.json`](../configs/v21_phase1_oof.json)、[`configs/v21_uot_semantic.json`](../configs/v21_uot_semantic.json) |
| QESA | [`configs/v21_qesa.json`](../configs/v21_qesa.json)、[`configs/v21_qesa_r2.json`](../configs/v21_qesa_r2.json) |
| QHDS | [`configs/v21_qhds.json`](../configs/v21_qhds.json) |
| Sparse joint | [`configs/v21_sparse_joint.json`](../configs/v21_sparse_joint.json) |
| Effect closure | [`configs/v21_effect_closure.json`](../configs/v21_effect_closure.json) |
| Static update stress | [`configs/v21_persona_sharded_partition_update_stress_r1.json`](../configs/v21_persona_sharded_partition_update_stress_r1.json) |
| Freeze/closure | [`configs/v21_method1_closure_r2.json`](../configs/v21_method1_closure_r2.json)、[`configs/v21_method1_closure_r3.json`](../configs/v21_method1_closure_r3.json) |

---

## 3. 方法设计

### 3.1 总体数据流

```text
离线 memory
  -> 冻结 768-d embedding
  -> 残差低秩 projection
  -> 64 个 UOT fuzzy prototypes + incidence H
  -> Persona x Prototype rank shards

在线 query
  -> query embedding
  -> protected ANN Top-5
  -> QESA query-conditioned prototype activation
  -> bounded Persona shard rank-prefix lookup
  -> arity-3 semantic Bundle Beam
  -> Cap-64 candidate frontier
  -> protected QHDS monotone submodular selection
  -> B=8 single-copy payload gather
  -> Agent handoff

新 memory
  -> frozen projection/incidence
  -> Persona x Prototype overlay
  -> immutable snapshot
  -> atomic publish
  -> periodic compaction
```

完整设计文档：[`METHOD1_V21_FINAL_METHOD_DESIGN_PAPER_20260715.md`](METHOD1_V21_FINAL_METHOD_DESIGN_PAPER_20260715.md)。

### 3.2 冻结 embedding 与 residual projection

基础 embedding 不重新训练，以保证不同 OOF fold、index revision 和增量阶段处于同一语义坐标系。残差适配器为：

```text
g_theta(x) = normalize(x + W_up GELU(W_down x))
```

其中 embedding dimension `d=768`，bottleneck `r in {64,128}`，`W_up` 零初始化，使训练初始状态接近原始 embedding。训练采用 multi-positive InfoNCE：一个 query/evidence anchor 可以有多个 acceptable positive memory，hard negatives 从 train-visible frozen embedding 近邻中产生。

Phase 1 内层选择顺序为：

1. maximize inner group colocation；
2. minimize single-prototype domination；
3. maximize prototype effective count；
4. minimize build time；
5. minimize recipe id（稳定 tie-break）。

### 3.3 UOT-FSA fuzzy semantic association

使用 `K=64` 个 spherical prototypes，加一个 dustbin（prior mass 0.05）。记投影后的 memory embedding 为 `z_i`，prototype 为 `p_k`，代价为：

```text
C_ik = 1 - z_i dot p_k
```

采用 log-domain unbalanced Sinkhorn 和 KL relaxed transport。dustbin 之外的 transport logits 行归一化为 fuzzy incidence `H_ik`，因此一条 memory 可以同时属于多个 prototype：

```text
sum_k H_ik = 1
```

每个 prototype 的 rank list 按 `(-H_ik, memory_position)` 稳定排序。

### 3.4 Persona x Prototype rank sharding

全局 rank list `R_k` 被切成 `R_{u,k}`：同一 memory 只出现在其所属 persona 的 shard，不复制 rank entries。在线对每个激活 prototype 只读当前 persona shard，直到收集到 32 个合法成员：

```text
E_qk = FirstEligible_32(R_{u_q,k}, E_q)
```

由于 query 的合法 memory 都属于同一 persona，全局列表中其他 persona 的条目最终都会被 eligibility filter 丢弃；因此在相同 eligibility、rank key 和 prefix limit 下，分片理论上与全局列表严格等价，实测也验证了该等价性。

### 3.5 QESA query-conditioned activation

QESA 使用 hidden dimension 128 的两层 MLP，把 query embedding 映射为 `A=32` latent aspect 分布，再通过 learned aspect-to-prototype matrix 得到 prototype probability `pi_q`。最终 anti-collapse 配置使用 aspect temperature `T_a=0.1`，默认激活 `Top-16` prototype，active24 为质量档，active32 为诊断档。

QESA 的目标不是预测唯一 memory ID，而是预测当前 query 应激活的语义区域。输入允许字段仅包含 query embedding、protected ANN ids、eligible ids、frozen UOT incidence、frozen memory embeddings 和 frozen model parameters。

### 3.6 Protected ANN seed

冻结 ANN Top-5 合法 memory 为 `P_q`，在候选构造与 QHDS 中持续保护。它提供直接语义相似度下界，并防止 prototype path 在分布偏移时完全丢失局部近邻。所有选择预算都必须满足 `B >= 5`。

### 3.7 三元 semantic Bundle Beam

对不同激活 semantic edges 构造 arity-3 bundle，要求三个 prototype 与三个 memory 都互不相同。局部分数为：

```text
phi(q,i,k) = cosine(q,i) + pi_q(k) * H_ik
```

Bundle 分数为三个局部分数之和。默认有界参数：

| 参数 | 值 |
|---|---:|
| prototype tuple 保留数 | 8 |
| 每条 edge 的 member 前缀 | 4 |
| partial bundle beam | 32 |
| 最终 bundle 数 | 32 |

一个监督衍生的 learned bundle-locality 分支被保留为负结果诊断；它不是最终 sparse joint quality component。这里的 bundle 是 evidence association，不是 observed runtime co-access。

### 3.8 Cap-64 frontier

候选前沿按确定性顺序构造：

1. protected ANN Top-5；
2. retained bundle 的 unique members；
3. 已返回 semantic edge members 的 cosine-sorted fallback；
4. 总候选数封顶 64。

禁止 prefix extension、full eligible-by-prototype scan、evidence_group_id/acceptable_memory_ids/gold_bundle 作为在线输入。

### 3.9 QHDS monotone submodular selection

每个候选使用三个特征：

```text
c_i = cosine(query, memory_i)
a_i = H_i dot pi_q
f_i = [c_i, a_i, c_i * a_i]
```

hidden dimension 16 的 relevance MLP 用 Softplus 保证非负 `r_i`。集合 utility 为：

```text
U(S) = lambda_s * sum_k pi_q(k) log(1 + sum_{i in S} H_ik r_i)
     + lambda_m * sum_{i in S} r_i
```

用受保护集合 `P_q` 初始化，从边际增益最大的候选开始贪心，一次生成嵌套的 B8/B16/B32 选择序列。默认 handoff 为 B8；B16/B32 用于质量曲线和诊断。

### 3.10 单副本 payload 与 Agent handoff

所有 feature/payload 保持 capacity-16 neutral single-copy memory Tile，不做 feature replication。最终只 gather 被选择的 payload。上下文格式为：

```text
[memory_id | session_id | session_type]
memory text
```

上下文上限为 1,800 words，最后一条可截断到剩余容量。Agent 只使用提供 context，不调用外部知识或工具，证据不足时返回固定 `insufficient information`，最大生成 48 tokens。Agent 实验使用本地 Qwen。

### 3.11 静态增量发布

增量 API 为 `upsert`、`delete`、`snapshot`、`publish`、`compact`。新 memory 先计算 frozen projection/incidence，再进入 Persona x Prototype overlay rank stream；查询 merge base 与 overlay；snapshot immutable，使用原子指针切换；删除用 tombstone，周期性 compaction 合并有效 overlay。

为了不伪造时间顺序，使用 `SHA256(memory_id)` 将 42,034 条 memory 确定性切成：

- frozen base：33,627（80%）；
- new-memory pool：8,407（20%）。

在 1/5/10/20% 增长点发布新 memory，并在 appended growth 的 10% 注入删除。这验证的是静态发布正确性，不是 causal dynamic access。

---

## 4. 阶段性实验状态

| 阶段 | 状态 | 结论 |
|---|---|---|
| Scope/data pivot | 完成 | 从不可识别 ordered-access/LACA claim 收缩到 semantic FSA-QEHG |
| Eligibility/provenance audit | 通过 | 1,355 train queries，0 mismatch，0 cross-persona/future leakage |
| UOT numerical screen | 通过 | 27 个 preregistered recipes 中保留稳定、非 collapse recipes |
| Gate A UOT-FSA | 通过 | 25/25 OOF units converged，所有 seed 的 cluster CI > 0 |
| QESA Gate B r1 | 未通过 | 负结果保留，不替换原始 artifact |
| QESA Gate B r2 | 未通过 | 预注册 revision 仍未建立 direct retrieval advantage |
| QESA Gate B2 | 通过 | 仅证明 representation validity，不能替代 end-to-end Gate C |
| Learned bundle locality | 负结果 | learned scorer 未获提升，cardinality 预测是瓶颈 |
| QHDS fixed-budget | 完成 | association path 有效，但不是最终 Gate C |
| Sparse joint OOF | 完成 | 25/25 units；attempt_019 为 Gate C 输入 |
| Gold/learned utility oracles | 完成 | 仅提供 headroom，不是 deployable result |
| Gate C | 通过 | 54 个 qualifying contrasts |
| Persona sharding/Tile | 通过 | 选择等价，rank/metadata/Tile/lookup 均下降 |
| Matched E2E controls | 通过 | active16/24 各 25 units，6,775 rows，selections identical |
| Static incremental publication | 通过 | 1/5/10/20% 增长全部 mismatch=0 |
| Calibration CRC/LTT | 失败 | 唯一 calibration read 中 CRC/LTT 均未达目标 |
| Formal test | 无资格 | 290/290 frozen test embeddings missing |
| Agent train-OOF | 完成但负向 | 所有 handoff 变体 answer accuracy=0.259041 |

---

## 5. 实验结果

### 5.1 Gate A：UOT-FSA

5 个 seed 的 session-clustered OOF 证据组 co-location 如下。对照是 matched cosine-kNN；该指标不是最终 retrieval quality，也不是 temporal/co-access claim。

| Seed | UOT-FSA | cosine-kNN | Difference | Session-clustered 95% CI |
|---:|---:|---:|---:|---|
| 20260712 | 0.861070 | 0.769496 | +0.091574 | [+0.066111, +0.117693] |
| 20260713 | 0.857565 | 0.769496 | +0.088069 | [+0.062924, +0.114220] |
| 20260714 | 0.853198 | 0.769496 | +0.083702 | [+0.057238, +0.110214] |
| 20260715 | 0.866790 | 0.769496 | +0.097294 | [+0.071880, +0.123190] |
| 20260716 | 0.870910 | 0.769496 | +0.101415 | [+0.077216, +0.126295] |

结论：Gate A 通过；UOT-FSA 允许继续 QESA/QHDS 开发，但不能单独证明超图 end-to-end superiority。详见 [`METHOD1_V21_UOT_FSA_RESULTS.md`](METHOD1_V21_UOT_FSA_RESULTS.md)。

### 5.2 QESA：r1/r2 负结果与 B2 representation gate

原始 Gate B r1 在 B=16 下相对 flat cosine、same-capacity MLP 等结果为负；预注册 r2 revision 仍未通过 direct retrieval gate：

| Revision/comparator | All-query effect | Clustered 95% CI | Holm p |
|---|---:|---|---:|
| r1 vs query-independent prior | -0.003752 | [-0.007343, -0.000149] | 0.0703965 |
| r1 vs flat cosine | -0.123715 | [-0.131233, -0.116275] | 0.00019999 |
| r1 vs same-capacity MLP | -0.050959 | [-0.057286, -0.044620] | 0.00019999 |
| r2 vs query-independent prior | +0.007835 | [+0.003608, +0.012079] | 0.000299985 |
| r2 vs flat cosine | -0.112128 | [-0.120090, -0.104130] | 0.00019999 |
| r2 vs same-capacity MLP | -0.039373 | [-0.045690, -0.032892] | 0.00019999 |

B2 是在冻结 r2 checkpoint 上进行的 representation-validity audit，证明 QESA 的 prototype/group affinity representation 优于 query-independent prior 和 shuffled-group supervision，但没有证明 retrieval advantage：

| B2 metric | Comparator | Effect | Clustered 95% CI |
|---|---|---:|---|
| group affinity | query-independent prior | +0.008222 | [+0.007350, +0.009100] |
| group affinity | shuffled QESA | +0.010522 | [+0.009680, +0.011375] |
| target cross entropy | query-independent prior | +0.143629 | [+0.131506, +0.155711] |
| target cross entropy | shuffled QESA | +0.127989 | [+0.116875, +0.139034] |

重要诊断：r1 的 mean prototype-coordinate std 为 `0.000784`，near-query-independent warning 为 `True`；r2 后 mean aspect-map cosine 为 `0.335003`，prototype-coordinate std 为 `0.007356`，但直接 retrieval Gate B 仍不通过。故 B2 只授权 QHDS development，不能改写 Gate B negative result。

### 5.3 Evidence-bundle locality：负结果

这是 train supervision-derived compositional locality，不是 runtime co-access。K>=2 查询上，learned permutation-invariant bundle scorer 相对 independent-member comparator：

| 指标 | Effect | Session-clustered 95% CI |
|---|---:|---|
| all-groups-covered rate | -0.008092 | [-0.013258, -0.003466] |
| bundle recall | -0.023603 | [-0.033702, -0.013785] |

相对 pairwise-only，bundle recall 提高 `+0.016763`，但 all-groups-covered effect 为 `0.000000`，因此 learned bundle scorer 没有被提升为主质量组件。learned cardinality head 在 multi-group OOF 上只 exact `25.43%`，是未解决瓶颈。详见 [`METHOD1_V21_EVIDENCE_BUNDLE_LOCALITY_PHASE_DECISION.md`](METHOD1_V21_EVIDENCE_BUNDLE_LOCALITY_PHASE_DECISION.md)。

### 5.4 QHDS fixed-budget

QHDS 阶段在 Gate C 前完成。代表性 B=8/16/32 结果如下：

| B | Comparator | All effect | Clustered 95% CI |
|---:|---|---:|---|
| 8 | flat cosine | +0.006568 | [+0.002943, +0.010163] |
| 8 | qesa association | +0.053444 | [+0.047988, +0.058910] |
| 8 | QHDS without latent aspects | +0.000640 | [-0.000295, +0.001570] |
| 16 | qesa association | +0.117220 | [+0.109971, +0.124927] |
| 16 | QHDS without latent aspects | +0.000246 | [-0.000293, +0.000825] |
| 32 | qesa association | +0.154736 | [+0.144241, +0.165322] |
| 32 | QHDS without latent aspects | +0.000098 | [-0.000342, +0.000542] |

该表主要说明 evidence-association feature path 有效；它不能单独分离出 QHDS、QESA、UOT 或 hypergraph 的独立贡献。完整对照和 Gate C 结果以 sparse joint 层为准。

### 5.5 Sparse joint / Gate C

Sparse joint attempt_019 完成全部 25 seed/fold units，使用：

- protected ANN seeds；
- bounded activated-edge rank prefixes；
- deterministic arity-3 semantic candidate construction；
- returned-edge fallback；
- dense QHDS 仅作为标注过的 reference path，不作为 online sparse path。

Gate C 的判定条件是：matched quality improvement、logical Tile 不增加、final top-5 不退化、session-clustered E2E latency interval 为负；payload bytes 单独报告，不能默认声称减少。共有 54 个 qualifying contrasts。

代表性 grid/B：

| Grid/B | Group coverage effect | Clustered CI | Final top-5 effect | Tile effect | E2E latency effect | Holm p |
|---|---:|---|---:|---:|---:|---:|
| 4/8/8/8 | +0.033383 | [+0.027422,+0.035570] | +0.000000 | +0.000000 | -0.631734 | 1.31e-77 |
| 4/8/8/16 | +0.054268 | [+0.046431,+0.057785] | +0.000000 | +0.000000 | -0.631984 | 5.28e-106 |
| 4/16/16/16 | +0.061488 | [+0.051519,+0.064990] | +0.000000 | +0.000000 | -0.680613 | 1.95e-89 |
| 4/32/32/16 | +0.081870 | [+0.071881,+0.085952] | +0.000000 | +0.000000 | -0.721986 | 7.62e-130 |
| 8/8/8/8 | +0.035560 | [+0.029314,+0.037979] | +0.000000 | +0.000000 | -0.696409 | 7.59e-71 |
| 8/8/8/16 | +0.060677 | [+0.051817,+0.064701] | +0.000000 | +0.000000 | -0.696550 | 5.98e-86 |

Gate C 是 OOF-only 的 quality/cost/latency decision，不是 calibration 或 formal test 的通过。

### 5.6 最终 retrieval tier

active16/cap64 的核心结果：

| Budget | Group recall | All groups covered | Payload bytes |
|---:|---:|---:|---:|
| 8 | 0.537294 | 0.474096 | 30,466.0 |
| 16 | 0.609397 | 0.537269 | 60,268.6 |
| 32 | 0.687638 | 0.614022 | 118,696.7 |

active24 在 B32 下：group recall `0.690836`、all-groups-covered `0.617712`；active32 为 `0.694588`、`0.620959`，但延迟显著增加。

final serving decision：

| Tier | Active prototypes | Candidate cap | Budget | Role |
|---|---:|---:|---:|---|
| Default | 16 | 64 | 8 | lowest measured E2E while retaining core quality |
| Quality | 24 | 64 | 8 | small quality gain with latency tradeoff |
| Diagnostic | 32 | 64 | 8 | not promoted due to large latency |

### 5.7 Persona sharding 与 Tile

在 25 OOF units、6,775 query-seed rows 上，Persona shard 返回的 members 与 global prototype rank index 完全一致，因此 retrieval quality、payload 和 final selections 不改变。

| 指标 | Global | Persona x Prototype | 相对变化 |
|---|---:|---:|---:|
| rank entries visited | 50,818.65 | 2,239.04 | -95.59% |
| metadata bytes accessed | 203,274.60 B | 8,956.17 B | -95.59% |
| metadata logical Tiles | 57.668 | 16.001 | -72.25% |
| candidate feature Tiles | 52.577 | 52.577 | 0.00% |
| selected payload Tiles | 6.156 | 6.156 | 0.00% |
| corrected total logical Tiles | 110.276 | 68.609 | -37.78% |
| standalone lookup mean | 1.296 ms | 0.716 ms | -44.70% |

这里的 rank storage size 不变，因为 sharding 是重排/分割，不是复制；feature replication factor 为 0。该结果是 host/logical measurement，不是 PIM measurement。

### 5.8 Matched integrated E2E

active16 的 global/sharded paired control：

| 项目 | 结果 |
|---|---:|
| OOF units | 25 |
| paired rows | 6,775 |
| global query E2E mean | 29.496 ms |
| Persona-sharded query E2E mean | 28.717 ms |
| mean delta | -0.779 ms |
| selections identical | True |

active24 也完成 25/25 matched controls，selection identical。

### 5.9 Static incremental publication

每个增长点执行 256 probes，要求 overlay、compacted lookup 和 B8 selection 与对应完整/发布快照一致：

| Growth | Appended | Deleted | Overlay lookup mean | Compaction | Dispatch mismatch | Compact mismatch |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 420 | 42 | 1.506 ms | 149.569 ms | 0 | 0 |
| 5% | 2,102 | 210 | 1.714 ms | 108.630 ms | 0 | 0 |
| 10% | 4,203 | 420 | 2.023 ms | 109.420 ms | 0 | 0 |
| 20% | 8,407 | 841 | 2.897 ms | 125.604 ms | 0 | 0 |

20% 增长时 active memory count 为 41,193，atomic publish 为 11.461 ms，full snapshot build 为 6,181.377 ms，logical overlay Tiles 为 526。该实验验证静态 publication contract，不验证 causal dynamic behavior。

### 5.10 Oracle headroom

Gold-coverage oracle 仅用于上界，不能作为 deployable selector：

| B | Group coverage | All groups covered | Budget-infeasible |
|---:|---:|---:|---:|
| 8 | 0.999631 | 0.998524 | 0.001476 |
| 16 | 1.000000 | 1.000000 | 0.000000 |
| 32 | 1.000000 | 1.000000 | 0.000000 |

Learned-utility exact oracle 在 cap-12 sparse frontier、B=8、6,775 rows 上 mean/p95 regret 都为 `0.000000`；这只能说明 frozen utility 的 greedy headroom，不能说明最终 retrieval 或硬件优势。

### 5.11 Agent train-OOF

Agent 执行设置：local Qwen、固定 1,800-word context cap、最大 48 generated tokens、1,355 train-OOF queries。所有主要 handoff 变体 answer accuracy 完全相同：

| Variant | Rows | Answer accuracy | Retrieval mean | Generation mean | Total Agent mean |
|---|---:|---:|---:|---:|---:|
| baseline B8 | 1,355 | 0.259041 | 28.637 ms | 3,284.509 ms | 3,313.146 ms |
| baseline B16 | 1,355 | 0.259041 | 28.637 ms | 3,285.134 ms | 3,313.772 ms |
| baseline B32 | 1,355 | 0.259041 | 28.653 ms | 3,283.446 ms | 3,312.099 ms |
| balanced packing B8 | 1,355 | 0.259041 | 28.637 ms | 3,377.947 ms | 3,406.584 ms |
| lexical handoff B8 | 1,355 | 0.259041 | 28.637 ms | 3,285.870 ms | 3,314.508 ms |
| evidence context B8 | 1,355 | 0.259041 | 28.637 ms | 3,455.639 ms | 3,484.276 ms |
| evidence context B32 | 1,355 | 0.259041 | 28.653 ms | 3,261.236 ms | 3,289.889 ms |

平均 prompt tokens 为 `2,538.4`，completion tokens 为 `48.0`。结论是 handoff/payload packing 没有改善最终 answer accuracy；应优先审阅 answer model、prompt protocol、scorer/label semantics 和 evidence-to-answer translation，而不是只扩大 B。

### 5.12 Calibration CRC/LTT

final freeze 后唯一允许的 calibration read 包含 290 queries，固定 grid `(4,8,8)`，selected B=`32`。实际 frontier risk 与 target：

| B | Frontier risk | 95% upper | CRC target | CRC pass | Final top-5 risk | LTT target | LTT after Holm |
|---:|---:|---:|---:|---|---:|---:|---|
| 8 | 0.565517 | 0.614453 | 0.376199 | False | 0.589655 | 0.295062 | False |
| 16 | 0.555172 | 0.604308 | 0.376199 | False | 0.589655 | 0.295062 | False |
| 32 | 0.551724 | 0.600922 | 0.376199 | False | 0.589655 | 0.295062 | False |

校准失败意味着当前 policy 没有通过预先设定的 risk target；它不是一个可以用候选数微调后自动忽略的结果。详见 [`METHOD1_V21_CALIBRATION_CRC_LTT_R1.md`](METHOD1_V21_CALIBRATION_CRC_LTT_R1.md)。

### 5.13 Formal test

正式测试在 final freeze 后只尝试一次：formal query-plan 读到 290 行，但 frozen query embedding index 对 290 个 `med_test/formal_test` query IDs 全部缺失。因此：

```text
status = formal_test_single_run_ineligible_missing_frozen_test_embeddings
formal runs = 1 attempt
valid formal metrics = none
missing frozen test embeddings = 290
```

没有伪造 embedding/label，也没有在 final freeze 后更改模型、selector、hyperparameter 或 dynamic policy。旧协议禁止直接把同一 formal attempt 重跑为第二次；如要继续，必须建立全新、独立授权且预注册的 held-out revision。详见 [`METHOD1_V21_FORMAL_TEST_ONCE_R1.md`](METHOD1_V21_FORMAL_TEST_ONCE_R1.md)。

---

## 6. 当前 closure 与状态解释

### 6.1 工程 closure

以下闭环已被报告和 manifest 记录：

- 25-unit train-OOF retrieval active16/24/32；
- 25-unit matched global/sharded E2E active16/24；
- corrected Tile OOF；
- static new-memory partition publication；
- atomic snapshot serving；
- overlay/compaction equivalence；
- `1,355 x B8/B16/B32` baseline Agent executions；
- balanced/lexical/evidence handoff revisions。

工程 closure 原文：[`METHOD1_V21_AUTHORIZED_FULL_VALIDATION_CLOSURE_R1_20260715.md`](METHOD1_V21_AUTHORIZED_FULL_VALIDATION_CLOSURE_R1_20260715.md)、[`METHOD1_V21_FINAL_DESIGN_CLOSURE_R3_20260715.md`](METHOD1_V21_FINAL_DESIGN_CLOSURE_R3_20260715.md)。

### 6.2 科学 closure

尚未闭合的 claim：

- final Agent answer quality as a generalization guarantee；
- independent formal held-out result；
- causal dynamic access/update behavior；
- LACA/OACH/order-aware memory demand；
- PIM physical result。

### 6.3 “calibration/formal 未读”状态的时间线解释

许多早期 OOF 配置和阶段报告中的 `calibration_read=false`、`formal_test_read=false` 是正确的阶段性状态：它们记录了那些阶段没有读取锁定 split。随后 final closure 阶段发生了：

1. 一次 calibration read（CRC/LTT，290 queries）；
2. 一次 formal access attempt，但因缺少 290 个 test embedding 而不具备有效运行资格。

因此外部审阅时应区分：

- **阶段级状态**：某个 OOF/config 阶段是否读取 locked split；
- **版本累计状态**：整个 v21 lifecycle 是否发生过 calibration/formal access。

当前版本累计事实是“calibration read 1 次、formal attempt 1 次且无有效 formal metrics”。

---

## 7. 可复现与审计状态

### 7.1 已验证的本地单元测试

使用系统 Python（因为 `agent-memory-profiling` Conda 环境当前没有安装 pytest）执行：

```text
python -m pytest -q versions/20260714_method1_fsa_dynamic_hypergraph_v21/tests
73 passed, 1 warning, 2.17s
```

warning 是一个 Tensor requires-grad 时直接转 float 的 PyTorch warning，不影响这 73 个测试的通过结果。正式复现前应把 pytest 和依赖版本补入环境锁定文件；不能仅依赖 `.pytest_cache` 作为测试证据。

### 7.2 运行/硬件状态

- 审计时没有发现 `run_method1_v03.py`、v21 sparse-joint、QESA、QHDS 或其他 Method 1 launcher 进程。
- `nvidia-smi` 在审计过程中出现过间歇性 driver communication failure；当前没有运行中的方法一进程，因此没有正在消耗 GPU 的 v21 job。
- v21 版本目录约 31 GB、3,838 files、701 manifest；未发现零字节文件；磁盘剩余约 4.1 TB。

### 7.3 主要机器可读产物

| 产物 | 用途 |
|---|---|
| `aggregate/v21_manifest.json` | 初始化版本和输入 hash |
| `aggregate/v21_manifest_phase1_r1.json` | Phase 1 manifest |
| `aggregate/v21_uot_fsa_oof_summary.json` | Gate A summary |
| `aggregate/v21_qesa_oof_summary_r2.json` | QESA r2 summary |
| `aggregate/v21_qesa_gate_b2_summary.json` | Gate B2 summary |
| `aggregate/v21_qhds_fixed_budget_summary.json` | QHDS fixed-budget summary |
| `aggregate/v21_sparse_joint_core_summary_r3_attempt019.json` | Gate C sparse-joint core input |
| `aggregate/v21_gate_c_decision_r1_20260714.json` | Gate C decision |
| `aggregate/v21_persona_sharded_hyper_index_summary_cap64_b8_r1_attempt001.json` | Persona sharding/Tile result |
| `aggregate/v21_persona_sharded_partition_update_stress_r1_attempt001.json` | static update stress |
| `aggregate/v21_calibration_crc_ltt_r1_20260714.json` | calibration outcome |
| `aggregate/v21_formal_test_once_r1_20260714.json` | formal attempt outcome |
| `aggregate/v21_authorized_full_validation_closure_r1_attempt001.json` | authorized engineering closure |
| `aggregate/v21_method1_final_design_closure_r3_attempt001.json` | final design closure r3 |

---

## 8. 给外部 GPT 的审阅任务

请不要只问“方法是否复杂”，而应围绕以下问题给出可证伪的优化建议：

### 8.1 科学问题与主张

1. 当前最稳健的论文主张是否应收缩为“train-only learned fuzzy semantic association + bounded evidence selection + exact-equivalent Persona sharding”，而不是泛化的“dynamic Agent memory”或“hypergraph superiority”？
2. Gate B r1/r2 失败、B2 representation 通过、Gate C 通过之间的逻辑关系是否清楚？哪些组件的独立贡献仍未被识别？
3. `group coverage`、`all-groups-covered` 和最终 answer accuracy 之间是否存在明显的 surrogate mismatch？应增加哪些中间指标或 causal ablations？

### 8.2 算法优化

1. QESA r2 的 representation 已有正向效果，但 direct retrieval 仍为负；应优先修改 query-to-prototype calibration、prototype diversity、activation policy，还是直接移除 QESA 并改用廉价的 query-conditioned association？
2. learned bundle locality 的 all-groups-covered 负结果和 cardinality exact rate 25.43% 暗示什么？应采用 set prediction、coverage-aware beam、动态 arity，还是停止 bundle scorer 方向？
3. Gate C 的质量收益是否主要来自 semantic association/candidate pool，而不是 QHDS 或 hypergraph 本身？如何设计 matched ablation 来分离 UOT、QESA、bundle、QHDS、Persona shard 五个贡献？
4. 在 active16/24/32 的延迟-质量曲线上，如何选择适合生产的 adaptive policy，而不违反 calibration failure 和 frozen policy 约束？
5. Persona sharding 已经降低 rank scan 95.59%，下一步是否应优化全量 snapshot rebuild（20% growth 为 6.18 s），例如 persona-local cache、deferred compaction 或增量 rank merge？

### 8.3 Agent 与评估

1. 为什么 B8/B16/B32、balanced、lexical、evidence handoff 的 answer accuracy 都完全固定为 0.259041？最可能的瓶颈是 scorer、prompt、answer model、context truncation、证据格式还是 query label semantics？
2. 在保持 retrieval 固定的条件下，应优先做哪些 answer-generation-only diagnostics，才能判断 retrieval quality 是否被 Agent model 掩盖？
3. 是否需要增加 per-query failure taxonomy：missing evidence、wrong evidence、context conflict、generation refusal、format/scorer failure？
4. 下一版 held-out protocol 如何确保 query embedding 完整存在，并在 formal run 前自动阻断 `missing=290` 这类输入不完整问题？

### 8.4 动态与硬件边界

1. 在没有 item-level read event 的当前数据上，静态 append publication 是否足以作为独立系统论文贡献？如果不足，需要采集什么最小真实 Agent read/co-access trace？
2. 哪些 host/logical Tile 指标最适合交给方法二做 hardware calibration，哪些指标不应再包装成硬件结论？
3. 是否需要把 PIM 相关内容完全移到方法二，并在方法一只交付 per-query tile/row/operation trace contract？

---

## 9. 建议的后续实验优先级

### P0：先修复评估闭环

1. 新建独立 held-out revision，冻结并 hash 完整 test query embedding index；formal eligibility sentinel 必须在运行前确认 `present=expected`。
2. 不覆盖旧 formal attempt，不伪造 embedding，不更改旧 split；新 revision 要有新的 run ID、配置、manifest 和 pre-registration。
3. 保留当前 Gate B negative、bundle negative、CRC/LTT failure 作为 immutable negative results。

### P1：拆解 Agent answer bottleneck

1. 固定同一 retrieval context，替换/对照 answer model、prompt、context serialization 和 scorer。
2. 输出逐 query error taxonomy；分别报告 evidence recall、answer correctness、abstention/format correctness。
3. 先在 train-OOF 完成 answer protocol 的 sanity gate，再授权新的 calibration/formal。

### P2：做组件独立性与高阶结构审计

1. 完成 matched pairwise factor graph、flat cosine、dense QHDS、UOT-only、QESA-only、bundle-only、QHDS-only 的同预算/同候选/同 logical Tile 对照。
2. 明确 Gate C 的 54 个 qualifying contrasts 是否在多个 seeds/folds、固定 bytes、iso-quality 和 latency Pareto 上稳定。
3. 如果高阶 bundle 不能改善 all-groups-covered，应把主张收缩为 fuzzy semantic association + bounded selector，而不是坚持“高阶 hypergraph superiority”。

### P3：优化静态维护与真实动态采集

1. 针对 6.18 s full snapshot build 做 persona-local rank cache、deferred compaction、batched overlay merge 的 ablation。
2. 若要保留 dynamic claim，采集独立 item-level Agent read event；否则正式文档固定使用 static incremental publication 术语。

---

## 10. 审阅者应采用的最终判定

建议将结论拆成四个独立 verdict，而不要给方法一一个笼统的“成功/失败”：

| Verdict | 当前判定 |
|---|---|
| Static semantic retrieval/index engineering | **通过** |
| Fixed-budget quality-cost Gate C | **通过（OOF-only）** |
| Independent formal generalization | **未判定/无资格** |
| End-to-end Agent effectiveness | **负结果，未闭合** |
| Causal dynamic access/update | **数据不具备资格** |
| PIM hardware benefit | **未测量** |

最终建议的论文/报告措辞：

> v21 demonstrates a train-only fuzzy semantic association and bounded evidence-selection system with exact-equivalent Persona-sharded lookup and static incremental publication. It does not yet establish independent held-out generalization, improved Agent answer quality, causal dynamic access behavior, or PIM hardware acceleration.

