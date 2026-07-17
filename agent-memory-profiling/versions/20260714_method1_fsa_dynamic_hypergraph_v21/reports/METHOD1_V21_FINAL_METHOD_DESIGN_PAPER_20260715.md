# 方法一：基于模糊语义超图与 Persona 分片索引的可增量 Agent Memory 检索

## 1. 方法定位

方法一面向长期 Agent Memory 场景：系统需要从规模不断增长、语义高度重叠、同一事实可能存在多个可接受表达的历史记忆中，为当前查询选择一个规模受控、覆盖充分且可直接交给下游 Agent 的证据集合。

最终方法将整个流程实现为一个闭环：

```text
离线记忆
  -> 冻结语义嵌入
  -> 残差低秩投影
  -> UOT 模糊语义关联
  -> Persona × Prototype 排名分片

在线查询
  -> 查询嵌入
  -> 受保护 ANN Top-5
  -> QESA 查询条件 Prototype 激活
  -> Persona 分片上的有界排名前缀读取
  -> 三元语义 Bundle Beam
  -> Cap-64 候选前沿
  -> QHDS 单调次模选择
  -> B=8 单副本 Payload Gather
  -> Agent Handoff

新记忆
  -> 冻结投影与 Prototype Incidence
  -> Persona × Prototype Overlay
  -> 不可变 Snapshot
  -> 原子发布
  -> 周期性 Compaction
```

该设计同时解决四个问题：

1. 用模糊 Prototype Incidence 表达一条记忆同时属于多个语义区域的事实。
2. 用语义 Bundle 显式表示组合证据，而不是只对单条记忆独立打分。
3. 用 Persona × Prototype 排名分片缩短在线排名前缀扫描，同时保持检索结果严格等价。
4. 用 Overlay、Snapshot 和 Compaction 支持不依赖时间戳的新记忆增量发布。

---

## 2. 问题定义与符号

设长期记忆集合为

$$
\mathcal{M}=\{m_i\}_{i=1}^{N}, \qquad N=42{,}034.
$$

每条记忆包含：

- 唯一标识 $id_i$；
- Persona 标识 $u_i$；
- 冻结语义嵌入 $x_i\in\mathbb{R}^{d}$，其中 $d=768$；
- 文本 Payload 与结构化 memory metadata；
- 一个由 UOT 得到的模糊 Prototype Incidence 向量 $H_i\in\mathbb{R}_{+}^{K}$。

最终系统使用 $K=64$ 个语义 Prototype。对查询 $q$，其冻结嵌入为 $x_q$。系统还接收一个合法记忆集合 $\mathcal{E}_q\subseteq\mathcal{M}$，用于实施 Persona、会话边界和可见性约束。

对于评估查询，证据标签表示为若干 OR 语义证据组：

$$
\mathcal{G}_q=\{G_{q,1},\ldots,G_{q,L_q}\}.
$$

每个 $G_{q,j}$ 包含一个或多个可接受记忆，只需选中组内任意一个成员即可覆盖该证据组。因此，最终优化目标不是简单地命中唯一 gold memory，而是尽可能覆盖更多证据组，并在固定预算内兼顾相关性、Prototype 覆盖与证据多样性。

主要符号如下。

| 符号 | 含义 |
|---|---|
| $x_i,x_q$ | 记忆与查询的冻结嵌入 |
| $g_\theta(\cdot)$ | 残差低秩语义投影 |
| $p_k$ | 第 $k$ 个 UOT 语义 Prototype |
| $H_{ik}$ | 记忆 $i$ 对 Prototype $k$ 的模糊 Incidence |
| $\pi_q(k)$ | QESA 预测的查询对 Prototype $k$ 的概率 |
| $A_q$ | 查询激活的 Prototype 集合 |
| $E_{q,k}$ | Prototype $k$ 返回的有界语义边成员 |
| $\mathcal{B}_q$ | 查询的三元语义 Bundle 集合 |
| $\mathcal{C}_q$ | Cap-64 候选前沿 |
| $S_q$ | QHDS 选出的最终记忆集合 |

---

## 3. 设计原则

最终系统遵循以下原则。

### 3.1 冻结基础表征，学习小规模适配层

系统不重新训练基础嵌入模型，而是在冻结嵌入上训练残差低秩投影。这样既保留通用语义空间，又允许记忆证据组在任务空间中更容易共置。

### 3.2 在线阶段只访问有界稀疏结构

在线查询禁止扫描完整 Incidence 矩阵。查询只能读取：

- 受保护 ANN 前缀；
- 激活 Prototype 对应的排名分片；
- 已被这些有界读取返回的 edge members。

候选前沿不足时，只能从已经返回的 edge members 中补齐，不能扩展排名前缀，也不能回退到 dense scan。

### 3.3 语义结构与物理组织分离

语义超图负责候选发现和组合证据建模；Payload 保持 capacity-16 的中性单副本布局。系统不为了训练超边复制或强制重排 Payload，从而避免存储膨胀和 OOF 泛化风险。

### 3.4 检索等价性优先

Persona 分片和增量 Overlay 都必须满足严格的结果等价性：在相同查询、相同合法集合和相同 Prototype 激活下，返回的 memory IDs、Incidence scores 和最终选择必须与对应的完整索引或 compacted snapshot 一致。

---

## 4. 离线语义记忆构建

### 4.1 冻结语义嵌入

每条记忆首先由冻结编码器映射为 $x_i\in\mathbb{R}^{768}$，随后执行 $\ell_2$ 归一化。查询嵌入使用同一基础语义空间，因此基础相关性可以直接写成余弦相似度：

$$
s_{\mathrm{cos}}(q,i)=x_q^\top x_i.
$$

冻结基础嵌入的原因是保证不同 OOF fold、不同索引 revision 和增量发布阶段共享同一个稳定的语义坐标系。

### 4.2 残差低秩投影

基础嵌入经过一个残差低秩适配器：

$$
g_\theta(x)=\operatorname{norm}\left(x+W_{\mathrm{up}}\,\operatorname{GELU}(W_{\mathrm{down}}x)\right).
$$

其中 $W_{\mathrm{down}}\in\mathbb{R}^{r\times d}$，$W_{\mathrm{up}}\in\mathbb{R}^{d\times r}$，$r\in\{64,128\}$。上投影矩阵初始化为零，因此训练开始时 $g_\theta(x)$ 与原始嵌入一致，随后只学习任务需要的残差修正。

适配器使用 multi-positive InfoNCE 训练。对于一个 anchor $a$、候选集合 $\mathcal{D}$ 和可接受正例集合 $\mathcal{P}(a)$，损失为

$$
\mathcal{L}_{\mathrm{MP}}(a)
=-\log
\frac{\sum_{p\in\mathcal{P}(a)}\exp(g(a)^\top g(p)/T)}
{\sum_{z\in\mathcal{D}}\exp(g(a)^\top g(z)/T)}.
$$

这里使用多正例分子，是因为同一证据组中可能存在多个可接受记忆。训练 anchor 同时包括 query embedding 和多成员证据组中的 memory embedding；hard negative 从冻结嵌入的近邻非正例中产生。

### 4.3 UOT 模糊语义 Prototype

投影后的记忆向量记为 $z_i=g_\theta(x_i)$。系统用球面 farthest-point sampling 初始化 $K=64$ 个 Prototype，并采用余弦距离定义传输代价：

$$
C_{ik}=1-z_i^\top p_k.
$$

为吸收无法被当前 Prototype 充分解释的质量，Prototype 侧增加一个 dustbin，其先验质量为 $0.05$。UOT 求解以下带 KL 松弛的目标：

$$
\min_{T\ge 0}
\langle T,C\rangle
+\epsilon\,\mathrm{KL}(T\|ab^\top)
+\tau_m\,\mathrm{KL}(T\mathbf{1}\|a)
+\tau_p\,\mathrm{KL}(T^\top\mathbf{1}\|b).
$$

其中 $a$ 是均匀 memory prior，$b$ 是 Prototype 与 dustbin prior。与 balanced OT 不同，$\tau_m$ 和 $\tau_p$ 允许实际行列质量偏离先验，因此更适合记忆密度不均和 Prototype 使用频率不平衡的长期记忆集合。

实现使用 log-domain unbalanced Sinkhorn。定义

$$
\rho_m=\frac{\tau_m}{\tau_m+\epsilon},\qquad
\rho_p=\frac{\tau_p}{\tau_p+\epsilon},
$$

则缩放势更新为

$$
\log u_i
=\rho_m\left(\log a_i-\operatorname{LSE}_k\left(-C_{ik}/\epsilon+\log v_k\right)\right),
$$

$$
\log v_k
=\rho_p\left(\log b_k-\operatorname{LSE}_i\left(-C_{ik}/\epsilon+\log u_i\right)\right).
$$

传输计划为

$$
T_{ik}=\exp\left(-C_{ik}/\epsilon+\log u_i+\log v_k\right).
$$

每次外循环后，用语义部分的传输质量更新 Prototype：

$$
p_k\leftarrow
\operatorname{norm}\left(
\frac{\sum_iT_{ik}z_i}{\sum_iT_{ik}}
\right).
$$

可选的 log-determinant proximal step 用于约束 Prototype 多样性。最终系统对 dustbin 之外的语义 transport logits 做行归一化，得到模糊 Incidence：

$$
H_{ik}=\operatorname{softmax}_k
\left(-C_{ik}/\epsilon+\log u_i+\log v_k\right),
\qquad \sum_{k=1}^{K}H_{ik}=1.
$$

与硬聚类不同，$H_i$ 保留了多 Prototype 隶属关系。一条记忆可以同时对多个语义区域具有非零贡献，从而支持组合查询和多证据覆盖。

### 4.4 Prototype 排名索引

对于每个 Prototype $k$，离线构造按以下键排序的 memory rank list：

$$
R_k=\operatorname{sort}_i(-H_{ik},i).
$$

第二排序键使用稳定 memory position，确保相同 Incidence 下结果完全确定。排名条目存储为 32-bit memory position。

---

## 5. Persona × Prototype 排名分片

### 5.1 分片结构

最终索引不是一个覆盖全局 memory corpus 的 $R_k$，而是对每个 Persona $u$ 和 Prototype $k$ 建立：

$$
R_{u,k}
=\operatorname{sort}_{i:u_i=u}(-H_{ik},i).
$$

所有 Persona shard 的总条目数仍为 $N\times K$，因为每条 memory 对每个 Prototype 只出现在其所属 Persona 的一个 shard 中。因此分片不会引入 rank storage replication：

$$
\sum_u|R_{u,k}|=N.
$$

### 5.2 有界排名前缀读取

查询只访问当前 Persona 对应的 shard。对激活 Prototype $k$，系统从 $R_{u,k}$ 开始顺序读取，并用合法集合 $\mathcal{E}_q$ 做 membership test，直到收集到 32 个合法成员或 shard 结束：

$$
E_{q,k}=\operatorname{FirstEligible}_{32}(R_{u_q,k},\mathcal{E}_q).
$$

读取过程采用块式前缀消费，并记录精确的 rank entries visited。它不根据标签扩展前缀，也不访问完整 Incidence 矩阵。

### 5.3 为什么 Persona 分片能保持严格等价

查询的合法记忆均属于同一 Persona。全局 rank list 中其他 Persona 的条目只会被 eligibility filter 丢弃。将这些必然被丢弃的条目离线移除，不会改变同一 Persona 内的相对排序，因此：

$$
\operatorname{FirstEligible}_{L}(R_k,\mathcal{E}_q)
=\operatorname{FirstEligible}_{L}(R_{u_q,k},\mathcal{E}_q).
$$

这使 Persona sharding 成为结果等价的物理索引优化，而不是新的近似检索方法。

---

## 6. 查询条件 Prototype 激活：QESA

### 6.1 查询到 Latent Aspect

QESA 使用一个两层 MLP 将查询嵌入映射为 latent aspect 分布：

$$
\alpha_q
=\operatorname{softmax}\left(W_2\operatorname{GELU}(W_1x_q)\right).
$$

最终模型使用 32 个 latent aspects，hidden dimension 为 128。

### 6.2 Latent Aspect 到 Prototype

模型学习一个 aspect-to-prototype logit 矩阵 $L\in\mathbb{R}^{A\times K}$：

$$
M_{ak}=\operatorname{softmax}_k(L_{ak}/T_a).
$$

查询的 Prototype 概率为

$$
\pi_q=\alpha_qM.
$$

最终 anti-collapse 配置使用 $T_a=0.1$。训练损失同时包含：

$$
\mathcal{L}_{\mathrm{QESA}}
=\lambda_{\mathrm{cov}}\mathcal{L}_{\mathrm{coverage}}
+\lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{target}}
+\lambda_{\mathrm{div}}\mathcal{L}_{\mathrm{aspect-diversity}}
+\lambda_{\mathrm{ent}}\mathcal{L}_{\mathrm{map-entropy}}
+\lambda_{\mathrm{use}}\mathcal{L}_{\mathrm{usage-KL}}.
$$

其中 coverage loss 使用证据组 OR 语义。对证据组 $G$，先取组内可接受成员对每个 Prototype 的最大 Incidence：

$$
a_G(k)=\max_{i\in G}H_{ik},
$$

再最大化查询 Prototype 分布与 $a_G$ 的重合。这样，QESA 学习的是“当前查询应该激活哪些语义区域”，而不是预测某一个唯一 memory ID。

### 6.3 激活策略

默认在线配置取

$$
A_q=\operatorname{TopK}(\pi_q,16).
$$

系统同时保留 active24 质量档。论文中的默认方法使用 active16，因为它在当前质量下具有更低的完整在线时延；active24 可作为面向质量优先场景的配置。

---

## 7. 受保护 ANN Seed

系统从冻结 ANN 结果中保留前 5 个合法 memory，记为 $P_q$。这些位置在后续候选构造和 QHDS 选择中始终被保护。

受保护 seed 有两个作用：

1. 保留稳定的局部语义近邻作为检索下界。
2. 防止 Prototype 路径在查询分布偏移时完全丢失直接相似的证据。

因此最终选择预算必须满足 $B\ge |P_q|=5$。

---

## 8. 三元语义 Bundle Beam

### 8.1 从语义边构造高阶证据

每个激活 Prototype 及其有界成员构成一条语义边：

$$
e_{q,k}=(k,E_{q,k}).
$$

系统从不同激活边中构造 arity-3 Bundle：

$$
b=((k_1,i_1),(k_2,i_2),(k_3,i_3)),
$$

要求 $k_1,k_2,k_3$ 互不相同，$i_1,i_2,i_3$ 也互不相同。该约束避免同一条记忆重复承担多个证据角色。

### 8.2 Bundle 分数

对成员 $i$ 和 Prototype $k$，定义局部分数：

$$
\phi(q,i,k)
=s_{\mathrm{cos}}(q,i)+\pi_q(k)H_{ik}.
$$

Bundle 分数为其三个局部分数之和：

$$
s_{\mathrm{bundle}}(b)
=\sum_{t=1}^{3}\phi(q,i_t,k_t).
$$

第一项保证成员与查询直接相关，第二项要求该成员对当前查询激活的 Prototype 具有较强 Incidence。

### 8.3 有界 Beam 实现

为控制组合爆炸，最终实现采用以下边界：

- 只保留 Prototype 权重和最高的前 8 个三边组合；
- 每条边只取 Incidence 最高的前 4 个成员参与 Bundle 扩展；
- 每一步保留前 32 个 partial bundles；
- 最终对成员集合去重并保留前 32 个完整 Bundle。

因此 Bundle 构造复杂度由固定 beam 控制，不随完整 memory corpus 直接增长。

---

## 9. Cap-64 候选前沿

最终候选前沿按以下确定性顺序构造：

1. 加入受保护 ANN Top-5；
2. 按 Bundle 分数顺序加入 retained bundles 的唯一成员；
3. 若尚未达到候选上限，则只从已经返回的 edge members 中补齐；
4. 补齐成员按 query-memory cosine 降序、memory position 升序排序；
5. 候选总数限制为 64。

形式化地：

$$
\mathcal{C}_q
=\operatorname{Cap}_{64}\left(
P_q
+\operatorname{UniqueMembers}(\mathcal{B}_q)
 +\operatorname{SortedFallback}\left(\bigcup_{k\in A_q}E_{q,k}\right)
\right).
$$

关键约束是 fallback 只能使用本次有界 sparse lookup 已经返回的成员。该设计保证：

- 候选数量足以支持最大选择预算 $B=32$；
- 在线阶段不进行 prefix extension；
- 不进行 dense Incidence scan；
- 所有比较方法共享完全相同的候选前沿。

---

## 10. QHDS 单调次模选择

### 10.1 候选特征

对候选 memory $i\in\mathcal{C}_q$，构造三个特征：

$$
c_i=x_q^\top x_i,
$$

$$
a_i=H_i^\top\pi_q,
$$

$$
f_i=[c_i,a_i,c_i a_i].
$$

$c_i$ 表示直接语义相似度，$a_i$ 表示候选与查询激活语义分布的关联强度，乘积项刻画两者的一致性。

### 10.2 非负相关性

相关性网络为一个 hidden dimension 16 的 MLP，并用 Softplus 保证输出为正：

$$
r_i=\operatorname{softplus}(\operatorname{MLP}(f_i))+10^{-8}.
$$

非负相关性是保证后续 utility 单调性的必要条件。

### 10.3 次模 Utility

对已选集合 $S$，定义每个 Prototype 的累计覆盖：

$$
z_k(S)=\sum_{i\in S}H_{ik}r_i.
$$

QHDS utility 为

$$
U(S)
=\lambda_s\sum_{k=1}^{K}\pi_q(k)\log(1+z_k(S))
+\lambda_m\sum_{i\in S}r_i,
$$

其中

$$
\lambda_s=\operatorname{softplus}(\eta_s),\qquad
\lambda_m=\operatorname{softplus}(\eta_m).
$$

第一项是结构覆盖项。$\log(1+z)$ 具有边际递减性质，因此重复选择同一 Prototype 上的相似 memory 收益会逐渐下降。第二项是 modular relevance，用于避免系统为了多样性选入低相关候选。

候选 $i$ 的边际增益为

$$
\Delta(i\mid S)
=\lambda_s\sum_k\pi_q(k)
\left[
\log(1+z_k(S)+H_{ik}r_i)-\log(1+z_k(S))
\right]
+\lambda_mr_i.
$$

### 10.4 受保护贪心选择

选择从 $S=P_q$ 开始，每一步加入边际增益最大的未选候选：

$$
i^*=\arg\max_{i\in\mathcal{C}_q\setminus S}\Delta(i\mid S).
$$

一次贪心递推生成一个最大长度为 32 的嵌套顺序，其前缀直接对应：

$$
S_q^{(8)}\subseteq S_q^{(16)}\subseteq S_q^{(32)}.
$$

默认 Agent handoff 使用 $B=8$；$B=16$ 和 $B=32$ 用于质量曲线、离线诊断或更大上下文预算场景。

---

## 11. 单副本 Feature 与 Payload Gather

最终设计不复制 feature 或 Payload。所有 memory 均保留在 capacity-16 的中性单副本 memory Tile 中。

在线路径先使用 rank metadata 与冻结 embedding/incidence feature 完成选择，只对最终集合 $S_q^{(B)}$ 执行 Payload Gather：

$$
\operatorname{Payload}(q)
=\{\operatorname{text}(m_i):i\in S_q^{(B)}\}.
$$

默认 $B=8$，因此 Agent 最多接收 8 条最终 memory。检索方法本身在 Payload Gather 前完成，因而下游 Agent 可以替换，而不改变索引、候选发现和 QHDS 选择契约。

### 11.1 确定性上下文打包

Payload 按 QHDS 的贪心选择顺序写入上下文，每条 memory 使用如下结构：

```text
[memory_id | session_id | session_type]
memory text
```

打包器从第一条 memory 开始顺序消费文本，达到 1,800-word 上限后立即停止；最后一条只保留仍可容纳的前缀。这样，同一选择序列总能生成同一上下文，且上下文成本不会随 $B$ 无界增长。`memory_id`、session 和 type 只承担证据边界标记，不参与 QHDS 打分。

### 11.2 Agent 执行契约

Agent 接收 query type、当前 session、query text 和打包后的 memory context。执行提示要求模型：

1. 只使用提供的 memory context；
2. 不调用外部知识或工具；
3. 证据不足时返回固定的 `insufficient information`；
4. 只输出一行简短最终答案。

闭环验证使用本地 Qwen，最大生成长度为 48 tokens，并分别记录 prompt tokens、completion tokens、生成时延和“检索 + 生成”总时延。这个接口把检索质量、上下文成本和最终回答执行放在同一条可测链路上，同时保持 Agent backend 可替换。

---

## 12. Host/Logical Tile 组织与计数

Tile 只表示 host 侧的逻辑访问单元，不表示 PIM 硬件测量。

### 12.1 Rank Metadata Tile

每个 rank entry 为 4 bytes，每个 metadata Tile 为 4 KiB，因此一个连续 rank block 包含：

$$
4096/4=1024
$$

个 rank entries。对每条激活边，根据实际访问的连续排名前缀计算 metadata Tile：

$$
T_{\mathrm{rank}}(q,k)
=\left\lceil\frac{V_{q,k}}{1024}\right\rceil,
$$

其中 $V_{q,k}$ 是找到 32 个合法成员前实际访问的 rank entries 数量。

### 12.2 Feature 与 Payload Tile

Feature 和 Payload 都使用 capacity-16 的 neutral single-copy memory Tile。最终系统不使用 feature replication：

$$
\text{feature replication factor}=0.
$$

查询的 corrected total logical Tiles 是以下集合的并集大小：

$$
T_{\mathrm{total}}(q)
=\left|
\mathcal{T}_{\mathrm{rank}}(q)
\cup\mathcal{T}_{\mathrm{feature}}(q)
\cup\mathcal{T}_{\mathrm{payload}}(q)
\right|.
$$

这种口径将实际连续 rank block、候选 feature 所在 memory Tile 和最终 Payload Tile 分开计数，避免将不同数据结构混为一个不可解释的 Tile 指标。

---

## 13. 新记忆增量发布

### 13.1 更新接口

新记忆在进入增量索引前，需要得到：

- memory ID；
- Persona ID；
- 冻结 embedding；
- 冻结投影后的 Prototype Incidence $H_i$；
- Payload 与 neutral Tile assignment。

增量索引提供：

```text
upsert(memory_id, persona_id, prototype_incidence)
delete(memory_id)
snapshot()
publish(snapshot)
compact()
```

冻结 base memory 不能被原地替换。如需修改，应 tombstone 旧 ID，并以新 ID append。

### 13.2 Persona × Prototype Overlay

Overlay 为每个新 memory 保存 $(u_i,H_i)$，并对每个 $(u,k)$ 维护按 $(-H_{ik},id_i)$ 排序的 overlay rank stream。

查询时，base rank stream 与 overlay rank stream 使用双指针 merge。每一步比较两侧当前元素：

$$
(-H^{\mathrm{base}}_{ik},id_i)
\quad\text{和}\quad
(-H^{\mathrm{overlay}}_{jk},id_j),
$$

选择排序键更小的一侧，并跳过 tombstone 或不在合法集合中的 memory。该过程一直运行到收集够 32 个成员。

因此增量查询在不重建完整 base index 的情况下，仍能返回与“将新 memory 完全并入后重新排序”相同的有界前缀。

### 13.3 不可变 Snapshot 与原子发布

Mutable overlay 只负责 staging。查询只能读取显式发布的 immutable snapshot。每个 snapshot 带有单调递增版本号：

$$
v_{t+1}>v_t.
$$

`AtomicPersonaShardedSnapshotPublisher` 使用锁保护 snapshot 指针替换，使每个查询只观察一个完整版本，不会读到半更新状态。

### 13.4 Tombstone 与 Compaction

删除通过 tombstone 实现，不立即重写 base rank list。周期性 compaction 执行：

1. 移除被 tombstone 标记的 base/overlay memory；
2. 将有效 overlay memory 合并到新的 base corpus；
3. 重新生成不可变 Persona × Prototype rank streams；
4. 发布新的 compacted snapshot。

### 13.5 不依赖时间戳的更新验证协议

为验证增量机制而不引入伪造时间顺序，使用

$$
\operatorname{SHA256}(memory\_id)
$$

对现有 memory 做确定性分区：80% 作为 frozen base，20% 作为 new-memory pool。随后按全量 corpus 的 1%、5%、10%、20% 逐步发布新 memory，并按追加规模的 10% 注入删除。

该协议验证的是“静态新记忆分区下的增量发布与维护正确性”，不需要时间戳，也不推断 causal access/update behavior。

---

## 14. 完整在线算法

### Algorithm 1：离线构建

```text
Input: memory corpus M, frozen embeddings X
Output: projection g, prototypes P, incidence H, persona rank shards R

1. Train residual projection g with multi-positive InfoNCE on outer-train folds.
2. Project and normalize all memory embeddings.
3. Initialize 64 spherical prototypes with farthest-point sampling.
4. Fit KL-UOT with a dustbin and update prototypes from transport mass.
5. Normalize semantic transport logits into fuzzy incidence H.
6. For each persona u and prototype k:
       sort memories of u by (-H[i,k], memory_position)
       store the resulting int32 rank shard R[u,k].
7. Train QESA and QHDS only on outer-train data.
```

### Algorithm 2：在线检索

```text
Input: query q, persona u, legal memory set E_q
Output: selected memories S_q

1. Encode q and obtain protected ANN Top-5 P_q.
2. QESA predicts prototype distribution pi_q.
3. Activate Top-16 prototypes A_q.
4. For each k in A_q:
       scan only R[u,k]
       retain the first 32 members that belong to E_q.
5. Build arity-3 semantic bundles with beam width 32.
6. Construct C_q:
       protected seeds
       + unique bundle members
       + already-returned edge-member fallback
       capped at 64.
7. Compute [cosine, incidence association, product] for each candidate.
8. Run protected QHDS greedy recurrence through budget 32.
9. Return the first B memories; default B=8.
10. Gather only the selected single-copy payloads and hand them to the Agent.
```

### Algorithm 3：增量发布

```text
Input: staged appends A, tombstones D, current snapshot S_v
Output: published snapshot S_(v+1)

1. For each new memory, compute frozen embedding/projection/incidence.
2. Insert (memory_id, persona_id, incidence) into overlay.
3. Add deleted IDs to the tombstone set.
4. Build immutable persona/prototype overlay rank streams.
5. Verify snapshot version v+1 > v.
6. Atomically replace the published snapshot pointer.
7. Queries merge base and overlay rank streams from one snapshot version.
8. Compact periodically; do not block every append on full rebuild.
```

---

## 15. 复杂度分析

### 15.1 存储

完整 fuzzy rank metadata 的存储复杂度为

$$
O(NK).
$$

Persona 分片只将全局 rank list 分割为多个 shard，不复制条目，因此总 metadata bytes 不变。Payload 与 feature 都是单副本 $O(N)$。

### 15.2 在线 Sparse Lookup

若查询激活 $K_q=16$ 个 Prototype，第 $k$ 条边实际访问 $V_{q,k}$ 个 rank entries，则 lookup 为

$$
O\left(\sum_{k\in A_q}V_{q,k}\right),
$$

而不是 $O(NK)$。Persona 分片的作用就是显著降低 $V_{q,k}$。

### 15.3 Bundle Beam

由于 Prototype tuple 前缀、单边成员前缀和 beam width 都是常数，Bundle 构造对 corpus 大小是有界的。默认 arity 为 3，beam width 为 32。

### 15.4 QHDS

候选数 $C\le64$、最大预算 $B\le32$、Prototype 数 $K=64$。向量化贪心递推约为

$$
O(BCK),
$$

且一次递推同时产生 B8、B16 和 B32 的嵌套结果。

### 15.5 增量更新

单次 upsert 只写入 overlay，查询通过 base/overlay merge 消化更新。完整 compaction 是周期性维护操作，不位于每次写入的关键路径。

---

## 16. 闭环实现配置

最终默认配置如下。

| 组件 | 最终配置 |
|---|---|
| Memory 数量 | 42,034 |
| Embedding dimension | 768 |
| UOT Prototype 数 | 64 |
| QESA latent aspects | 32 |
| QESA hidden dimension | 128 |
| 默认激活 Prototype | 16 |
| 质量档激活 Prototype | 24 |
| 每条语义边返回成员 | 32 |
| Bundle arity | 3 |
| Bundle beam width | 32 |
| 受保护 ANN seeds | 5 |
| Candidate frontier cap | 64 |
| QHDS hidden dimension | 16 |
| 选择预算 | 8 / 16 / 32，默认 8 |
| Rank entry | 4 bytes |
| Metadata logical Tile | 4 KiB |
| Feature/Payload Tile capacity | 16 memories |
| Feature/Payload replication | 0，单副本 |

---

## 17. 实现闭环与验证结果

### 17.1 检索质量

默认 active16、cap64 路径在 5 seeds × 5 session-blocked folds 上得到：

| Budget | Group recall | All groups covered | Payload bytes |
|---:|---:|---:|---:|
| 8 | 0.537294 | 0.474096 | 30,466.0 |
| 16 | 0.609397 | 0.537269 | 60,268.6 |
| 32 | 0.687638 | 0.614022 | 118,696.7 |

这里 B32 用于报告完整检索能力，B8 是默认 Agent handoff 预算。

最终系统还提供 active24 质量档。在 B32 口径下，其 group recall 为 0.690836、all groups covered 为 0.617712；默认 active16 的对应结果为 0.687638 和 0.614022。active24 用更多 Prototype 读取换取小幅质量提升，因此作为显式可选档，而不是替代低时延默认档。

### 17.2 Persona 分片效果

| 指标 | Global rank index | Persona × Prototype shards | 变化 |
|---|---:|---:|---:|
| Rank entries visited | 50,818.65 | 2,239.04 | -95.59% |
| Metadata bytes accessed | 203,274.60 B | 8,956.17 B | -95.59% |
| Metadata logical Tiles | 57.668 | 16.001 | -72.25% |
| Corrected total logical Tiles | 110.276 | 68.609 | -37.78% |
| Standalone lookup | 1.296 ms | 0.716 ms | -44.70% |

分片前后返回成员、Incidence scores 和最终选择保持完全一致。

### 17.3 Matched Integrated E2E

在相同模型权重、相同单一 grid 和相同选择下：

| 路径 | Mean query E2E |
|---|---:|
| Global rank control | 29.496 ms |
| Persona-sharded rank index | 28.717 ms |
| Paired delta | -0.779 ms |

共验证 25 个 OOF units、6,775 个 paired query rows，所有 selections identical。

### 17.4 增量发布闭环

确定性分区使用 33,627 条 base memories 和 8,407 条 new-memory pool。在 1%、5%、10%、20% 增长点上，256 个 probes 的 dispatch、overlay/compacted lookup 和 B8 selection mismatch 均为 0。

20% 增长时：

| 指标 | 数值 |
|---|---:|
| Appended memories | 8,407 |
| Tombstones | 841 |
| Active memories | 41,193 |
| Overlay lookup mean | 2.897 ms |
| Atomic publish | 11.461 ms |
| Full snapshot build | 6,181.377 ms |
| Compaction | 125.604 ms |
| Logical overlay Tiles | 526 |

这验证了从 frozen base、增量 overlay、immutable snapshot、atomic publish 到 compaction 的完整静态新记忆发布闭环。

### 17.5 Agent 执行闭环

在允许的 train-OOF 查询上，最终选择被实际送入本地 Agent，而不是停留在 retrieval proxy 指标。共执行 1,355 个查询和 B8/B16/B32 三个预算，共 4,065 条回答。默认 B8 结果如下：

| 指标 | B8 结果 |
|---|---:|
| Answer accuracy | 0.259041 |
| Mean prompt tokens | 2,538.4 |
| Mean completion tokens | 48.0 |
| Mean retrieval latency | 28.637 ms |
| Mean generation latency | 3,284.509 ms |
| Mean total Agent latency | 3,313.146 ms |

该结果完成了从 memory retrieval、context packing 到答案生成、token 与总时延计量的执行闭环。它是 train-OOF 范围内的系统测量，不等同于独立 formal heldout 上的泛化保证。

---

## 18. 方法边界

本文中的 Tile 是 host/logical Tile，不是 PIM 硬件测量。增量实验验证的是不依赖时间戳的静态新记忆分区发布，不推断真实时间顺序、访问顺序或 causal update behavior。

所有模型选择与检索验证均采用 5 seeds × 5 session-blocked train-OOF 设计；Agent 执行覆盖 1,355 个允许的 train-OOF 查询。在线接口不读取 evidence group、gold memory、formal-test label 或其他 supervision-only 字段。本文所称“完整闭环”是指离线建模、在线检索、索引访问、静态增量发布、上下文交付和 Agent 执行均已实现并可测；它不扩张为未获授权的 formal heldout 泛化结论。

---

## 19. 设计总结

方法一的核心不是单独引入一个超图打分器，而是把语义建模、索引物理组织、受控候选发现、固定预算选择和增量发布统一为同一套闭环：

1. 残差投影把冻结嵌入调整到多证据任务空间。
2. UOT 将记忆映射为可重叠的模糊 Prototype Incidence。
3. QESA 将查询映射到需要激活的语义区域。
4. Persona × Prototype rank shards 以严格等价方式减少无效扫描。
5. 三元 Bundle Beam 显式建模组合证据候选。
6. Cap-64 frontier 保证候选计算与最大预算均有界。
7. QHDS 通过单调次模边际增益平衡相关性与 Prototype 覆盖。
8. 单副本 Payload Gather 避免存储复制。
9. Overlay、Snapshot、Atomic Publish 和 Compaction 支持不依赖时间戳的新记忆增量维护。
10. 确定性上下文打包与受约束 Agent 接口闭合最终答案、token 和总时延测量。

最终系统因此实现了从离线语义构建、在线稀疏检索、固定预算证据选择、Payload 交付、Agent 执行到新记忆维护的完整 Method 1 设计闭环。
