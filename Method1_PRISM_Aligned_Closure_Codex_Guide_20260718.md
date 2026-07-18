# 方法一重新实现指南：面向 Reranker 前候选前沿优化的 PRISM 对齐闭环

> 适用对象：在服务器项目 `agent-memory-profiling` 中继续开发的 Codex。  
> 本轮目标：完成方法一的最小可信闭环，得到可复现、非负且至少在一个质量—效率维度上具有正收益的结果。  
> 本轮不要求达到 SOTA，不开展正式消融，不实现方法二或 PIM 仿真。

---

## 0. 执行契约

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

4. 若 Codex 当前终端已经位于服务器并且 `conda env list` 能找到该环境，可直接执行。否则必须通过项目已经配置的远程服务器入口执行；若服务器地址或远程入口不可发现，停止正式任务并只向用户询问远程入口，不得在聊天沙箱内用 CPU 结果冒充正式结果。
5. 不得使用 `base` 环境，不得创建替代环境，不得静默升级已有核心依赖。新增依赖只能安装到 `agent-memory-profiling`，安装前后都要导出环境快照。
6. GPU 只用于适合 GPU 的阶段：embedding、ANN 批量查询、cross-encoder reranking，以及经过正确性对照后确有收益的张量操作。稀疏超图传播可以保留在 CPU；不得为了“使用 GPU”而将低效控制流强行迁移到 GPU。

### 0.2 本轮唯一目标

实现下面这条完整流水线：

```text
长期记忆库
→ 轻量初始检索得到 query seeds
→ 查询条件化双视图超图扩展与候选前沿生成
→ C 条候选进入固定 Cross-Encoder Reranker
→ Top-K 记忆
→ 检索质量或 Agent Memory 回放结果
```

方法一的职责是减少昂贵 reranker 需要处理的 query–memory 对，同时尽量维持或提高候选质量。方法一自身不需要先成为原始 Agent 流水线中的最大瓶颈；它必须满足的是：

```text
超图前沿生成开销 + rerank(C)
< rerank(N)
```

其中 `N` 是高预算参考流水线送入 reranker 的候选数，`C < N` 是方法一的输出预算。

### 0.3 本轮明确延期的工作

以下内容不得阻塞本轮闭环：

- 完整 SOTA 方法复现与主表对比；
- 论文级正式消融实验；
- PRISM 内部 progressive cluster pruning 的重新实现；
- 方法二、PIM 模拟器、真实 PIM 硬件映射；
- Tile-aware layout、bank 映射与能耗模型；
- Qwen3 4B/8B、BGE-MiniCPM 等多模型扩展；
- 全部 18 个 PRISM 微基准数据集；
- 真实手机在线执行与完整 VLM 部署；
- 为获得正结果而进行无边界的大规模参数搜索。

这些内容只能在本指南的闭环 Gate 全部通过后启动。

### 0.4 科学性底线

“不能是负效果”表示：**不得把负结果包装成闭环成功**，而不是要求 Codex 修改数据或反复窥视测试集直到出现正结果。

必须遵守：

1. train/history、development、test 三部分严格隔离；
2. 所有阈值、超参数、instruction 和预算在 development 上确定后冻结；
3. test 只运行一次正式配置；
4. 若正式配置仍为负结果，冻结该结果并报告，不得继续用 test 调参；
5. 不得使用测试集 gold evidence、测试 query 类型或未来访问记录构图；
6. 所有方法使用相同记忆可见范围、相同 reranker、相同最大长度和相同硬件；
7. 质量比较、时延比较必须基于逐 query 配对结果；
8. 不得只报告百分比，必须同时报告绝对指标和绝对毫秒。

---

## 1. 方法一的正式定位

### 1.1 与 PRISM 的关系

PRISM 优化 reranker 内部推理：它让已经进入 cross-encoder 的候选在中间层逐步被接受、延迟或剪除。方法一优化 reranker 之前的候选前沿：它决定哪些记忆值得进入 cross-encoder。

因此二者位于同一条 semantic selection 流水线，但优化位置不同：

```text
Retriever/Seeds
→ [方法一：N 或全库 → C]
→ [PRISM：更快完成剩余 C 条的 rerank]
→ Top-K
```

本轮只实现方法一，并使用与 PRISM 一致的 Qwen3 embedding/reranker 组合、候选规模实验、质量—时延评测方式。闭环后再进行：

```text
Full-N + HuggingFace Reranker
Ours-C + HuggingFace Reranker
Full-N + PRISM
Ours-C + PRISM
```

### 1.2 形式化输入输出

给定记忆库：

```text
M = {m1, m2, ..., m|M|}
```

对查询 `q`，轻量检索器先生成 `S` 个 seeds：

```text
S_q = ANN(q, S)
```

方法一利用语义超图 `H_sem` 与历史共同检索超图 `H_hist`，生成固定预算 `C` 的前沿：

```text
F_q^C = Select(q, S_q, H_sem, H_hist, C)
```

固定 reranker 对 `F_q^C` 进行 cross-encoder 推理，并返回 `K` 个结果：

```text
R_q^K = Reranker(q, F_q^C, K)
```

本轮需要同时建立两个参考：

```text
Dense-C:       ANN 直接取 C 条 → 同一 reranker → Top-K
Full-Rerank-N: ANN/Hybrid 取 N 条 → 同一 reranker → Top-K
```

其中 `Dense-C` 判断超图在相同 reranker 候选预算下是否有价值；`Full-Rerank-N` 判断方法一能否用更少候选逼近高预算质量并获得净加速。这两项是闭环必需参考，不属于可以延期的正式消融。

### 1.3 本轮允许声称的贡献

若闭环成功，只允许声称：

> 查询条件化双视图超图利用模糊语义归属与历史共同检索关系，在 reranking 前生成更小的高质量候选前沿，从而减少 cross-encoder 处理的候选数，并在目标工作负载上获得正的端到端 semantic-selection 净收益。

本轮不得声称：

- 全面优于 Agent Memory SOTA；
- 超图在所有数据集上都优于图或 ANN；
- 已经实现真实 PIM 加速；
- A100 结果等价于端侧结果；
- `co-candidate` 或 `co-retrieval` 日志等价于模型真实读取行为；
- 方法一定适用于任意记忆规模和任意 Agent。

### 1.4 对 v22/v23 失败实现的处理

旧结果必须保留，不得覆盖。可以复用经过验证的数据加载、评测、日志和计时代码，但不得直接复用以下失败路径作为默认方法：

- 已坍缩的 UOT prototype；
- 把整个 namespace 拼成一个长时间 episode 的访问链；
- PrefixSpan 伪时序关系；
- 未实现 query-conditioned edge weight 的全局传播；
- 约 50 ms 的 Python 对象级 greedy/graph selector；
- 未经正确性证明的 QHDS 或 Tile-opening 优化器；
- 使用 test gold evidence 构造超边。

新版本建议创建为：

```text
versions/v24_prism_aligned_method1_closure/
```

若该目录已存在，创建新的递增版本，不得覆盖旧运行。

---

## 2. 统一数据契约

所有数据集必须先转换成数据集无关结构。禁止在核心选择器内读取 `persona`、题型名称或 gold evidence。

### 2.1 MemoryRecord

```json
{
  "memory_id": "string",
  "namespace_id": "string",
  "content": "string",
  "arrival_order": 0,
  "metadata": {},
  "eligible_from_order": 0
}
```

说明：

- `namespace_id` 只用于隔离不同用户、语料库或任务空间，不作为相关性特征；
- `arrival_order` 是逻辑到达顺序，不要求墙钟时间戳；
- 当前查询只能访问 `eligible_from_order <= query.cutoff_order` 的记忆。

### 2.2 QueryRecord

```json
{
  "query_id": "string",
  "namespace_id": "string",
  "query": "string",
  "cutoff_order": 0,
  "split": "history|development|test"
}
```

gold evidence、qrels 和最终答案必须存放在独立评测文件中，不能传入选择器。

### 2.3 HistoryEvent

```json
{
  "event_id": "string",
  "namespace_id": "string",
  "query_id": "string",
  "query_embedding_id": "string",
  "selected_memory_ids": ["m1", "m3", "m8"],
  "event_kind": "co_candidate|co_retrieval|co_use",
  "task_success": null,
  "arrival_order": 0
}
```

定义必须准确：

- `co_candidate`：初始检索器曾共同返回的候选；
- `co_retrieval`：最终曾共同送入下游模型的记忆；
- `co_use`：Agent 或工具明确使用的记忆。

没有真实读取日志时，不得把 `co_candidate` 改名为 `co_access`。

### 2.4 QueryRunResult

每个 query 和方法都必须输出一行 JSONL：

```json
{
  "run_id": "...",
  "query_id": "...",
  "method": "full_n|dense_c|ours_c",
  "seed_ids": [],
  "expanded_pool_ids": [],
  "frontier_ids": [],
  "reranked_ids": [],
  "seed_ms": 0.0,
  "edge_activation_ms": 0.0,
  "diffusion_ms": 0.0,
  "frontier_finalize_ms": 0.0,
  "rerank_ms": 0.0,
  "semantic_selection_ms": 0.0,
  "peak_cpu_bytes": 0,
  "peak_gpu_bytes": 0,
  "reranker_pairs": 0,
  "reranker_input_tokens": 0
}
```

`semantic_selection_ms` 必须包含从查询到 Top-K 的完整选择成本：

```text
seed retrieval
+ edge activation
+ hypergraph expansion/diffusion
+ frontier finalization
+ reranking
```

---

## 3. 方法一核心算法

### 3.1 总体结构

本轮实现一个训练轻量、查询条件化的双视图超图前沿生成器：

```text
H_sem：模糊语义超图
H_hist：历史共同检索超图
S_q：ANN query seeds
→ 查询条件化边激活
→ 局部稀疏超图随机游走/扩散
→ dense relevance 与结构支持融合
→ Top-C frontier
```

所有 `C` 个位置均由统一融合分数排序产生，不使用“超图只能补少量残差槽位”的最终设计。dense 信号通过 restart 和融合项提供安全锚点，而不是硬编码占据大多数位置。

### 3.2 模糊语义超图 `H_sem`

#### 3.2.1 初始实现

本轮不使用已失败的 UOT。采用稳定、可复现的 soft spherical clustering：

1. 使用冻结的 Qwen3-Embedding-0.6B 编码 history/train 记忆；
2. 对 embedding 做 L2 normalization；
3. 在 history/train 记忆上拟合 spherical k-means 或归一化 MiniBatchKMeans；
4. 每条记忆保留与其最相近的 `r_sem` 个 prototype；
5. 使用温度 softmax 将余弦相似度转换成软隶属度；
6. 一条记忆可同时属于多个语义超边。

软隶属度：

```text
h_sem(m, e)
= softmax_e(cos(z_m, c_e) / tau_sem)
```

初始 development 搜索范围：

```text
n_prototypes ∈ {64, 128, 256}
r_sem        ∈ {2, 4}
tau_sem      ∈ {0.05, 0.10, 0.20}
```

不得在 test 上选择这些参数。数据量很小时，应将 prototype 数限制为不超过 `sqrt(number_of_history_memories)` 的合理倍数，避免产生大量空原型。

#### 3.2.2 表征健康检查

构图后必须输出：

- 非空 prototype 数量与占比；
- 每个 prototype 的成员数分布；
- centroid pairwise cosine 分布；
- 每条记忆的 assignment entropy；
- 每条记忆有效隶属的超边数；
- 最大/中位节点度；
- 是否存在单个 prototype 覆盖绝大多数记忆。

出现以下情况必须停止，不得只 warning 后继续：

- 大部分 prototype 为空；
- prototype 彼此接近到不能区分；
- 几乎所有记忆具有相同的软分配；
- 单一超边连接几乎整个 namespace；
- incidence 中出现 NaN、Inf 或负数。

只有在 spherical 方案通过健康检查后，未来才允许重新尝试 UOT，并把它作为替代构图器而不是方法身份。

### 3.3 历史共同检索超图 `H_hist`

每个历史查询或 Agent 步骤是一个独立事件边界：

```text
e_t = {该事件中共同成为候选、共同送入下游或共同被使用的记忆}
```

不要求时间戳，也不构造跨查询的伪顺序链。频繁共同出现会自然表现为多个历史超边之间的重复重叠。

构建规则：

1. 只使用 history/train 查询产生的日志；
2. 一条事件内去重 memory ID；
3. 去除 singleton event；
4. 超长事件按当时的冻结相关性顺序截断，默认最大 16 或 32 条，并记录截断率；
5. 完全相同的事件可合并并增加 `support`；近似事件暂不强制合并；
6. 若有任务结果，保存 `task_success`，但没有结果时不得伪造；
7. history query embedding 作为该超边的上下文向量 `z_e`；
8. test 查询、test qrels、test gold evidence 不得进入 `H_hist`。

对于 MedMemoryBench，如果没有真实 Agent 读取日志：

- 用 history/train query 的冻结检索输出构造 `co_candidate` 或 `co_retrieval` 超边；
- 不使用 evidence group 构造超边；
- 论文表述必须称为共同候选/共同检索局部性。

对于 MobiAgent/AgentRR 回放：

- 优先使用历史成功任务、轨迹候选和实际复用记录；
- 若只能获得 reranker 输入列表，则标记为 `co_candidate`；
- 只有明确被回放或采用的轨迹才能标记为 `co_use`。

### 3.4 查询条件化超边激活

对当前查询 `q`，先取得 `S` 个 ANN seeds，并把归一化 dense 分数写入初始分布 `x0`。

每条超边的在线激活至少同时依赖：

1. 当前查询与超边上下文/semantic prototype 的相似度；
2. 当前 seeds 落在该超边上的归一化相关性质量；
3. 超边自身可靠度。

建议实现：

```text
query_match(e, q)
= sigmoid((cos(z_q, z_e) - mu_type) / tau_type)

seed_mass(e, q)
= sum_{m in e ∩ S_q} normalized_dense_score(m)

a_e(q)
= reliability(e) × query_match(e, q) × (epsilon + seed_mass(e, q))
```

其中：

- semantic edge 的 `z_e` 是 prototype；
- history edge 的 `z_e` 是历史 query embedding；
- `reliability(e)` 可由无需标签的边紧致度、support 和可选的任务成功信息计算；
- 无任务结果时使用结构可靠度，不得默认任务成功率为 100% 并在报告中当成事实；
- `mu_type`、`tau_type` 按 edge type 在 development 上冻结。

只保留激活度最高、且与当前 seeds 或查询 prototype 有联系的边。不得在每个 query 上扫描所有 history hyperedges。

初始建议：

```text
top semantic edges:  8–32
top history edges:   16–64
expanded pool cap:   4C–8C
```

具体值在 development 上确定。

### 3.5 局部超图扩散

在激活边及其成员形成的局部子图上构造稀疏 incidence matrix `H`。`H` 的形状为 `|V_local| × |E_active|`。

令：

```text
A_q = diag(a_e(q))
D_e = diag(H^T 1)
D_v = diag(H A_q 1)
```

行随机转移矩阵：

```text
P_q = D_v^-1 H A_q D_e^-1 H^T
```

若 `x_t` 使用列向量，则带 restart 的更新为：

```text
x_(t+1) = alpha × x0 + (1 - alpha) × P_q^T x_t
```

仅执行少量固定步数：

```text
T     ∈ {1, 2, 3}
alpha ∈ {0.6, 0.8, 0.9}
```

restart 项确保查询相关 seed 信号不会被全局连通性淹没。不得运行至全局收敛，也不得跨 namespace 传播。

### 3.6 前沿融合与选择

对局部候选池中的每条记忆，都可以直接用预计算 memory embedding 与当前 query embedding 计算便宜的 dense relevance，无需再次运行 cross-encoder。

最终分数：

```text
score(m | q)
= beta × norm(dense_similarity(q, m))
+ (1 - beta) × norm(x_T(m))
```

初始范围：

```text
beta ∈ {0.5, 0.7, 0.9}
```

`norm` 必须在当前局部候选池内采用固定、可复现的 rank normalization 或 robust z-score。不得把量纲不同的原始分数直接相加。

按最终分数选择完整的 Top-C，保证：

- ID 唯一；
- 所有 ID 因果可见；
- 数量恰好为 `min(C, visible_pool_size)`；
- 相同分数使用 memory ID 做确定性 tie-break；
- 输出可回溯到激活边和得分分量。

### 3.7 高性能实现要求

上一版本约 50 ms 的 Python 图选择器不能继续作为正式实现。新版本必须：

- 使用 CSR/CSC 稀疏矩阵或连续数组保存 incidence；
- 为 `memory → incident_edges` 和 `edge → members` 建立压缩倒排表；
- 不在 query 热路径中创建 NetworkX 图；
- 不在节点级使用 Python set 的多层嵌套循环；
- 批量计算 query–prototype 和 query–memory cosine；
- 预分配局部 buffer，避免每 query 大量对象分配；
- 将候选池上限固定在 `4C–8C`；
- 分别记录 edge activation、pool materialization、diffusion 和 finalization 时延；
- 优先保证 CPU 稀疏路径正确，再判断 GPU 稀疏张量是否有真实收益。

### 3.8 最小增量能力

本轮不要求完整在线重分区，但方法一必须具备功能闭环：

1. 新 memory 到达后生成 embedding；
2. 只计算该 memory 对现有 semantic prototypes 的 soft membership；
3. 新 history event 只追加一条 incidence row/edge 或更新完全相同事件的 support；
4. 更新倒排索引；
5. 不允许全量重新编码全部历史记忆；
6. 执行插入前后正确性测试，确保旧 query 的可见性和新 memory 的 causal eligibility 正确。

增量时延只作闭环记录，不作为本轮主结果。

---

## 4. 新模型部署

### 4.1 本轮强制模型

只强制部署两个官方模型：

```text
Qwen/Qwen3-Embedding-0.6B
Qwen/Qwen3-Reranker-0.6B
```

原因：

- 与 PRISM 的主要模型设置直接对齐；
- embedding 与 reranker 属于同一模型系列；
- 0.6B 足以完成闭环并控制显存、下载和推理成本；
- 官方支持自定义 retrieval instruction；
- 闭环后可无缝扩展到 4B/8B。

本轮不得先部署 4B/8B 来掩盖算法问题。

### 4.2 环境和依赖检查

进入项目根目录后首先运行并保存输出：

```bash
conda env list
conda run -n agent-memory-profiling python -V
conda run -n agent-memory-profiling python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
conda run -n agent-memory-profiling python -c "import accelerate; print(accelerate.__version__)"
nvidia-smi
```

在安装前保存：

```text
artifacts/environment/conda_explicit_before.txt
artifacts/environment/pip_freeze_before.txt
artifacts/environment/gpu_before.txt
```

Qwen3 官方实现要求：

```text
transformers >= 4.51.0
```

处理规则：

1. 若现有版本满足要求且 smoke test 通过，不升级；
2. 若低于 4.51.0，先运行 dry-run 并检查是否会替换 `torch`、CUDA 相关包或大量核心依赖；
3. 优先采用 PRISM 使用的兼容版本 `transformers==4.52.4`；
4. `accelerate==1.6.0` 仅在现有版本不兼容时考虑；
5. 不得升级或降级 PyTorch/CUDA 来迁就模型；
6. 若存在冲突风险，停止并向用户报告拟变更包，不得强装；
7. 安装后再次导出 `conda_explicit_after.txt` 与 `pip_freeze_after.txt`。

不强制安装 `sentence-transformers`。优先直接使用 Transformers 官方包装，减少对现有环境的扰动。

### 4.3 模型获取规则

1. 先搜索服务器现有官方 checkpoint，避免重复下载；
2. 只允许使用官方 Hugging Face 仓库或项目已有的官方模型副本；
3. 不得使用非官方镜像、名字相似的社区量化版本或重命名 checkpoint；
4. 若官方源在服务器不可访问，停止并报告，不得静默切换第三方镜像；
5. 模型不得放入版本结果目录，统一放在项目模型根目录；
6. 记录 repo ID、resolved revision/commit、文件路径、dtype 和下载时间；
7. 下载完成后正式实验使用 `local_files_only=True`，避免运行中访问网络。

推荐目录：

```text
${PROJECT_ROOT}/models/Qwen3-Embedding-0.6B/
${PROJECT_ROOT}/models/Qwen3-Reranker-0.6B/
```

如果项目已有 `model/` 而非 `models/`，复用现有约定，不得同时创建两套重复权重。

若 `hf` CLI 已安装，可使用官方下载命令；若只有旧版 CLI，则使用对应的 `huggingface-cli download`。不得为了 CLI 额外升级整个环境。示意：

```bash
hf download Qwen/Qwen3-Embedding-0.6B --local-dir "${MODEL_ROOT}/Qwen3-Embedding-0.6B"
hf download Qwen/Qwen3-Reranker-0.6B --local-dir "${MODEL_ROOT}/Qwen3-Reranker-0.6B"
```

必须生成：

```text
artifacts/models/models_manifest.json
```

### 4.4 Embedding 封装

必须遵循官方 last-token pooling、left padding 和 L2 normalization：

```text
AutoTokenizer(..., padding_side="left")
AutoModel(...)
last_token_pool(last_hidden_state, attention_mask)
L2 normalize
```

第一轮固定：

```text
dtype: bfloat16（A100）；端侧不支持时使用 float16
attention: sdpa
max_length: 512（默认微基准）
embedding dimension: 1024
```

不要把模型支持的 32K context 直接作为默认长度。必须按 workload 固定最大长度并报告截断率。

query 使用英文 instruction，document 不加 query instruction。建议：

通用检索：

```text
Given a search query, retrieve relevant passages that answer the query.
```

Agent Memory：

```text
Given the current agent task, retrieve historical memory items or successful trajectories that are useful for completing the task.
```

MedMemoryBench：

```text
Given a question about a user's history, retrieve memory records that provide the evidence needed to answer it.
```

instruction 必须在 development 阶段冻结，test 不得改写。

### 4.5 Reranker 封装

使用官方 `Qwen3-Reranker-0.6B` 的 yes/no logit 评分逻辑，不能把生成文本作为排序结果。默认：

```text
AutoModelForCausalLM(...).eval()
padding_side="left"
enable_thinking=False 或使用官方固定 suffix
score = P("yes") / (P("yes") + P("no"))
```

第一轮固定：

```text
dtype: bfloat16（A100）
attention: sdpa
max_length: 512
batch size: 根据显存一次确定，所有方法相同
```

FlashAttention 不是闭环依赖。只有服务器已经正确安装，并且与 SDPA 的排名一致性测试通过后，才能作为后续加速配置；否则统一使用 SDPA。

### 4.6 模型 smoke test

正式数据运行前必须通过：

#### Embedding

- 输出维度为 1024；
- 无 NaN/Inf；
- L2 norm 接近 1；
- 正相关 query–document 相似度高于明显负样本；
- 相同输入重复运行误差在 dtype 合理范围；
- batch 与 single inference 排序一致；
- instruction 生效且没有被错误添加到 document。

#### Reranker

- `yes`、`no` token ID 有效且不同；
- 正样本分数高于明显负样本；
- batch 与 single inference 排名一致；
- 更改候选顺序不会改变每个 pair 的分数；
- 相同 pair 重复运行稳定；
- max length、截断方向和 padding 方式正确；
- 记录第一次加载时间、warm inference 时间和峰值显存。

Gate 未通过时不得开始超图实验。

---

## 5. 数据集与工作负载

### 5.1 闭环层级

本轮不直接复制 PRISM 的全部 18 数据集，而采用三层闭环：

1. **主算法闭环：MedMemoryBench English**；
2. **通用检索验证：1–3 个 PRISM/BEIR 风格数据集**；
3. **Agent Memory 闭环：PRISM/MobiAgent Agent Memory 离线回放**。

只有第一层通过后才进入第二层，前两层通过后才进入第三层。

### 5.2 MedMemoryBench

复用当前正式数据，但重新建立独立版本的数据适配器。已知规模可以作为输入核对，不得硬编码为算法逻辑：

```text
memory: 42,034
queries: 1,939
```

要求：

- 优先沿用数据集官方或当前已冻结的 train/development/test 划分；
- history hyperedges 只能由 train/history query 的实际冻结检索输出生成；
- evidence group 只用于评测；
- query type 只用于最终分组分析，不得输入选择器；
- namespace 只用于可见性隔离；
- 重新生成所有 Qwen3 embeddings，不得混用旧 embedding 空间。

### 5.3 通用检索数据

闭环阶段优先选择有稳定 qrels、规模可控且领域不同的 1–3 个数据集，例如：

```text
BEIR NFCorpus
BEIR SciFact
BEIR FiQA
```

先选择一个完成闭环，再扩展到三个。不得一开始下载和处理完整 BEIR。

如果数据集没有可直接使用的 history/development/test：

1. 按 query ID 的稳定 hash 做确定性划分；
2. 保存 split manifest；
3. history query 的检索输出用于构建 `H_hist`；
4. qrels 只用于评测和 development 参数选择；
5. test query 不得以任何形式参与构图。

### 5.4 Agent Memory 回放

优先复用 PRISM artifact 中的 Agent Memory 实验数据和脚本接口；如该 artifact 只提供接口而缺少数据，再使用官方 MobiAgent/AgentRR 仓库中的公开轨迹与回放数据。

本轮不要求部署真实手机和完整 VLM。先实现确定性的离线回放：

```text
当前 Agent task/step
→ 历史轨迹或经验候选
→ 方法一前沿
→ Qwen3-Reranker-0.6B
→ 轨迹选择/复用判断
→ 离线任务结果或与参考决策的一致性
```

外部代码规则：

- 放在 `third_party/` 或项目已有外部依赖目录；
- 保存来源 URL、license 和 commit hash；
- 不直接修改 upstream 文件；
- 通过 adapter 接入本项目统一数据契约；
- 不安装完整 MobiAgent 依赖栈，除非离线数据适配确实需要；
- 若依赖与当前 Conda 环境冲突，停止并报告，不得创建新环境或破坏现有环境。

Agent Memory 必须记录：

- 每任务/步骤检索调用次数；
- 每次进入 reranker 的候选数；
- reranker 时间；
- semantic-selection 总时间；
- 轨迹选择正确率或参考一致性；
- 可获得时记录任务成功率；
- 完整任务累计节省时间。

### 5.5 候选规模

先执行 PRISM 对齐 sanity 配置：

```text
N = 20
C = 10
K = 5
```

它只用于验证流水线，不作为方法主结果。主闭环使用更能体现 pre-rerank 压缩的规模：

```text
N ∈ {64, 128, 256}
C ∈ {16, 32, 64}, C < N
S ∈ {8, 16}
K ∈ {5, 10}, K <= C
```

其中：

- `N`：Full-Rerank 高预算候选数；
- `C`：方法一和 Dense-C 送入 reranker 的候选数；
- `S`：方法一初始 ANN seeds 数；
- `K`：最终输出数。

必须报告实际 Agent workload 的候选数分布，不得仅使用人为放大的候选规模。若收益只在 `N >= 128` 出现，必须将其写成适用范围，而不是隐藏 `N=20/64` 的结果。

---

## 6. 闭环必需方法与公平性

### 6.1 Full-Rerank-N

```text
同一冻结 retriever 取 N 条
→ Qwen3-Reranker-0.6B 全量 rerank
→ Top-K
```

用途：高预算质量参考和 reranker 成本基线。

### 6.2 Dense-C

```text
同一 Qwen3 embedding 直接取 C 条
→ 同一 reranker
→ Top-K
```

用途：相同 reranker 候选预算下的强简单基线。

### 6.3 Ours-C

```text
同一 Qwen3 embedding 取 S 条 seeds
→ 查询条件化 H_sem + H_hist
→ 输出 C 条
→ 同一 reranker
→ Top-K
```

用途：本轮完整方法。

### 6.4 公平性要求

- 三种方法使用相同可见记忆库；
- embedding 模型、query instruction 和记忆 embedding 完全相同；
- reranker 模型、prompt、dtype、max length 和 batch size 完全相同；
- `Dense-C` 与 `Ours-C` 都恰好向 reranker 发送 C 条；
- 所有候选去重；
- 相同 query 在相同设备、相同进程策略和相同缓存条件下运行；
- 禁止只为 Ours 使用 test query 类型、gold evidence 或额外大模型；
- Ours 的 ANN seed、超图激活和扩散全部计入时延；
- Full-N 的 ANN/Hybrid retrieval 同样计入相应流水线时延；
- 模型首次加载时间单独报告，不混入 warm per-query 时延；
- cold/warm 两种条件分开报告。

本轮不要求 graph、Hyper-RAG、HippoRAG、MMR、PRISM 等正式外部方法，也不要求 semantic-only/access-only 等消融。

---

## 7. 指标与统计

### 7.1 候选前沿质量

在 rerank 前记录：

- relevant recall@C；
- evidence coverage@C；
- evidence density@C；
- multi-group complete coverage（MedMemoryBench 可用时）；
- 与 Full-N 候选池的 oracle overlap；
- expanded pool size 和 frontier size。

### 7.2 最终排序质量

至少记录：

- Precision@K；
- Recall@K；
- nDCG@10；
- MRR@10；
- 与 Full-Rerank-N Top-K 的 overlap/agreement；
- MedMemoryBench 最终答案或 evidence 指标；
- Agent Memory 轨迹选择正确率、参考一致性或任务成功率。

不要只用 Full-Rerank agreement 代替 gold quality：Full-Rerank 也可能排序错误。

### 7.3 效率指标

至少记录：

- seed retrieval ms；
- edge activation ms；
- expanded pool materialization ms；
- diffusion ms；
- frontier finalization ms；
- rerank ms；
- semantic-selection total ms；
- reranker query–candidate pair 数；
- reranker input token 数；
- peak CPU RSS；
- peak GPU allocated/reserved memory；
- 索引构建时间与磁盘占用；
- 每条 memory 增量插入时间。

Logical Tile、bank traffic 和 PIM energy 可以继续由已有 profiler 记录，但不进入本轮 Gate，也不得作为 PIM 实测结果。

### 7.4 净收益

逐 query 计算：

```text
rerank_saved_ms(q)
= rerank_ms_full_n(q) - rerank_ms_ours_c(q)

method_overhead_ms(q)
= seed_ms + edge_activation_ms + diffusion_ms + frontier_finalize_ms

net_gain_ms(q)
= semantic_selection_ms_full_n(q)
 - semantic_selection_ms_ours_c(q)
```

同时报告：

```text
speedup = total_ms_full_n / total_ms_ours_c
candidate_reduction = 1 - C / N
```

### 7.5 统计方法

- 每个正式配置至少完成足够 query 数，不得用极少样本下结论；
- 时延先 warmup，再重复测量；
- GPU 计时前后使用正确同步；
- 报告 median、mean、P95 和 95% bootstrap CI；
- 质量差异使用 query-level paired bootstrap；
- Agent 任务成功率使用配对或适合二项结果的置信区间；
- 所有随机种子固定并写入配置；
- 不得删除慢 query 或失败 query，除非预注册为数据错误并保留审计记录。

---

## 8. 闭环 Gate

### Gate A：环境与模型

必须满足：

- 正式服务器和 Conda 环境正确；
- 环境前后快照完整；
- 两个 Qwen3 0.6B 官方 checkpoint 可离线加载；
- embedding 与 reranker smoke test 全部通过；
- 模型 revision、dtype、max length 和 instruction 已记录。

### Gate B：数据与泄漏审计

必须满足：

- 三类统一记录格式通过 schema 校验；
- history/development/test manifest 已冻结；
- test query 和 test gold 未进入任何超边；
- 所有 query 只访问因果可见记忆；
- Full-N、Dense-C、Ours-C 使用同一可见范围；
- candidate ID 和 memory ID 全部可追溯。

### Gate C：候选算法开发集闭环

在 development 上先检查：

1. `H_sem` 健康，无 prototype collapse；
2. `H_hist` 的 event size、overlap 和节点度合理；
3. 查询级激活边与 query/seed 有实际联系；
4. Ours-C 输出数量和因果资格正确；
5. Ours-C 的 rerank 前 relevant/evidence coverage 不弱于 Dense-C；
6. Ours-C 的候选池不是被单个高阶 hub 支配；
7. 在线方法开销随 `C` 和激活边数量受控增长。

如果 Gate C 不通过，不能进入 test。

### Gate D：development 正收益

至少满足以下两种正向模式之一：

#### 模式 1：同预算质量提升

```text
quality(Ours-C) >= quality(Dense-C)
```

并且至少一个核心质量指标具有稳定正增益。

#### 模式 2：同质量候选减少

```text
quality(Ours-C) 在预先冻结的非劣界内接近 Full-Rerank-N
且 C 显著小于 N
```

同时必须满足：

```text
semantic_selection_ms(Ours-C)
< semantic_selection_ms(Full-Rerank-N)
```

建议在 development 上预先冻结质量非劣界，例如 nDCG@10 或 Precision@K 的绝对下降不超过 0.5–1.0 个百分点；具体界限必须根据任务指标尺度写入配置，不能 test 后再定义。

### Gate E：正式 test

冻结唯一正式配置后运行一次 test。闭环成功需要：

1. Ours-C 相对 Dense-C 的平均质量差不为负，且配对 95% CI 下界不低于预先冻结的 `-epsilon_dense`；
2. Ours-C 相对 Full-Rerank-N 满足预先冻结的质量非劣界；
3. Ours-C 的 semantic-selection 总时延低于 Full-Rerank-N；
4. `net_gain_ms` 的均值和中位数为正；
5. 至少一个真实或离线 Agent Memory 工作负载中，轨迹选择/任务质量不下降；
6. 所有失败 query 和负 tail case 被保留并解释。

若 test 没有通过，必须把本轮标记为失败版本，不得继续对 test 调参。

### Gate F：方法一闭环报告

只有 A–E 全部通过，才能把方法一标记为：

```text
METHOD1_CLOSURE_PASSED
```

---

## 9. 受控参数选择

正式 test 前，只允许在 development 上进行有限搜索。推荐先固定大部分参数，再采用逐层或 successive-halving 方式，而不是全笛卡尔积：

```text
1. 固定模型、N、C、S、K
2. 选择 n_prototypes / r_sem / tau_sem
3. 冻结 H_sem 后选择 edge activation cap
4. 选择 alpha / T
5. 最后选择 beta
```

初始中心配置：

```yaml
model:
  embedding: Qwen/Qwen3-Embedding-0.6B
  reranker: Qwen/Qwen3-Reranker-0.6B
  dtype: bfloat16
  attention: sdpa
  max_length: 512

budget:
  full_n: 128
  seed_s: 16
  frontier_c: 32
  final_k: 5
  expanded_pool_cap: 256

semantic_hypergraph:
  n_prototypes: 128
  top_memberships: 4
  temperature: 0.10

history_hypergraph:
  max_event_size: 16
  top_active_edges: 32

diffusion:
  alpha_restart: 0.8
  steps: 2
  beta_dense: 0.7

random_seed: 20260718
```

该配置只是 development 起点，不是预设最优结果。

---

## 10. 代码与目录结构

建议结构：

```text
versions/v24_prism_aligned_method1_closure/
├── README.md
├── configs/
│   ├── model_qwen3_06b.yaml
│   ├── medmemorybench.yaml
│   ├── beir.yaml
│   └── closure_main.yaml
├── src/
│   ├── data_contracts.py
│   ├── model_embedding_qwen3.py
│   ├── model_reranker_qwen3.py
│   ├── ann_seed_retriever.py
│   ├── semantic_hypergraph.py
│   ├── history_hypergraph.py
│   ├── query_conditioned_diffusion.py
│   ├── frontier_selector.py
│   ├── baselines.py
│   ├── metrics.py
│   └── profiling.py
├── scripts/
│   ├── 00_snapshot_environment.py
│   ├── 01_verify_models.py
│   ├── 02_prepare_dataset.py
│   ├── 03_encode_memories.py
│   ├── 04_freeze_candidate_reference.py
│   ├── 05_build_history_events.py
│   ├── 06_build_dual_hypergraph.py
│   ├── 07_run_development.py
│   ├── 08_freeze_config.py
│   ├── 09_run_test_once.py
│   ├── 10_run_agent_memory_replay.py
│   └── 11_aggregate_closure_report.py
├── tests/
│   ├── test_model_wrappers.py
│   ├── test_split_leakage.py
│   ├── test_causal_eligibility.py
│   ├── test_hypergraph_transition.py
│   ├── test_frontier_contract.py
│   ├── test_batched_rerank_equivalence.py
│   └── test_incremental_insert.py
├── artifacts/
│   ├── environment/
│   ├── models/
│   ├── manifests/
│   ├── hypergraph/
│   └── frozen/
├── runs/
├── aggregate/
└── reports/
```

不得把模型权重、原始大数据集或第三方仓库复制到版本目录。

### 10.1 必须冻结的中间产物

```text
models_manifest.json
dataset_manifest.json
split_manifest.json
memory_embeddings manifest
history_events.jsonl
semantic_hypergraph_stats.json
history_hypergraph_stats.json
full_n_candidate_ids.jsonl
dense_c_candidate_ids.jsonl
frozen_development_config.yaml
test_query_results.jsonl
latency_breakdown.jsonl
aggregate_metrics.csv
closure_report.md
```

所有产物必须带 config hash、code commit、数据版本和模型 revision。

---

## 11. 推荐执行顺序

### Phase 1：审计现状，不运行正式实验

1. 确认服务器、项目根目录、Git 状态和 Conda 环境；
2. 定位 v22/v23 的可复用 loader、metrics 和 profiling；
3. 列出将复用与废弃的文件；
4. 创建新版本目录；
5. 导出环境快照；
6. 不修改旧版本结果。

### Phase 2：部署并验证 Qwen3 0.6B 模型

1. 检查是否已有官方 checkpoint；
2. 检查 Transformers 兼容性；
3. 必要时执行有记录的最小依赖调整；
4. 下载/定位两个模型；
5. 完成 embedding/reranker wrapper；
6. 运行全部 model smoke tests；
7. 冻结模型 manifest。

### Phase 3：建立三条无超图参考流水线

1. 重新编码记忆；
2. 生成 ANN seeds；
3. 实现 Full-Rerank-N；
4. 实现 Dense-C；
5. 验证候选 ID、reranker score 和 Top-K；
6. 测量 PRISM 对齐 sanity 配置 `20 → 10 → 5`；
7. 测量 `N={64,128,256}` 的 reranker scaling；
8. 确认减少候选确实能减少 reranker 时间。

如果 reranker 时间不随候选数合理增长，先修复 batching、计时或模型调用，不能继续超图实现。

### Phase 4：构建双视图超图

1. 只用 history/train 记忆拟合 semantic prototypes；
2. 运行表征健康检查；
3. 只用 history/train query 日志构建 history events；
4. 生成 CSR/CSC incidence 和倒排索引；
5. 运行 leakage、causal 和 transition row-sum 测试；
6. 输出结构统计，不运行 test。

### Phase 5：development 候选质量

按顺序检查：

```text
expanded-pool upper bound
→ Ours-C frontier coverage
→ Ours-C reranked quality
→ 相对 Dense-C
→ 相对 Full-Rerank-N
```

如果 expanded pool 本身不包含足够证据，不能调最终 selector 掩盖问题；先修复边激活和扩展。

### Phase 6：development 净时延

1. 冻结候选质量可接受的配置；
2. 分别测量方法各阶段；
3. 与 Full-Rerank-N 比较完整 semantic-selection；
4. 画出 `N/C` 与质量、净收益的曲线；
5. 确定正收益的实际候选规模范围；
6. 冻结唯一 test 配置。

### Phase 7：正式 test 一次

1. 写出 config hash；
2. 检查测试集未被读过；
3. 运行 Full-N、Dense-C、Ours-C；
4. 聚合质量、时延和置信区间；
5. 不修改配置重跑；
6. 通过则进入 Agent Memory 回放，失败则冻结报告。

### Phase 8：Agent Memory 离线回放

1. 适配 PRISM/MobiAgent 官方数据；
2. 用历史部分构造双视图超图；
3. 当前任务只访问历史轨迹；
4. 比较 Full-N、Dense-C、Ours-C；
5. 记录每步与整任务累计时延；
6. 记录轨迹选择质量和可获得的任务成功率；
7. 形成最终闭环报告。

---

## 12. 如果 development 仍是负效果

严格按以下顺序定位，不得直接扩大模型或窥视 test：

### 12.1 先排除实现错误

- Qwen3 query instruction 是否正确；
- document 是否错误加入 instruction；
- embedding 是否归一化；
- reranker prompt、yes/no token 和截断是否正确；
- candidate ID 与文本是否错位；
- history/test 是否泄漏或混用；
- query-conditioned edge weight 是否真实生效；
- `P_q` 是否行随机，列向量传播是否使用 `P_q^T`；
- batch 与 single 结果是否一致。

### 12.2 再检查扩展池上界

若 expanded pool coverage 不高于 Dense-C：

- 检查 semantic prototype 健康；
- 检查 query–prototype instruction/embedding 空间；
- 减少 hub edge；
- 提高 seed overlap 在边激活中的权重；
- 降低 history edge 的全局度数偏置；
- 调整 active edge 数和 pool cap；
- 检查历史事件是否确实存在共同检索重复。

### 12.3 再检查 frontier 排序

若 pool 有证据但 Top-C 丢失：

- 提高 restart `alpha`；
- 提高 dense 融合 `beta`；
- 将扩散步数限制到 1–2；
- 使用 rank normalization；
- 检查高阶 hub 的 degree normalization；
- 输出每个丢失 gold memory 的 dense/semantic/history 分量。

### 12.4 最后检查时延

若质量通过但净收益为负：

- 确认没有扫描全部边；
- 检查 Python object/set 热路径；
- 缩小 active edges 和 expanded pool cap；
- 使用稀疏矩阵批量传播；
- 检查 reranker 是否真正只收到 C 条；
- 检查 GPU 同步和 batch strategy；
- 报告收益 crossover，而不是只看单一 `N`。

### 12.5 停止条件

在 development 上完成预注册的有限参数搜索后，如果：

```text
Ours-C 仍显著弱于 Dense-C
或
Ours-C 无法在任何现实 N/C 下获得正净收益
```

则停止本版本，输出失败报告。不得进入 test，也不得通过增大 reranker、减少测试样本或选择性删除 query 制造正结果。

---

## 13. 最终报告格式

`reports/closure_report.md` 必须包含：

### 13.1 一页结论

- 方法是否闭环；
- 是否通过每个 Gate；
- 最佳但冻结的 `N/S/C/K`；
- 相对 Dense-C 的质量变化；
- 相对 Full-N 的质量变化；
- reranker 候选减少比例；
- 方法额外开销；
- reranker 节省；
- 最终 net gain 和 speedup；
- Agent Memory 回放质量与累计时延；
- 适用规模和失败范围。

### 13.2 必需主表

| Dataset/Workload | Method | N | S | C | K | Precision@K | nDCG@10 | Ref overlap | Method ms | Rerank ms | Total ms | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 13.3 必需曲线

暂时只生成用于判断的简洁图，不按论文美化：

1. `C` 或 candidate-reduction 对最终质量；
2. `N/C` 对 semantic-selection 总时延；
3. `N/C` 对 net gain；
4. 每阶段时延 breakdown；
5. Agent 每任务累计选择时延与任务质量。

### 13.4 负例与边界

- 失败 query 类型；
- 超图发生 semantic drift 的实例；
- Dense-C 更好的实例；
- history locality 不充分的 namespace；
- 低候选规模下无净收益的情况；
- 所有 OOM、超时、缺失数据和依赖问题。

---

## 14. 闭环通过后的下一阶段

只有 `METHOD1_CLOSURE_PASSED` 后，才能依次开展：

1. 扩展至 PRISM 的更多 BEIR、LoTTE、Wikipedia、CodeRAG 数据；
2. 部署 Qwen3-Reranker-4B/8B 和 BGE reranker；
3. 在真实端侧设备测量，而不是用 A100 代替；
4. 集成官方 PRISM，完成 2×2 组合实验；
5. 加入正式外部 baseline；
6. 开展 semantic-only、history-only、无 query gate、图 vs 超图等正式消融；
7. 做大规模记忆、候选数和历史长度敏感性；
8. 再进入方法二的 PIM mapping、Tile layout、traffic 与 energy 仿真。

---

## 15. Codex 最终执行要求

Codex 接到本指南后应直接工作，不要先重新设计另一套方法。执行时：

1. 先输出对现有项目和旧版本的简短审计；
2. 建立新版本目录并保留旧结果；
3. 部署和验证两个官方 Qwen3 0.6B 模型；
4. 先完成 Full-N 与 Dense-C，证明 reranker 成本随候选规模变化；
5. 再实现查询条件化双视图超图；
6. 只在 development 上迭代；
7. 每个 Gate 完成后写入状态和证据文件；
8. test 只运行一次冻结配置；
9. 不因本轮暂不做消融而删除中间日志，后续消融需要这些接口；
10. 最终无论成功或失败，都必须提供完整结果、运行命令、配置、日志路径和下一步建议。

本轮最重要的判断不是“某个超参数是否最好”，而是：

> 双视图超图是否能用较小的 `C` 保持或提高候选质量，并让其自身开销小于减少候选所节省的 cross-encoder reranking 成本。

---

## 16. 官方参考入口

- PRISM / MenSelect 论文：`https://doi.org/10.1145/3767295.3803572`
- PRISM artifact（以论文附录公布入口为准）：`https://doi.org/10.5281/zenodo.18809731`
- Qwen3-Embedding-0.6B：`https://huggingface.co/Qwen/Qwen3-Embedding-0.6B`
- Qwen3-Reranker-0.6B：`https://huggingface.co/Qwen/Qwen3-Reranker-0.6B`
- Qwen3 Embedding 官方代码：`https://github.com/QwenLM/Qwen3-Embedding`
- MobiAgent / AgentRR：`https://github.com/IPADS-SAI/MobiAgent`
- MobiAgent 论文：`https://arxiv.org/abs/2509.00531`
- BEIR 官方实现：`https://github.com/beir-cellar/beir`

若论文附件、官方仓库与第三方说明不一致，以论文附件和官方仓库为准。
