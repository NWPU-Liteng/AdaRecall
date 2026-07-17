# AdaRecall 方法一：Codex 实现与实验执行指南

> 文档用途：本文件用于直接指导 Codex 在现有 `agent-memory-profiling` 项目中完成方法一的实现闭环、可靠性实验和与方法二的接口交付。
>
> 方法一正式名称：**预算约束的访问基超图编译**（Budget-Constrained Access-Basis Hypergraph Compilation）。
>
> 核心边界：方法一优化 flat retriever 与最终 cross-encoder reranker 之间的 candidate frontier construction。它学习可复用的高阶访问基、生成预算约束的超边访问调度，并把调度编译成紧凑 Memory Tile。方法一不替代现有 BGE retriever、cross-encoder 或最终 Agent 推理，也不在没有方法二实测/仿真结果时声称 PIM 加速。

---

## 0. Codex 必须遵守的执行原则

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

### 0.2 已知 GPU 限制

前序 motivation 实验已经记录：在受管环境中设置 `CUDA_VISIBLE_DEVICES=<idx>` 曾导致 `torch.cuda.is_available()` 变为 `False`；不设置该变量并使用 `LOCAL_QWEN_CUDA_DEVICE=6` 可以正常工作。因此 Codex 必须：

1. 默认先取消 `CUDA_VISIBLE_DEVICES`；
2. 使用显式逻辑设备参数，例如 `--device cuda:6`；
3. Qwen 相关脚本沿用 `LOCAL_QWEN_CUDA_DEVICE=<idx>`；
4. 不得机械写死 GPU 6。先检查该设备存在、空闲且显存足够；若更换设备，必须记录原因和设备号；
5. 每个正式任务启动前验证 PyTorch、模型和张量确实位于同一 GPU。

推荐预检模板：

```bash
cd /data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling
unset CUDA_VISIBLE_DEVICES
export ADARECALL_GPU_ID=6
export LOCAL_QWEN_CUDA_DEVICE=${ADARECALL_GPU_ID}

conda run -n agent-memory-profiling python - <<'PY'
import os
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_count", torch.cuda.device_count())
print("requested_device", os.environ.get("ADARECALL_GPU_ID"))
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is unavailable; do not start a formal run")
idx = int(os.environ["ADARECALL_GPU_ID"])
if idx >= torch.cuda.device_count():
    raise SystemExit(f"FATAL: cuda:{idx} does not exist")
print("device_name", torch.cuda.get_device_name(idx))
x = torch.ones(1024, device=f"cuda:{idx}")
print("tensor_device", x.device, "sum", x.sum().item())
PY

nvidia-smi -i "${ADARECALL_GPU_ID}"
```

任何一项失败都必须停止，不能自动退回 CPU 继续正式实验。开发期的小型单元测试可以在 CPU 上运行，但必须标记为 `dev_only`。

### 0.3 数据与结果保护

1. 不得修改 motivation v3/v4 的任何已有文件。
2. 方法一必须创建新的版本目录，例如：

   ```text
   versions/YYYYMMDD_method1_access_basis_v1
   ```

3. 复用 motivation 文件时使用只读路径或相对符号链接，并生成 SHA256 清单。
4. 开发结果和正式结果严格分开：

   ```text
   runs/dev/
   runs/formal/
   ```

5. 不得覆盖已有正式运行。每次运行使用独立 `run_id`，配置、日志、Git commit、环境、模型哈希和输出必须一一对应。
6. 所有正式汇总必须能够从原始逐查询结果重新生成，不允许手工编辑 CSV 中的结果数字。

---

## 1. 方法一的科学问题与统一目标

### 1.1 要解决的问题

已有 motivation 已经证明：

- flat frontier 从 8 扩展到 64 时，selection 和 reranking 成本显著增加；
- chronological layout 下，最终高价值记忆分散在多个 tile，存在明显 access amplification；
- 较早访问中存在可迁移到后续访问的非随机共访问结构，但完整节点复用和不同阶段的复用强度有限；
- 当前证据只支持“存在结构化优化空间”，不支持超图优于图、最终布局有效或 PIM 已获得加速。

因此方法一不能假设历史 top-5 集合会完整重复。它应从低完整复用率的历史访问集合中学习可组合的公共子结构，即“访问基超边”。

### 1.2 数据模型

设历史 invocation 数为 (N)，可访问记忆数为 (M)。构建二值访问矩阵：

\[
A\in\{0,1\}^{N\times M},\qquad
A_{qi}=1
\]

表示记忆 (m_i) 属于 invocation (q) 的有效访问集合。

学习：

\[
H\in\{0,1\}^{R\times M},\qquad
Z\in\{0,1\}^{N\times R}
\]

其中 (H) 的每一行是一条访问基超边，(Z_{qr}=1) 表示 invocation (q) 激活第 (r) 条访问基。Boolean 重构为：

\[
\hat A_{qi}=\bigvee_{r=1}^{R}(Z_{qr}\land H_{ri}).
\]

### 1.3 不使用人工线性加权，使用约束和词典序目标

禁止将方法核心实现为：

```text
w1 * semantic_similarity
+ w2 * coaccess_frequency
+ w3 * recency
- w4 * tile_cost
```

方法一使用以下词典序目标：

1. 首先最小化约束违反：证据/参考覆盖不足、overfetch 超预算、tile 超容量、复制超预算；
2. 在可行解中最小化 held-out/replay 的 tile activation；
3. 再最小化访问基模型的 MDL 描述长度；
4. 最后最小化元数据字节数、编译时间和更新代价。

实现中定义统一 `ObjectiveTuple`：

```python
ObjectiveTuple(
    coverage_violation,
    overfetch_violation,
    capacity_violation,
    replica_violation,
    replay_tile_activations,
    mdl_bits,
    metadata_bytes,
    update_cost,
)
```

所有候选超边加入、删除、tile 移动和复制操作都通过词典序比较接受或拒绝。可以在内部使用拉格朗日乘子求解子问题，但乘子不能被描述成方法的语义特征权重，最终解必须满足原始硬约束。

---

## 2. 与前序 motivation 的强制数据接口

### 2.1 只读输入包

主要输入（动机实验已独立归档）：

```text
/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/motivation_experiments/20260705_method1_motivation_medmemorybench_v4_completion
```

继承的 v3 源包：

```text
/data2/user25263093/userliteng/26_code/preTest/agent-memory-profiling/versions/motivation_experiments/20260704_method1_motivation_medmemorybench_v3
```

必须优先复用以下文件，而不是重新生成一个不一致的数据版本：

- `aggregate/B_stage_trace_v4.jsonl`
- `aggregate/B_chronological_split_manifest.csv`
- `20260704_method1_motivation_medmemorybench_v3/aggregate/M2_frontier_budget_invocations.jsonl`
- `20260704_method1_motivation_medmemorybench_v3/aggregate/M2_frontier_budget_summary.csv`
- `20260704_method1_motivation_medmemorybench_v3/aggregate/M3_tile_scattering.csv`
- `20260705_method1_motivation_medmemorybench_v4_completion/aggregate/C_keypoint_alignment_rows.jsonl`
- `20260705_method1_motivation_medmemorybench_v4_completion/aggregate/C_strict_evidence_budget_summary.csv`
- v3/v4 的 memory/query JSONL、schema、config、dataset hash 和 runtime manifest

如果某个复用文件实际位于 v3 并由 v4 符号链接引用，应保留这一关系并记录真实源路径。

### 2.2 必须冻结的前序设置

- 数据集：官方英文 MedMemoryBench 本地副本；
- persona：20；
- query：1,939 个总查询，motivation 正式子集 240 个；
- memory item：42,034 个；
- embedding：`bge-base-en-v1.5` 的同一权重；
- reranker：`cross-encoder-ms-marco-MiniLM-L6-v2` 的同一权重；
- 核心 frontier：(K_f=64)；
- 最终 selected：(K_o=5)；
- persona 内按时间最早 60% / 最晚 40% 划分；
- M3 tile capacity：8、16、32；
- 因果约束：不能访问未来 session、同 session 的未来 turn 或其他 persona 的私有记忆。

### 2.3 stage 的使用边界

`B_stage_trace_v4.jsonl` 中的四个 stage invocation 可以作为访问矩阵的历史行，但：

- stage 只能作为数据来源标识和事后分层诊断字段；
- 方法一不得建立 task/phase/module 多视图；
- stage 不得进入访问基学习或在线超边选择的特征；
- 同一原始 query 的所有 stage 行必须进入同一个 split，统计时以原始 query 为聚类单位，不能把四行当作四个独立样本夸大显著性。

### 2.4 正式开始前的 motivation 一致性哨兵

Codex 必须编写 `scripts/00_verify_motivation_contract.py`，至少核对：

- M2 中 (K_f=8,16,32,64) 的 `n_reranked` 确实分别为 8、16、32、64；
- (K_o=5)；
- M2 预算均值能够在浮点容差内复现 58.55/97.91/143.77/229.23 ms 的 selection 趋势和 0.554/0.720/0.858/1.000 的 `RerankerOracleOverlap@5`；
- M3 capacity 16 chronological 的 mean activated tiles 约为 4.21，access amplification 约为 13.48；
- held-out aggregate seen-node coverage 约为 0.466，stage advantage 约为 0.1392；
- chronological split 中不存在 persona 泄漏和时间倒置；
- v4 claim-boundary 和 validation 文件存在。

这里不是要求重新运行模型，而是验证方法一读到的确实是与 motivation 一致的输入。哨兵失败时禁止启动正式方法实验。

---

## 3. 建议工程目录与模块责任

Codex 先检查已有项目风格；若没有同类结构，创建：

```text
versions/YYYYMMDD_method1_access_basis_v1/
├── README.md
├── configs/
│   ├── data.yaml
│   ├── basis.yaml
│   ├── scheduler.yaml
│   ├── layout.yaml
│   ├── baselines.yaml
│   ├── formal_eval.yaml
│   └── hardware_contract.yaml
├── adarecall_m1/
│   ├── contracts.py
│   ├── data_loader.py
│   ├── split.py
│   ├── access_matrix.py
│   ├── mdl.py
│   ├── boolean_cover.py
│   ├── column_generation.py
│   ├── access_basis.py
│   ├── edge_connectivity.py
│   ├── submodular_scheduler.py
│   ├── schedule_unfolding.py
│   ├── tile_hypergraph.py
│   ├── multilevel_partition.py
│   ├── replication.py
│   ├── replay.py
│   ├── incremental.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── statistics.py
│   └── schemas.py
├── scripts/
│   ├── 00_verify_environment.py
│   ├── 00_verify_motivation_contract.py
│   ├── 01_build_access_matrix.py
│   ├── 02_run_feasibility_gate.py
│   ├── 03_learn_access_basis.py
│   ├── 04_build_edge_connectivity.py
│   ├── 05_generate_schedules.py
│   ├── 06_compile_tiles.py
│   ├── 07_replay_layout.py
│   ├── 08_run_baselines.py
│   ├── 09_run_ablation.py
│   ├── 10_run_formal_evaluation.py
│   ├── 11_run_incremental_evaluation.py
│   ├── 12_export_method2_contract.py
│   └── 99_validate_package.py
├── tests/
├── artifacts/
│   ├── access_matrix/
│   ├── basis/
│   ├── connectivity/
│   ├── schedules/
│   ├── layout/
│   └── method2_contract/
├── aggregate/
├── figures/
├── reports/
├── manifests/
├── logs/
└── runs/
    ├── dev/
    └── formal/
```

脚本只做参数解析和任务编排，核心算法放在 `adarecall_m1/`，并用单元测试覆盖。不得把全部逻辑堆进一个实验脚本。

---

## 4. Phase A：环境、数据和标签审计

### 4.1 生成运行清单

`00_verify_environment.py` 必须输出：

- UTC 与本地时间；
- hostname、OS、CPU、内存；
- GPU 型号、GPU UUID、driver、CUDA runtime、显存；
- `nvidia-smi` 中的其他活跃进程；
- Python、PyTorch、NumPy、FAISS、sentence-transformers、sklearn 等版本；
- Conda 环境显式路径；
- Git commit、当前 diff；
- 模型路径和 SHA256/可验证 revision；
- dataset 和 motivation 输入哈希；
- `CUDA_VISIBLE_DEVICES`、`LOCAL_QWEN_CUDA_DEVICE` 和实际张量设备。

写入：

```text
manifests/runtime_manifest.json
manifests/conda_explicit_before.txt
manifests/input_hashes.sha256
```

### 4.2 明确三类监督信号

不要把所有 top-5 都称为 gold evidence：

1. **主要结构监督：** (K_f=64) reranker 后的 top-5；称为 `effective_access_set` 或 `reranker_reference_set`。
2. **保守 session 级证据：** motivation 已有的 session anchor；用于 secondary robustness，不作为精确 gold。
3. **严格 keypoint 子集：** 仅 8/240，作为 appendix robustness；必须称为 deterministic strict source-keypoint-aligned subset，不能称为 human-verified ground truth。

访问矩阵每行必须保存：

```text
invocation_id
original_query_id
persona_id
query_time/session_order
source_stage_for_audit_only
seed_memory_ids
reranked_frontier_ids
effective_access_ids
eligible_memory_count
split
source_artifact
```

### 4.3 划分规则

1. 严格复用 v4 的 persona 内 chronological 60/40 train/test manifest；
2. 在 60% train 内再按时间划分早期 80% 为 `fit`、后期 20% 为 `calibration`；
3. `test` 在全部模型、预算、阈值和候选生成策略冻结前不可用于调参；
4. 同一 original query、同一 session 的相关行不得跨 split；
5. 若 split 修正是必要的，保留原 manifest，输出差异报告并解释，不能静默重划分。

---

## 5. Phase B：先执行可行性闸门，不要直接大规模实现

该阶段回答：完整集合复用低时，是否仍存在能迁移到未来的高阶访问基，以及其收益是否足以支撑后续 PIM。

### 5.1 训练期候选访问基

只用 `fit` 数据构建：

- 单例隐式因子；
- 二元因子；
- 大小 (ge 3) 的高阶候选因子；
- 超边连接图：两条访问基共享至少一个记忆时相连。

高阶候选不能仅由固定共现阈值决定。使用 residual/MDL 列生成的简化实现，计算每个候选加入后是否降低完整编码长度并满足 overfetch 约束。

### 5.2 在 calibration 上冻结以下选择

- 最大搜索半径 (R\in\{0,1,2\})；
- evidence miss 预算 (epsilon)；
- overfetch 预算 (delta)；
- frontier 预算 (K\in\{8,16,32,64\})；
- tile capacity (C\in\{8,16,32\})；
- 是否允许少量边界 header 复制及其字节预算；
- 列生成 beam width、停止容差和最大迭代数；
- GPU batch size。

选择依据必须是预先声明的词典序目标，不得观察 test 后选择“最好看”的配置。

### 5.3 闸门指标

在冻结配置后，只运行一次 held-out test，输出：

1. seed 的 0-hop reference/evidence coverage；
2. 1-hop 和 2-hop coverage；
3. basis reconstruction rate：held-out 集合可由训练访问基组合覆盖的比例；
4. candidate amplification：每新增一个被覆盖目标引入多少额外候选；
5. tile activation 和 access amplification；
6. 因子大小分布、支持查询数和跨时间稳定性；
7. 高阶因子贡献：移除 (|h|\ge3) 后 coverage/cost 的变化；
8. graph 限制模型与 hypergraph 模型在相同存储、候选和 coverage 预算下的目标差；
9. 未见节点、未见 session 和低支持查询的 fallback 比例。

### 5.4 Go/No-Go 规则

只有同时满足以下条件才继续完整方法一：

1. 高阶模型相对 (|h|\le2) 的 graph 限制模型，在 matched quality 和 matched metadata budget 下具有正的 held-out 物理成本收益，且以 query 为单位的配对置信区间不跨 0；
2. 1/2-hop 带来的 coverage 增量没有被更大的 candidate/tile amplification 完全抵消；
3. 训练访问基对 test 的覆盖不是仅由单个 persona 或仅由 Plan stage 驱动；若收益集中，应明确限定适用范围；
4. 根据方法二硬件契约计算的 break-even 条件存在可行区间：

   \[
   T_{host\_saved}+\frac{Bytes_{saved}}{BW_{host}}
   >T_{dispatch}+T_{PIM}+T_{merge}.
   \]

若失败，生成 `reports/FEASIBILITY_GO_NO_GO.md` 并停止扩大实现。不得通过更换 test split、删除困难 persona 或增加人工权重“挽救”结论。

---

## 6. Phase C：算法 1——物理约束的 Boolean 访问基学习

### 6.1 MDL 目标

使用：

\[
\min_{H,Z} L(H)+L(Z)+L(A\mid Z\odot H)
\]

满足：

\[
\mathrm{MissRate}(A,\hat A)\le\epsilon,
\qquad
\mathrm{OverfetchRate}(A,\hat A)\le\delta,
\qquad
|h_r|\le S_{max}.
\]

必须实现并写清 MDL 编码方案，而不是调用库后只输出一个黑盒 loss。至少包括：

- 编码超边数量和每条超边大小；
- 编码超边成员，例如 `log2(comb(M, |h|))` 或等价可复现实码；
- 编码每行激活的因子集合；
- 编码 false negative 和 false positive 残差；
- 隐式单例 fallback 的编码成本；
- 超边 incidence 元数据字节数。

### 6.2 稀疏表示

访问矩阵非常稀疏，不得构建不可控的 (N\times M) dense float 矩阵。推荐：

- CSR/CSC 保存访问矩阵；
- 每个 memory 的 invocation support 使用 RoaringBitmap、Python int bitset 或经过基准验证的等价位图；
- 超边成员使用有序整数数组；
- 缓存 support intersection；
- 训练时只对 fit 中可见且因果可访问的节点建候选，未见节点走显式 fallback。

### 6.3 残差驱动列生成

伪代码：

```text
Input: sparse access matrix A, budgets epsilon/delta, current layout P (optional)
H <- implicit singleton basis
best <- Evaluate(H)

repeat:
    Z <- ConstrainedBooleanCover(A, H, epsilon, delta)
    R <- PositiveResidual(A, BooleanProduct(Z, H))
    C <- GenerateColumnsFromResidualBitsets(R)

    for candidate h in C:
        Z_h <- IncrementalCoverUpdate(A, H union {h})
        obj_h <- ObjectiveTuple(H union {h}, Z_h, P)

    h_star <- lexicographically best feasible candidate

    if obj(h_star) is not strictly better than best:
        break

    H <- H union {h_star}
    H <- RemoveDominatedAndRedundantFactors(H)
    best <- Evaluate(H)

return H, Z, residuals, optimization_trace
```

候选生成至少实现两种并用于消融：

1. 从未解释 residual 中的高支持 item pair 开始，按 support intersection 生长；
2. 从当前查询覆盖中反复共同出现的因子做受约束合并。

候选扩展的接受条件是完整 `ObjectiveTuple` 改善，不是“共现次数超过阈值”。阈值只能用于搜索剪枝，并且必须做敏感性分析。

### 6.4 Boolean cover 子问题

给定 (H)，每个 invocation 求满足 coverage/overfetch 约束的最小因子覆盖。实现贪心近似和小规模精确求解器：

- 小规模 synthetic/抽样问题使用 ILP 或穷举得到 optimum；
- 正式大规模使用贪心 set cover；
- 报告贪心相对小规模 optimum 的 gap；
- 因子的 marginal gain 只计算新覆盖目标和新增 overfetch，不使用 stage 或手工语义权重。

### 6.5 正确性测试

至少包括：

- Boolean product 与 dense reference 一致；
- MDL 各项可独立重算；
- 新增被接受因子使目标严格下降；
- 删除冗余因子不破坏约束；
- graph 限制 `max_edge_size=2` 确实没有高阶因子；
- 同一 seed/config 重跑产生相同结构哈希；
- fit 中未来记忆被屏蔽；
- test 信息没有参与候选列生成。

---

## 7. Phase D：算法 2——预算约束的超边级访问调度

### 7.1 根超边与局部子图

新查询先使用冻结的 BGE 检索器得到种子集合 (S_q)。根超边为：

\[
E_q^{(0)}=\{h\in H\mid h\cap S_q\ne\varnothing\}.
\]

在超边连接图上构建至多 (R) 跳的局部子图。连接图节点是超边；共享记忆时相连。任何被选择的远端超边都必须把从根到它的连接路径计入访问成本，不能“瞬移”到远端超边。

### 7.2 从训练数据估计激活概率

不得用 test 标签给在线调度打分。对访问基 (h)，仅使用 fit 统计其在给定种子交叠签名下被有效激活的后验概率。使用 Beta-Binomial 平滑，至少考虑：

- 根种子与 (h) 的交叠数量；
- 根种子的 rank bin；
- (h) 的历史激活 support；
- 不使用 stage/task/module 标签。

在 calibration 上报告 Brier score、校准曲线和 ECE；若后验不可校准，退化为仅依赖集合覆盖的确定性调度，并如实报告。不得额外叠加一串人工特征权重。

### 7.3 次模集合效用

对候选超边集合 (mathcal S)，定义：

\[
F_q(\mathcal S)=
\sum_{m\in U_q}
\left[
1-\prod_{h\in\mathcal S:m\in h}
(1-\hat p_q(h))
\right].
\]

该效用对重复覆盖表现出边际收益递减。求解：

\[
\max_{\mathcal S\subseteq E_q}F_q(\mathcal S)
\]

满足：

\[
\mathrm{Tiles}_P(\mathcal S)\le B_T,
\qquad
\left|\bigcup_{h\in\mathcal S}h\right|\le K.
\]

实现：

1. 小规模局部子图做精确枚举，用于正确性和 optimality-gap 检查；
2. 正式运行采用“少量根组合枚举 + marginal gain/incremental tile cost”的预算次模近似；
3. 每次加入超边时重算新增 tile、候选和路径成本；
4. 达到预算后停止；
5. 按连通路径和最小工作集原则展开访问顺序。

只有在满足标准单 knapsack、固定 item cost 的实验子问题中才能引用 (1-1/e) 近似保证。包含连通路径和多预算的完整工程版本不得声称同样的全局保证，必须报告其对小规模 exact optimum 的 gap。

### 7.4 Memory-friendly schedule

吸收 Octopus 的 schedule-based compact partitioning 思路：

- 在已选超边诱导子图中构建辅助连接图；
- 比较不同根和 chain unfolding 顺序的探索半径、峰值 working set 和新增 tile 数；
- 选择最小化 `(exploration_radius, peak_working_set, tile_activations)` 的词典序调度；
- 生成每一步的 `frontier_before/frontier_after/new_tiles/new_candidates` trace。

此处优化的是访问顺序，不重新引入 stage view。

### 7.5 安全 fallback

当出现以下情况时，必须回退到现有 flat frontier，而不是返回空结果：

- 种子没有命中任何已学习访问基；
- 全部访问基都违反 overfetch/tile 预算；
- 新记忆比例过高；
- posterior 不可靠；
- 局部子图损坏或索引版本不一致。

逐查询结果必须记录 `fallback_reason`，并将 fallback 成本计入总体平均值，不能只报告成功命中查询。

---

## 8. Phase E：算法 3——调度感知的 Memory Tile 编译

### 8.1 训练调度转为布局超图

对 fit/calibration 中每个查询生成冻结调度：

\[
\pi_q=(h_{q1},h_{q2},...,h_{ql}).
\]

构建布局超图：

- 顶点：memory header、hyperedge header 和 incidence record；
- net：一次查询访问调度的前缀/紧凑工作集；
- net 权重：训练期该调度模式的真实频次；
- stage 不参与 net 构建；
- payload 不放入常规 header/relation tile。

分区目标：

\[
\min_P\sum_q(\lambda_P(\pi_q)-1)
\]

满足：

\[
\mathrm{size}(T_j)\le C,
\qquad
\mathrm{ReplicaBytes}(P)\le B_R.
\]

其中 (lambda_P(pi_q)) 是完成调度需要跨越的 tile 数。

### 8.2 多层分区

实现或集成多层超图分区：

1. **Coarsening：** 收缩经常出现在相同调度前缀中的记录；
2. **Initial partition：** 在容量平衡约束下生成 k-way 初始布局；
3. **Uncoarsening：** 逐层展开；
4. **FM refinement：** 以真实 replay tile activation 作为增益，移动边界记录；
5. 每次移动后检查容量、因果可访问性和索引一致性。

若使用 Mt-KaHyPar 或其他外部分区器：

- 固定版本并记录命令行和 seed；
- 将其作为通用求解骨架，不把“调用分区器”本身当成论文创新；
- 仍需实现 AdaRecall 的 schedule-net 构造、真实 replay cost 和自定义 refinement；
- 若依赖不可用，不得静默换成 k-means，应报告并先实现可验证的小规模 reference partitioner。

### 8.3 有预算的边界复制

将每个可能的 header/incidence 副本视为一个集合覆盖 item：它可以消除一组跨 tile 调度访问。求解：

\[
\max_{R'\subseteq R}\mathrm{EliminatedActivations}(R')
\]

满足：

\[
\mathrm{Bytes}(R')\le B_R.
\]

采用 budgeted maximum coverage 贪心，并和小规模精确解比较。Payload 默认不复制；若方法二需要评估 payload cache，必须作为独立消融，不与 header replication 混合。

### 8.4 联合交替优化

完整离线编译循环：

```text
H <- LearnAccessBasis(A, initial_physical_cost)
P <- ChronologicalOrInitialPartition(H)

repeat:
    Pi <- GenerateBudgetedSchedules(H, P, fit/calibration queries)
    P_new <- MultilevelSchedulePartition(H, Pi)
    H_new <- PhysicalAwareColumnRefinement(A, H, P_new)

    old_obj <- Replay(H, P, Pi)
    new_obj <- Replay(H_new, P_new, RegenerateSchedules(...))

    if new_obj is lexicographically better:
        H, P <- H_new, P_new
    else:
        break

return H, P, Pi, compiler_trace
```

不得在 test 上执行联合编译或 physical-aware column refinement。test 只能使用冻结的 (H,P) 做一次正式 replay。

---

## 9. Phase F：增量更新

新 memory 初始进入 Delta Zone，并使用隐式单例访问；新 invocation 作为访问矩阵的新行追加。维护：

\[
\mathrm{ResidualRate}=
\frac{\sum_q\|A_q-\hat A_q\|_0}{\|A\|_0}.
\]

只有在预注册的 coverage/overfetch 约束被破坏时触发局部更新：

1. 在新增 residual 上执行局部列生成；
2. 只更新受影响超边及连接图；
3. 只重放相关查询；
4. 对受影响 tile 做局部 FM refinement；
5. 当累计未来节省超过重编译开销时提交新版本；
6. 保留旧索引直到新版本原子发布，运行中查询不能看到半更新结构。

增量实验至少模拟：1%、5%、10%、20% 新记忆比例，以及按真实时间顺序逐批到达。报告 update latency、额外元数据、fallback rate、布局老化和 amortization horizon。

---

## 10. Baselines 与消融

### 10.1 强制 baseline

在完全相同 retriever、reranker、split、candidate budget 和 tile capacity 下比较：

| ID | Baseline | 目的 |
|---|---|---|
| B0 | Chronological layout + flat retrieval | 对齐 motivation M2/M3 |
| B1 | Random-per-persona layout | 检查收益不是偶然顺序效应 |
| B2 | Semantic clustering layout | 比较内容相似布局 |
| B3 | Pairwise co-access graph，限制基数不超过 2 | 直接检验高阶超图必要性 |
| B4 | Direct episode hyperedges | 检验“每次 top-5 直接建边”的低复用问题 |
| B5 | Access-basis BMF，不做 schedule-aware layout | 分离逻辑结构收益 |
| B6 | Access-basis + online scheduling，chronological layout | 分离调度收益 |
| B7 | Access-basis + generic hypergraph partition | 检验自定义 schedule nets/refinement |
| Full | Access-basis + budgeted schedule + schedule-aware tiles | 完整方法一 |

可选加入缓存/LRU、普通图 partitioner、oracle per-query packing lower bound，但 oracle 必须明确标为不可实现下界，不能作为正常系统 baseline。

### 10.2 强制消融

- 无 MDL，自定义固定超边数；
- 无 physical-aware refinement；
- 最大超边大小 1、2、3、4、无限制；
- $R=0,1,2$；
- 无 posterior、无次模去冗余；
- cardinality schedule vs memory-friendly schedule；
- 无复制 vs header replication；
- 单轮编译 vs 联合交替优化；
- 不同 tile capacities；
- 不同 frontier budgets；
- 包含/排除 stage 字段的泄漏检查：正式模型必须与排除 stage 的结果一致。

---

## 11. 指标体系

### 11.1 质量/覆盖指标

主要指标：

- `RerankerReferenceOverlap@5`：与冻结的 $K_f=64$ reranker top-5 比较；必须称为 reranker reference fidelity，不称为答案质量；
- `ReferenceRecall@K`；
- `ReferenceCompletion@K`；
- fallback-inclusive coverage。

次要指标：

- session-level Evidence Recall/Completion；
- strict keypoint subset 的 recall/completion，仅作为 appendix；
- 若未来补充人工证据标注，单独报告并保留 annotation protocol。

### 11.2 算法指标

- MDL total bits 及各分量；
- factor rank $R$；
- 因子大小、support、生命周期分布；
- 高阶因子比例和高阶因子覆盖贡献；
- held-out reconstruction miss/overfetch；
- 0/1/2-hop reachability；
- exact/approx gap；
- 列生成迭代数、编译时间和峰值内存；
- graph-vs-hypergraph objective gap。

### 11.3 系统与布局指标

- candidate count、candidate amplification；
- activated tiles/query；
- access amplification；
- header/relation/payload bytes read；
- cross-tile transitions；
- peak working set；
- metadata bytes、replica bytes；
- host selection、frontier construction、rerank 和端到端时延；
- p50/p95/p99；
- fallback rate；
- incremental update latency 和 amortized cost。

### 11.4 PIM 可行性接口指标

方法一只输出操作计数和访问轨迹：

- ContextMatch 次数；
- bitmap/filter 操作数；
- hyperedge expand 次数；
- incidence records read；
- local dedup 输入/输出数；
- local top-k 输入/输出数；
- tile/row/bank bytes；
- Host↔PIM command 和返回字节；
- cross-bank merge 数；
- payload fetch 数。

没有方法二的硬件执行或校准仿真时，这些不能转换成“PIM speedup x 倍”。

---

## 12. 实验可靠性协议

### 12.1 预注册与冻结

在正式 test 前生成只读：

```text
manifests/formal_protocol.json
manifests/frozen_config.sha256
reports/FORMAL_EXPERIMENT_PROTOCOL.md
```

写明：primary metric、primary baseline、split、预算、随机种子、停止条件、统计检验、排除规则和失败处理。正式 test 后不得修改同一个 protocol；任何修改必须新建 run/version 并解释。

### 12.2 随机性

- 训练/分区随机算法至少使用 5 个预注册 seed；
- 完全确定性算法做两次独立重跑并比较结构/结果哈希；
- 固定 Python、NumPy、PyTorch、FAISS 和 partitioner seed；
- correctness run 可启用 deterministic kernels；性能 run 可以使用正式高性能内核，但必须验证输出集合与 correctness run 一致或差异在预注册容差内。

### 12.3 统计单位与检验

- 主要统计单位是 original query；同一 query 的 stage 行作为簇；
- 使用 paired query-level bootstrap，至少 10,000 次；
- 以 persona 为簇的 bootstrap 作为敏感性分析；
- graph vs hypergraph、baseline vs Full 使用配对 randomization/permutation test；
- 多个次要比较使用 Holm 校正；
- 同时报均值/中位数、95% CI、绝对差和相对差；
- 不能只报 p-value；
- 不能把 960 stage rows 当作 960 个独立 query。

### 12.4 GPU 性能测量

1. 正式计时前模型和索引预热；
2. GPU kernel 前后 `torch.cuda.synchronize(device)`；
3. 同时报 wall-clock 和 CUDA event time；
4. embedding、ANN、reranking、frontier expansion、payload materialization 分段计时；
5. 编译时间和在线时间分开；
6. 记录 batch size、精度、显存峰值、GPU utilization；
7. 正式时避免同 GPU 的其他重负载进程；无法独占时记录干扰并重复；
8. 每个配置至少进行足以稳定估计尾延迟的重复测量；优先对完整 query 顺序做多轮运行，而不是对单个 query 连续微基准；
9. 第一轮冷启动单独报告，不与 steady-state 混合；
10. 不从模拟 tile count 推导实际 GPU/PIM latency。

### 12.5 泄漏和完整性审计

必须自动检查：

- fit/calibration/test query ID、session 和时间边界；
- 所有访问的 memory timestamp 不晚于 query；
- persona 一致；
- test 行没有参与因子、posterior、partition 或阈值学习；
- answer/gold answer 文本没有进入结构特征；
- strict keypoint 对齐没有使用 reranker 输出反向构造；
- baseline 与 Full 使用相同 retriever seeds；
- fallback 查询保留在总体结果中；
- 失败/超时查询保留并报告，不能静默删除。

### 12.6 结果验证

`99_validate_package.py` 至少验证：

- 配置、manifest、原始逐查询结果、汇总和图均存在；
- 汇总可从逐查询文件重算；
- 每个 baseline/query/config 行数一致；
- 所有预算真实生效；
- 覆盖率范围合法；
- tile 容量和复制预算无违反；
- 结构哈希和版本一致；
- test 仅运行于 frozen config；
- 主要图中的数值可追溯到 aggregate 文件；
- 报告中的 claim 不超出证据。

验证失败不得写 `COMPLETED`。

---

## 13. 方法一向方法二交付的稳定接口

方法一结束时必须生成机器可读的 `method2_contract_v1`。方法二不应重新解析方法一内部对象或 Python pickle。

### 13.1 必需文件

```text
artifacts/method2_contract/
├── CONTRACT.md
├── schema_version.json
├── hardware_assumptions.yaml
├── memory_header.parquet
├── payload_pointer.parquet
├── hyperedge_table.parquet
├── hyperedge_incidence.parquet
├── edge_connectivity.parquet
├── tile_directory.parquet
├── tile_membership.parquet
├── memory_to_tile.parquet
├── replica_map.parquet
├── query_seed_trace.parquet
├── query_schedule.parquet
├── pim_operation_trace.jsonl
├── flat_baseline_trace.jsonl
└── hashes.sha256
```

### 13.2 关键字段

`tile_directory.parquet`：

```text
layout_version, tile_id, logical_capacity, used_bytes,
header_bytes, relation_bytes, replica_bytes,
member_count, hyperedge_count, hotness,
suggested_bank_group, checksum
```

`query_schedule.parquet`：

```text
query_id, persona_id, split, layout_version,
step_id, root_edge_id, hyperedge_id, tile_id,
hop, path_parent_edge_id,
frontier_before, new_candidates, frontier_after,
new_tiles, cumulative_tiles,
header_bytes_read, relation_bytes_read,
fallback_reason
```

`pim_operation_trace.jsonl` 每一步至少包含：

```text
query_id, step_id, tile_id,
context_match_ops, bitmap_ops,
incidence_reads, hyperedge_expand_ops,
dedup_input, dedup_output,
local_topk_input, local_topk_output,
header_bytes, relation_bytes, payload_bytes,
host_command_bytes, host_return_bytes,
dependency_step_ids
```

### 13.3 方法一和方法二的职责分界

方法一负责：

- 访问基超图；
- 逻辑 tile 和记录字节；
- 查询级访问顺序；
- 操作和字节轨迹；
- fallback 与 quality 结果。

方法二负责：

- 具体 DRAM row/bank/channel 映射；
- PIM 指令或微操作实例化；
- bank 冲突和负载均衡；
- 异步 access/filter/compute pipeline；
- PIM dispatch、merge、时延和能耗；
- 真实硬件或经过校准的仿真验证。

方法一可以提供 `suggested_bank_group`，但不得把它当作已完成的物理映射。方法二改变物理映射时不能改变方法一的候选质量语义。

### 13.4 PIM 必要性的最终判定

方法一报告两组数字：

1. Host 上 frontier construction 的实测开销；
2. 可下沉操作的数量、字节和并行度。

方法二使用同一 trace 计算/测量：

\[
T_{PIM,total}=T_{dispatch}+T_{tile\_ops}+T_{merge}+T_{payload}.
\]

只有当 matched quality 下：

\[
T_{PIM,total}<T_{host,frontier}
\]

并且能耗、容量、复制和更新开销均纳入后，才可以声称 PIM 有必要和有效。若方法一已经把在线工作压缩到极小，以至 dispatch 占主导，也必须如实得出“不需要 PIM/仅大规模时需要 PIM”的适用范围结论。

---

## 14. 正式执行顺序

Codex 必须按照以下顺序推进，每一步通过验证后再继续：

```text
Step 0  远程服务器、Conda、GPU 和输入包预检
Step 1  创建只读输入清单和新版本目录
Step 2  复现 motivation contract 哨兵
Step 3  构建 stage-agnostic 稀疏访问矩阵
Step 4  运行可行性闸门
Step 5  输出 GO/NO-GO 报告
Step 6  实现并验证访问基学习
Step 7  实现并验证超边级预算调度
Step 8  实现并验证 schedule-aware tile compiler
Step 9  联合离线编译，仅在 fit/calibration 上冻结配置
Step 10 运行全部 baseline 和 ablation
Step 11 对 held-out test 做一次正式评估
Step 12 运行增量维护实验
Step 13 导出 method2_contract_v1
Step 14 执行 package validation 和 claim audit
Step 15 生成最终报告、图和复现命令
```

建议主入口：

```bash
conda run -n agent-memory-profiling python scripts/00_verify_environment.py \
  --device "cuda:${ADARECALL_GPU_ID}"

conda run -n agent-memory-profiling python scripts/00_verify_motivation_contract.py \
  --motivation-root versions/motivation_experiments/20260705_method1_motivation_medmemorybench_v4_completion

conda run -n agent-memory-profiling python scripts/02_run_feasibility_gate.py \
  --config configs/formal_eval.yaml \
  --device "cuda:${ADARECALL_GPU_ID}"
```

实际参数以项目 CLI 风格为准，但不得删除设备、配置、run ID 和输出目录等可追溯参数。

---

## 15. 正式产物和论文图表

### 15.1 报告

必须生成：

- `reports/ENVIRONMENT_AND_PROVENANCE.md`
- `reports/MOTIVATION_CONTRACT_AUDIT.md`
- `reports/FEASIBILITY_GO_NO_GO.md`
- `reports/ACCESS_BASIS_ALGORITHM.md`
- `reports/SCHEDULER_OPTIMALITY_AUDIT.md`
- `reports/LAYOUT_COMPILER_AUDIT.md`
- `reports/GRAPH_VS_HYPERGRAPH.md`
- `reports/INCREMENTAL_UPDATE.md`
- `reports/METHOD2_HANDOFF.md`
- `reports/FORMAL_RESULTS.md`
- `reports/CLAIM_BOUNDARIES.md`
- `reports/REPRODUCTION.md`
- `reports/VALIDATION.md`

### 15.2 主图建议

1. **Held-out basis reachability curve：** 0/1/2-hop coverage 与 candidate amplification；
2. **Graph vs hypergraph Pareto：** matched metadata/quality 下的 coverage–tile cost；
3. **Tile activation CDF：** chronological、semantic、graph、generic partition、Full；
4. **Frontier quality–cost Pareto：** (K=8/16/32/64) 下 reference fidelity、Host latency 和 bytes；
5. **Cross-layer breakdown：** basis、schedule、layout 各自贡献；
6. **方法二接口图：** 每查询的 tile/row/operation trace，不在方法一中绘制未经实测的 PIM speedup。

### 15.3 论文主张条件

只有正式结果满足对应证据时才能使用：

- “可复用高阶访问基存在”：高阶因子对 held-out 带来显著、非单 persona 驱动的收益；
- “超图优于普通图”：在 equal budget/matched quality 下，高阶模型优于 |h|≤2 限制模型；
- “调度感知布局有效”：相对最强 generic partition baseline，真实 held-out replay 的 tile/byte cost 更低；
- “适用于未来任务”：严格 chronological held-out 结果成立；
- “PIM 加速有效”：必须等待方法二真实/校准结果；
- “提高答案质量”：当前 motivation 不支持，除非补充独立、充分的答案质量实验。

---

## 16. Codex 完成定义

只有同时满足以下条件，方法一才算完成：

1. 所有正式实验在沙箱外服务器、`agent-memory-profiling` 环境和验证过的 GPU 上运行；
2. motivation 输入完全复用并通过一致性哨兵；
3. 数据划分无未来、persona、query cluster 泄漏；
4. 访问基学习不是固定阈值/人工加权规则，而是可重算 MDL、约束 Boolean cover 和残差列生成；
5. online 选择是预算化集合优化，并报告 exact/approx gap；
6. tile 编译使用调度超图、多层分区和 replay refinement；
7. graph、direct hyperedge、semantic layout、generic partition 等 baseline 完整；
8. 所有 fallback、失败和超时查询计入总体结果；
9. 正式统计以 original query 为单位并报告置信区间；
10. method2_contract_v1 schema、数据和哈希齐全；
11. validation 全部通过；
12. 报告明确区分已测事实、模拟成本、推断和待方法二验证的主张。

若其中任一条件未满足，Codex 应报告当前阻塞、已完成证据和下一步，不得用“逻辑已实现”代替实验闭环。
