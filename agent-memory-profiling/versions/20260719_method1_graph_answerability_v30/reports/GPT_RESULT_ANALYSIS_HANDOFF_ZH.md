# 方法一 Graph-only v30 实验结果与过程交接

更新时间：2026-07-19

版本目录：`agent-memory-profiling/versions/20260719_method1_graph_answerability_v30`

最终状态：`METHOD1_GRAPH_ONLY_OOF_CLOSED_AWAITING_FRESH`

## 0. 给后续 GPT 的任务

请基于本文和所列机器产物，重点分析：

1. 当前结果是否足以支持 Graph-only 主路径有效；
2. 质量增益、系统增益和统计不确定性应如何准确表述；
3. 模糊主题超图、层次子边、单调次模选择和 Semantic Tile 是否形成了足够清晰的算法创新；
4. 为什么 Semantic 成功、co-retrieval Access 失败、Dual 没有形成更强系统收益；
5. 下一步应如何沿现有设计优化 selector loss 和局部 compaction，而不是替换主路径；
6. 哪些结论可以进入论文，哪些只能称为 OOF 开发证据。

不要把当前状态解释成完整 fresh closure，不要隐藏 BM25 强控制，也不要把估算的 Qwen 时延描述成逐 query 实测 GPU 时延。

## 1. 实验目标和不可违反边界

目标是实现以下主路径：

```text
Query embedding
-> 只和少量 causal semantic edge descriptor 对齐
-> 激活粗主题边与细粒度 child edge
-> fuzzy semantic graph 传播
-> Graph-only 单调次模 selector 产生 Top-32
-> Qwen 对全部 32 条重排并输出 Top-5
-> Semantic / co-retrieval / Dual Tile 物理读取
-> causal incremental update
```

主方法严格禁止：

- memory-level Dense/ANN search；
- candidate query-memory embedding cosine；
- Dense score、Dense tie-break、Dense protected slots；
- 候选不足时使用 Dense fallback；
- 用当前 Query 的 Qwen 分数反向修改候选。

当前正式主方法满足：visible memory >= 32 时，Graph-owned slots 为 32/32，Dense fallback 为 0，安全违规为 0。

## 2. 数据和证据等级

- 数据：MedMemoryBench English train；
- memories：42,034；
- train queries：1,355；
- namespaces：20；
- causal snapshots：148；
- 选择协议：严格 5-fold OOF；
- paired bootstrap：10,000 次；
- train OOF 只能称为开发证据；
- development/test、LoCoMo 均已消费，未重跑；
- 当前没有读取新的 fresh gold。

六类 query 数量：

| Query type | 数量 |
|---|---:|
| entity exact match | 278 |
| inference generation | 269 |
| multi-hop clinical deduction | 116 |
| multiple choice | 289 |
| state update | 138 |
| temporal localization | 265 |

## 3. 当前核心算法配置

### 3.1 模糊层次语义图

- 动态粗主题数：`K=min(N_visible, clamp(ceil(sqrt(N_visible)), 32, 64))`；
- 每条 memory 最多属于 Top-3 主题；
- membership temperature：0.08；
- 每条 edge 3 个 medoids；
- edge degree > 192 时确定性 split；
- split children 基础值：2；
- merge cosine：0.985，release：0.975，minimum age：16；
- 高置信 query 激活 4 条粗边，低置信 query 激活 8 条；
- query-edge alignment：centroid 0.72，medoid aggregation 0.28；
- medoid 只作为 edge descriptor，不直接选择对应 memory；
- semantic propagation：restart 0.78，1 step。

### 3.2 Graph-only selector

候选池上限 256，最终选择 32 条。目标由非负单调项构成：

- graph relevance lambda：1.0；
- edge coverage lambda：0.25；
- incidence/shared-edge facility lambda：0.08；
- history weight：0.08；
- semantic-history agreement：0.04；
- recency：0.04。

facility similarity 只来自 fuzzy incidence/shared edges，不使用 candidate embedding cosine。

### 3.3 Qwen 控制

- 本地 Qwen3-Reranker-0.6B；
- 所有方法使用相同 prompt packer；
- max length 2,048；
- prefix 和 assistant generation suffix 永久保留；
- 极少数超长文档使用固定 50/50 head-tail packing；
- Qwen 分数不反馈到候选生成；
- Full/Dense/IVF/BM25/Semantic/Dual 使用相同 packing 策略。

## 4. 实验过程和关键转折

### 4.1 v27 provenance conflict

交接记录的 aggregate SHA 前缀为 `51f3a05e`，磁盘 aggregate SHA 前缀为 `09203e19`，且缺少完整 rows/source/log/commit 链路。v27 已封存为 `PROVENANCE_CONFLICT`，只保留 exploratory，不用于晋级或论文数字。

### 4.2 v28 Frontier Gate C 通过，但原 Qwen Gate D 失败

正式 Frontier run：`v28_frontier_oof_20260719_attempt003_913cd3f`。

- 1,355 queries，148 causal snapshots；
- Semantic 和 Dual 均通过 Gate C；
- 32/32 graph-owned；
- Dense fallback、underflow、安全违规均为 0；
- 原始 Qwen full-ranking 中 Semantic/Dual 均失败 Gate D。

### 4.3 v29 rank posterior 失败

执行了五折 cross-fit graph/Qwen rank posterior，Semantic/Dual 仍失败。该结果证伪了简单 rank fusion，后续不再调 alpha。

### 4.4 v30 机制诊断

逐 query 精确复现 v28 Semantic 候选：

- 1,355/1,355 candidate lists 完全一致；
- active graph pool 含 gold：508 queries；
- 最终 Top-32 含 gold：93 queries；
- 有 435 queries 的 gold 已在合法 graph pool 内，但被 selector 丢失；
- pool 内未选 gold 的 `x_sem` 中位排名为 139；
- 未选 gold 更常来自较宽、cohesion 较低的 active edge。

因此下一 selector 优化方向应是多分辨率 child edge、宽边内部覆盖和平衡选择，不应重新加入 Dense 或继续简单 Qwen rank fusion。

### 4.5 发现原 Qwen 输入有效性缺陷

原实现对完整 chat prompt 右截断：

- Semantic 43,360 个 candidate prompts 中 90.74% 超过 512 token；
- tokenizer `truncation_side=right`；
- assistant generation suffix 经常被截掉；
- final-token yes/no logits 实际落在文档中间；
- 55 个 graph-novel gold 100% 受影响；
- graph-novel gold 平均 prompt 长度约 1,032 token。

修复后使用 suffix-preserving 2,048-token packer，覆盖审计中 99.79% 的 Semantic prompts，剩余超长输入使用固定 head-tail packing。真实 CUDA smoke 和 3/3 当时专项单测通过。

### 4.6 Corrected-Qwen Gate D 通过

正式 run：`v30_corrected_qwen_oof_20260719`。

- 4 张 RTX 3090：GPU 0/1/4/5；
- runtime：3,220.86 秒；
- 8,130 rows；
- candidate identity violations：0；
- generation suffix violations：0；
- 全排序持久化。

### 4.7 Physical Tile Gate E 和端到端 Gate F 通过

正式 run：`v30_layout_io_e2e_20260719`。

- 1,355 queries，148 snapshots；
- candidate identity violations：0；
- physical byte conservation violations：0；
- padding/header/payload/embedding/metadata 全部进入真实 read range；
- physical store 在 materialize 后执行 flush + fsync；
- cold 使用 `POSIX_FADV_DONTNEED` 请求，但不声称完全绕过 OS/device cache。

### 4.8 Causal Delta-Tile Gate G 通过

正式 run：`v30_incremental_gate_g_20260719`。

- 按 arrival order 对每条新 memory 做 Top-3 在线 prototype sufficient-stat 更新；
- payload、embedding、metadata 实际追加到 delta Tile；
- 每个 query boundary 执行 flush + fsync；
- 然后执行冻结图语义的精确边界重建和 Semantic Tile 物化；
- 过去 snapshot 使用不可变 receipt 固化；
- frozen-policy Qwen Top-5 事件在当前 snapshot 固化后才写入 replay ledger；
- event 准确命名为 `co_retrieval`；
- `native_agent_access_claim=false`。

边界操作准确命名为 `full_exact_boundary_rebuild`，没有包装成局部 compaction。

## 5. Gate C：Top-32 Frontier 结果

| 方法 | Recall@32 | Group coverage@32 | vs Dense Recall | Recall 95% CI | vs Dense GroupCov | GroupCov 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Dense-HNSW-32 | 0.043296 | 0.079151 | - | - | - | - |
| Semantic Graph-only-32 | 0.051107 | 0.096371 | +0.007811 | [-0.002768, +0.018512] | +0.017220 | [+0.003075, +0.031551] |
| Dual Graph-only-32 | 0.051476 | 0.096740 | +0.008180 | [-0.002399, +0.018881] | +0.017589 | [+0.003383, +0.032103] |
| Lexical BM25-32 | 0.604674 | 0.674785 | +0.561378 | [+0.534809, +0.586963] | +0.595633 | [+0.568696, +0.622263] |

解释：Graph-only 对 Dense 的 Recall@32 是正均值但 CI 跨零；Group coverage@32 的 CI 为严格正区间。BM25 显著更强，必须一直并列。

## 6. Gate D：Corrected-Qwen Top-5 结果

| 方法 | nDCG@10 | Recall@5 | GroupCov@5 | 平均估算 Qwen 时延 |
|---|---:|---:|---:|---:|
| Full-HNSW-128 | 0.075423 | 0.085363 | 0.136777 | 7,296.36 ms |
| Dense-HNSW-32 | 0.031811 | 0.032841 | 0.059963 | 1,824.09 ms |
| Standard-IVF-32 | 0.031494 | 0.032472 | 0.060578 | 1,824.09 ms |
| Semantic Graph-only-32 | 0.037759 | 0.039176 | 0.076138 | 1,824.09 ms |
| Dual Graph-only-32 | 0.037720 | 0.039176 | 0.076138 | 1,824.09 ms |
| Lexical BM25-32 | 0.415478 | 0.483825 | 0.545203 | 1,824.09 ms |

Semantic 相对 Dense 的 paired 结果：

| 指标 | 均值差 | 95% CI | 解释 |
|---|---:|---:|---|
| nDCG@10 | +0.005948 | [-0.002982, +0.014878] | 正均值，CI 跨零 |
| Recall@5 | +0.006335 | [-0.003137, +0.015744] | 正均值，CI 跨零 |
| Precision@5 | +0.002214 | [-0.000148, +0.004576] | 正均值，CI 略跨零 |
| GroupCov@5 | +0.016175 | [+0.003381, +0.028905] | 严格正区间 |

Gate D 通过的是预冻结非劣/增益规则，不应把所有质量指标都描述成统计显著提升。

## 7. Gate E：Semantic Tile 物理局部性

候选 ID 在所有 layout 中完全相同，只改变物理排布。

| Layout | Mean Tiles | P95 Tiles | Mean bytes | Mean requested-cold | Gate E |
|---|---:|---:|---:|---:|---|
| Chronological | 16.814 | 24 | 3.940 MB | 8.352 ms | 对照 |
| Semantic | 4.480 | 8 | 1.043 MB | 2.481 ms | 通过 |
| Dual | 6.060 | 9 | 1.381 MB | 7.415 ms | 通过，但弱于 Semantic |
| Co-retrieval | 16.149 | 23 | 3.774 MB | 9.690 ms | 失败 |

Semantic 相对 Chronological：

- mean Tiles：-73.36%；
- P95 Tiles：-66.67%；
- fetched bytes：-73.53%；
- requested-cold mean latency：-70.30%；
- warm read mean：0.175 ms，对照为 0.635 ms。

Co-retrieval 只降低约 4% Tiles/bytes，cold mean 反而慢 16.02%，是必须保留的 Access 负结果。

## 8. Gate F：端到端和 Pareto

总时延包含真实 retrieval/graph 和 physical warm I/O，加上 corrected-Qwen run 导出的估算 Qwen cost。Qwen 部分不是对每个方法重新执行的独立 GPU wall-clock。

| 方法 | Retrieval/graph mean | Warm I/O mean | Qwen mean estimated | Total warm mean | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Full-HNSW-128 | 0.468 ms | 3.016 ms | 7,296.36 ms | 7,342.28 ms | 0.075423 |
| Dense-HNSW-32 | 0.468 ms | 0.675 ms | 1,824.09 ms | 1,867.67 ms | 0.031811 |
| Standard-IVF-32 | 17.362 ms | 0.622 ms | 1,824.09 ms | 1,884.51 ms | 0.031494 |
| Semantic Graph-only-32 | 34.428 ms | 0.206 ms | 1,824.09 ms | 1,901.16 ms | 0.037759 |
| Dual Graph-only-32 | 24.827 ms | 0.333 ms | 1,824.09 ms | 1,891.69 ms | 0.037720 |
| Lexical BM25-32 | 20.117 ms | 0.554 ms | 1,824.09 ms | 1,887.20 ms | 0.415478 |

Semantic：

- 比 Full-128 平均快 5,441.12 ms；
- 比 Dense-32 平均慢 33.49 ms；
- 比 IVF-32 平均慢 16.65 ms；
- 但 nDCG/Recall/GroupCov 均值均高于 Dense/IVF，因此未被其严格支配；
- Gate F 是质量-系统 Pareto 通过，不是“Graph 最快”。

Dual：

- 比 Full-128 平均快 5,450.59 ms；
- 比 Dense-32 平均慢 24.02 ms；
- 比 IVF-32 平均慢 7.18 ms；
- 系统上仍弱于 Semantic 的物理局部性。

## 9. Gate G：因果增量结果

| 指标 | 结果 |
|---|---:|
| Visible causal insertions | 30,755 |
| Query-boundary snapshots | 148 |
| OOF queries | 1,355 |
| Frozen-policy replay events | 1,355 |
| Insertion mean / P50 / P95 | 0.299 / 0.285 / 0.407 ms |
| Affected semantic edges mean | 2.948 |
| Affected delta Tiles per insertion | 1 |
| Write amplification mean / P95 | 1.021x / 1.017x |
| Delta fsync P50 / P95 | 5.61 / 109.49 ms |
| Exact rebuild mean / P95 | 1,342.98 / 2,681.13 ms |
| Semantic layout materialization mean | 49.54 ms |
| Online insertion/full rebuild ratio mean | 7.41% |
| Prototype drift median / P95 | 0.0000527 / 0.005792 |
| Formal runtime | 287.40 s |

总 physical delta writes 为 228.66 MB，logical payload+embedding 为 224.05 MB。max write amplification 1.829x 来自对齐 Tile 首条记录承担 padding，并不代表常规插入。

安全和一致性违规：

```text
snapshot hash             0
candidate identity        0
physical/layout check     0
future                    0
namespace                 0
duplicate                 0
past snapshot mutation    0
```

独立 cross-run verifier 又重建了 148 snapshots，并和 Gate E 原始 rows 对比：

- snapshot hash violations：0；
- immutable layout hash violations：0；
- 1,355 query Tile ID violations：0；
- 1,355 candidate identity violations：0。

当前增量结论：在线摄入低成本、状态影响局部、物理写入可核验、边界结果精确；但边界 compaction 仍是平均 1.343 秒的全量重建，真正的局部 exact compactor 尚未完成。

## 10. 已证伪、负结果和限制

1. v27 provenance 不完整，不能作为确认性证据；
2. 原 v28 Qwen 输入在长 prompt 上无效，旧 Gate D 失败受 truncation confound；
3. v29 简单 graph/Qwen rank posterior 仍失败，不应继续调 alpha；
4. co-retrieval layout Gate E 失败，不能把 Access 分支包装成成功；
5. Dual 通过质量和 Gate，但没有优于 Semantic 的主系统指标；
6. BM25 质量远高于 Graph-only，不能声称 Graph 是最佳检索器；
7. Semantic 对 Dense 的 nDCG@10/Recall@5 是正均值，但 CI 跨零；
8. Frontier 机制诊断显示 selector 仍丢失大量已进入 graph pool 的 gold；
9. Gate F 的 Qwen cost 为统一 packing 下的估算成本，physical I/O 才是本轮逐 query 实测；
10. requested-cold 使用 OS fadvise，不等价于保证 raw-device cache 完全清空；
11. 当前只有 train 5-fold OOF 开发证据，没有独立 fresh 结论；
12. query-boundary compaction 仍为 full exact rebuild，不是局部增量 compaction。

## 11. 当前可以支持的结论

较稳妥的结论是：

> 在严格禁止 memory-level Dense/ANN、candidate query-memory cosine 和 Dense fallback 的条件下，模糊层次 Graph-only 检索在相同 32-candidate 预算下，相对 Dense-HNSW/Standard-IVF 获得了正向最终质量均值，并显著改善 evidence group coverage；Semantic graph-aware Tile layout 同时将物理 Tile 激活数、读取 bytes 和 requested-cold latency 降低约 70%。该方法形成了相对 Dense/IVF 的质量-物理局部性 Pareto，但尚未优于强 BM25 控制，也尚未完成独立 fresh 验证和局部 exact compaction。

不应支持的过强结论：

- Graph-only 已全面优于所有传统检索；
- 所有质量指标都统计显著；
- Access/co-retrieval 分支成功；
- 已实现真实局部 compaction；
- 已获得 fresh independent confirmation；
- 已达到 SOTA。

## 12. 完成状态

| Gate | 状态 | 说明 |
|---|---|---|
| Gate C Frontier | 通过 | Semantic、Dual 均通过 |
| Gate D Qwen | 通过 | 使用修复后的统一 suffix-preserving packing |
| Gate E Physical Tile | 通过 | Semantic、Dual 通过；co-retrieval 失败 |
| Gate F End-to-end Pareto | 通过 | Graph 质量更高但比 Dense/IVF 略慢 |
| Gate G Incremental replay | 通过 | 30,755 插入，全部安全/一致性违规为 0 |
| Gate H Fresh | 未执行 | 没有合法未消费本地 holdout |

当前正式状态：

```text
METHOD1_GRAPH_ONLY_OOF_CLOSED_AWAITING_FRESH
```

## 13. Fresh 数据审计

本轮没有读取新的 gold，也没有消耗 fresh run。

- LongMemEval：已有 2026-06-16、06-21、07-03 多轮正式/动机实验；
- LoCoMo：指南明确已消费，并有多轮历史结果；
- MSC：已有正式 reranking/replay；
- MemoryCD：已有真实 candidate-frontier 和 Qwen task-quality 实验。

因此这些本地数据都不能作为“gold 从未读取”的 Gate H。需要新的兼容官方数据，先用 synthetic fixture 完成 adapter，再冻结 config/commit/manifest/read lock，最后一次性读取 gold。

## 14. 建议后续 GPT 重点提出的优化

必须沿原主路径优化，不能用 Dense 替换：

1. 针对 435 个 pool 内 selector loss，分析宽 edge 内部 child-edge coverage 和 degree-balanced marginal；
2. 设计 parent-child residual coverage，防止高分窄边重复占据 32 个 slot；
3. 在不使用 candidate embedding cosine 的前提下，改进 incidence-space facility/diversity；
4. 研究 query-confidence 自适应的 coarse/child 激活，但必须预冻结且不按 gold/query type 调参；
5. 将 full exact boundary rebuild 优化为受影响 edge 的局部 exact split/merge compaction；
6. 为局部 compaction设计“与 full rebuild snapshot/layout/candidate hash 一致”的新 Gate；
7. 取得新的未消费 holdout 后，只运行一次 Gate H；
8. BM25 应作为持续强控制，后续可以研究 Graph 与 lexical 的互补性，但不能把 lexical score偷偷加入当前 Graph-only 主结果。

## 15. 机器证据索引

### 核心配置

- `../20260718_prism_aligned_method1_graph_only_v28/configs/graph_only_main.yaml`
- `configs/qwen_suffix_preserving.yaml`
- `configs/system_gates.yaml`
- `configs/incremental_gate_g.yaml`

### 正式 runs

- Gate C：`../20260718_prism_aligned_method1_graph_only_v28/runs/v28_frontier_oof_20260719_attempt003_913cd3f/`
- Gate D：`runs/v30_corrected_qwen_oof_20260719/`
- Gates E-F：`runs/v30_layout_io_e2e_20260719/`
- Gate G：`runs/v30_incremental_gate_g_20260719/`

### 关键报告

- `reports/V28_MECHANISM_AND_QWEN_VALIDITY_AUDIT.md`
- `reports/CORRECTED_QWEN_GATE_D.md`
- `reports/LAYOUT_IO_E2E_GATES.md`
- `reports/INCREMENTAL_GATE_G.md`
- `reports/METHOD1_GRAPH_ONLY_OOF_CLOSURE.md`
- `reports/FRESH_HOLDOUT_AVAILABILITY_AUDIT.md`

### 关键收据与 SHA256

```text
v28 Frontier rows
834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98

v28 Frontier aggregate
8a618b0b958f64597e4064996126e51fb50d0ee0bf3dbf3cde9fcfe1f0f7de99

Corrected-Qwen rows
475fdb788cb4d2cf64880cafd3743f10b5f00045d89f61f6596780a076df0805

Corrected-Qwen aggregate file
ad29fd28ef585e0240876ad2f80b66ed1623be0e8ef0f6df18feffc1dba60c35

Layout/I/O rows
8813ce0f389bbb4015b1b4c69da8eaacc0b3eaa7d5be8e8a4c9beaeae284e1d0

Layout/I/O aggregate file
c1d9ffa2b3edc762feb0441309b57978790e947a784f8cc47dfec297531faf54

Incremental insertion rows
c3f507435da8da024e57bb15d895363c82ec8508c81e840a090a35e7226b8a9a

Incremental snapshot rows
f8137ce07aed99d8e68b47f6417c72cdf8c2c06a2e863034173b5faf01911e39

Incremental agent replay
f77ee41e0228310384dfe644a4dc59484fa8f61880e38cd5840cca1623c33968

Incremental aggregate file
eed0c36f50af025ce13b87dd30a0c0687de5914960ce918dbc65b995d8d864b6

Cross-run layout audit
b3316852913b892cc15bf139e12be6bcc49c6d3afea952b5d6e253133c54013e
```

### 关键 commits

```text
d95ece2  Record v28 and v29 Qwen closure failures
b76fdec  Diagnose v28 selector loss and Qwen truncation
503b5f4  Freeze suffix-preserving Qwen control implementation
658d7a3  Record corrected-Qwen Gate D promotion
685ebb1  Freeze physical Tile and end-to-end gate implementation
073f1a5  Record Method 1 physical Tile and Pareto gates
860480e  Implement causal delta Tile replay for Method 1
b692012  Freeze Method 1 incremental Gate G
0a463f4  Close Method 1 graph-only OOF gates
1d6efc8  Verify incremental layout identity across runs
```

## 16. 一句话总结

方法一已经在严格 OOF 条件下证明：Graph-only 主路径可以相对 Dense/IVF 获得正向质量均值和显著 group coverage 增益，同时通过 Semantic Tile 获得约 70% 的物理局部性收益，并完成低成本因果写入与精确回放；但 BM25 仍显著更强，边界 compaction 仍是全量重建，且缺少独立 fresh 验证。
