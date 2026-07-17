# 方法一动机实验：论文写作技术说明

## 1. 实验定位

本组实验的目标不是证明正式方法 FSA-QEHG 优于基线，而是回答一个更基础的问题：在长期 Agent memory 场景中，是否普遍存在足以支持“结构化、有界且可增量维护的 memory organization”这一研究方向的工作负载特征。

主动机链为：

```text
长期历史扩大候选处理与数据访问压力（M1）
  -> 查询呈现多组、可替代的组合 evidence demand（S1）
  -> 扩大 flat frontier 只能以快速增长的成本逼近高预算结果（M2）
  -> 被选 memory 在逻辑 Tile 上分散，产生过取与低利用率（M3）
  -> memory 持续增长时，全量重建处理大量未改变状态（S2）
```

旧版 controller-defined stage 实验不再属于主动机链。它只能作为附录中的 workload observation，不能用于推导 stage-aware retrieval 或 access-chain update，因为当前正式方法不使用 stage/workflow labels。

## 2. 研究问题

- **RQ1：历史规模压力。** 长期 memory history 是否增加 selection latency 和 host-side bytes touched？主要成本来自哪个阶段？
- **RQ2：组合证据需求。** 查询是否普遍需要多个 evidence groups？通用 flat cosine top-B 能否完整覆盖这些 groups？
- **RQ3：候选扩张权衡。** 增大 flat frontier 是否能逼近高预算 reranker reference？需要支付多少 selection cost？
- **RQ4：Tile 访问放大。** 相关 memory 在朴素布局下是否分散到多个 Tile，从而造成低利用率和 over-fetch？
- **RQ5：内容维护压力。** 对持续插入的 memory，局部 append maintenance 是否能够避免 full rebuild work，同时保持 lookup 等价？

## 3. 数据、输入与实验边界

实验使用 MedMemoryBench English。memory corpus 包含 42,034 条 turn-pair memory，原始 query 集包含 1,939 条 query。query 类型包括 entity exact match、inference generation、multi-hop clinical deduction、multiple choice、state update 和 temporal localization。

M1/M2/M3 复用 `20260705_method1_motivation_medmemorybench_v4_completion` 中已经审计的 workload aggregates；其底层 query-level invocations 来自 v3。S1 使用 1,355 条 train-only query plans、train-only evidence groups、冻结的 query/memory embeddings 和 eligible-memory constraints。S2 只读取原始 memory/query 内容。

以下输入被明确禁止：v21 calibration/test labels、v21 method predictions、正式 selector outputs、stage/workflow labels、future access、answer-generation outputs 和由 evidence groups 人为构造的 temporal/access order。

因此，本组实验属于 data/workload characterization，不属于正式方法对比；这里的 `formal` 是旧版 v4 workload run mode，不等同于 v21 被锁定的 formal-test protocol。

## 4. M1：历史规模与系统压力

### 4.1 设置

M1 构造两种历史条件：`medical_only` 与 `medical_plus_official_noise`。前者只包含医疗相关历史，后者进一步加入官方噪声内容。实验覆盖 128、256、512、1024 和 `full` 等历史条件，但两种条件并非在所有规模上都有有效样本。

`medical_only` 的有效样本量分别为：128（202）、256（161）、512（88）、full（240）；`medical_plus_official_noise` 分别为：128（225）、256（212）、512（184）、1024（136）、full（240）。

检索路径保持不变，记录 total selection、ANN 和 reranking latency。host bytes touched 定义为：

```text
metadata_bytes_read + payload_bytes_read + candidate_vectors.nbytes
```

这是一项 host-side logical traffic/footprint proxy，不是硬件总线流量或 PIM 测量。

### 4.2 指标与图形

- Panel a：两种历史条件下的 mean selection latency 和已有 95% CI。
- Panel b：mean host bytes touched 和已有 95% CI。
- Panel c：full-history 下 ANN、rerank 和 residual-other latency 的堆叠分解。

将 latency、bytes 和 stage breakdown 分面展示，可以避免双 Y 轴，同时区分“规模压力”与“瓶颈来源”。

### 4.3 可支持结论

历史扩展明显提高 host-side bytes touched；full-history selection 中，reranking 占据绝大部分 latency。这说明长期 memory 系统不能只关注 ANN lookup，还必须控制进入昂贵后处理阶段的候选规模。

M1 不支持“latency 随历史长度严格单调或线性增长”。不同规模的有效 query 组成和样本量不同，`full` 也是一个数据条件而不是 1024 后的等距数值点。

## 5. S1：组合 evidence demand

### 5.1 设置

S1 使用全部 1,355 条 train query plans：Exact match 278、Inference 269、Multi-hop 116、Multiple choice 289、State update 138、Temporal 265。

每个 query 的监督结构为无序 evidence groups：

```text
q -> {A_q1, ..., A_qG}
```

其中 `A_qg` 是第 g 个 evidence aspect 的 acceptable-memory alternatives。命中同一 group 中任一 memory 即视为该 group 被覆盖；不同 groups 表示需要组合覆盖的不同 evidence aspects。该结构不表示 temporal order、reasoning chain 或 joint physical access。

对每条 query，只在 eligible memories 中使用冻结 embedding 做 flat cosine 排序，并考察 B in {4, 8, 16, 32}。该 flat probe 是方法无关的 workload reference，不是 v21 正式 baseline run。

### 5.2 指标

```text
GroupCoverage@B
= (# evidence groups with at least one selected acceptable memory) / G

AllGroupsCovered@B
= 1, if every evidence group is covered; otherwise 0
```

同时记录 evidence-group count、acceptable-memory count、alternative count、cross-group overlap 和 acceptable alternatives 的 top-2 cosine margin。

### 5.3 数据结构与结果

1,009 条 query 只有 1 个 group，238 条有 2 个 groups，98 条有 3 个 groups，10 条有 4+ groups；因此 25.54% 的 query 具有明确的 multi-group demand。每个 group 平均约有 2 个 acceptable alternatives。

Multi-hop query 平均需要 2.69 个 groups。其 AllGroupsCovered 在 B=4/8/16/32 时约为 2%/5%/10%/28%；即使扩大到 B=32，多数 multi-hop query 仍未完整覆盖所有 evidence aspects。

### 5.4 图形设计

- Panel a：query type × B 的 AllGroupsCovered 热图，回答“完整任务需求是否满足”。
- Panel b：query type × B 的 mean GroupCoverage 热图，回答“平均覆盖了多少 evidence aspects”。
- Panel c：各 query type 的 group-count composition，证明困难不是绘图或预算构造造成，而是数据需求本身存在。
- Panel d：按 group count 聚合 AllGroupsCovered，并展示 B=8/16/32 和近似 binomial 95% CI，直接检验 composition 增长与 flat retrieval failure 的关系。

S1 可以支持“组合 evidence demand 普遍存在，且 flat top-B 难以完整覆盖”；它本身不能证明 hypergraph、FSA 或任何正式方法更优。4+ groups 只有 10 条 query，必须保留宽 CI 并避免强结论。

## 6. M2：flat frontier 的成本与 reference fidelity

### 6.1 设置

M2 使用 240 条 v4 workload query，每种 query type 40 条。对每条 query 使用相同 flat retrieval/reranking pipeline，将 frontier budget 设置为 B in {8, 16, 32, 64}，最终均输出 top-5。

B=64 的 top-5 被定义为高预算 reranker reference。对 B<64，计算其 top-5 与 B=64 top-5 的 overlap。该 reference 只衡量“低预算结果对高预算 reranker 结果的保真度”，不是人工 gold、answer accuracy 或任务成功率。

### 6.2 指标

- total selection latency 及其 95% CI；
- reranker-reference overlap@5 及其 95% CI；
- host bytes、rerank latency 和 frontier occupancy 作为审计指标。

全体 query 的 weighted mean 结果为：B=8 时 58.55 ms / 0.554 overlap，B=16 时 97.91 ms / 0.720，B=32 时 143.77 ms / 0.858，B=64 时 229.23 ms / 1.000。

### 6.3 图形设计

图采用 2x3 query-type small multiples。每个 panel 的 X 轴是 selection latency，Y 轴是与 B=64 reference 的 agreement；每个预算点同时显示 X/Y confidence intervals，并直接标注 B。

这种 Pareto 表达避免双 Y 轴，并能检验 tradeoff 是否跨 query types 一致。B=64 overlap=1 是由 reference 定义决定的，因此图的核心证据是从 B=8 到 B=32 的 cost-fidelity trajectory，而不是 B=64 端点本身。

### 6.4 可支持结论

扩大 flat frontier 确实提高对高预算 reranker result 的 fidelity，但 selection latency 也快速增长。因此，“简单扩大 flat candidate set”不能同时解决质量保真与系统成本问题，需要更有结构的 bounded candidate organization。

## 7. M3：Tile scattering 与 over-fetch

### 7.1 设置

M3 固定使用 M2 B=64 最终选择的 top-5 memories，共 240 条 query。它不改变检索结果，只将同一 selected set 映射到不同逻辑布局，以隔离 placement 对 Tile access 的影响。

Tile capacity 取 8、16、32。比较三种布局：按 persona 内 chronological rank 排列、固定随机种子的 per-persona random 排列，以及 query-specific oracle packing lower bound。oracle 假设一条 query 的 selected memories 可以被理想地打包到最少 Tile 中，只是理论下界，不是可部署 baseline。

### 7.2 指标

对 K=5 的 selected set：

```text
ActivatedTiles = number of distinct Tiles containing selected memories
IdealTiles = ceil(K / capacity)
TileUtilization = K / (capacity * ActivatedTiles)
AccessAmplification = capacity * ActivatedTiles / K
ScatterRatio = ActivatedTiles / IdealTiles
```

因为 capacity >= 8 且 K=5，三个容量下 IdealTiles 都为 1。

### 7.3 结果与图形

capacity=16 时，chronological/random/oracle 的 mean ActivatedTiles 分别为 4.21/4.90/1.00；chronological 的 useful-payload utilization 约为 7.8%，access amplification 约为 13.5x。

- Panel a：ActivatedTiles heatmap，展示访问块数量。
- Panel b：AccessAmplification heatmap，展示 over-fetch 程度。
- Panel c：useful payload utilization heatmap，展示被激活 Tile 中真正有用的数据比例。
- Panel d：capacity=16 的 query-level ScatterRatio ECDF，避免均值掩盖分布。

M3 支持“朴素布局存在显著 Tile-scattering 和 physical co-location headroom”，但不证明任何已实现 layout 优越，也不能推导 PIM latency、energy 或 hardware speedup。

## 8. S2：静态内容增量维护

### 8.1 设置

S2 对 42,034 条 memory 按 chronological metadata 构造确定性 maintenance slices。初始 base 为 80%，即 33,627 条；在完整 corpus 的 1%、5%、10%、20% growth checkpoints，cumulative appended count 分别为 420、2,102、4,203、8,407。

实验使用方法无关的 token-to-memory inverted index。文本经过 lowercase ASCII alphanumeric tokenizer，每条 memory 对每个唯一 token 增加一条 posting。比较 append-only incremental update 与从当前全部 memory 重新构建 full index。

从原始 query 中等距抽取 128 条固定 probes，每次执行 top-8 token-overlap lookup。在四个 growth checkpoints 上比较 incremental index 与 rebuilt index，共进行 512 次结果等价性检查。逻辑 payload Tile capacity 固定为 16。

### 8.2 指标与结果

- new/changed posting operations；
- full rebuild posting operations；
- incremental update 和 full rebuild wall time；
- full-to-incremental operation/time ratio；
- lookup mismatch count。

20% growth 时，当次新增 postings 约占当前 full postings 的 9.69%；当前 run 中 full rebuild wall time 是 incremental update 的 8.60 倍；所有 checkpoints 合计 0/512 lookup mismatches。

### 8.3 图形设计

- Panel a：new postings 与 full rebuild postings 的绝对工作量。
- Panel b：incremental/rebuild wall time，使用 log scale。
- Panel c：确定性 operation ratio 与 measured-time ratio；前者是主要可复现证据，后者是受机器状态影响的辅助证据。
- Panel d：正确性、20% checkpoint 数值和 scope 声明。

S2 只支持 static content insertion maintenance。chronological metadata 仅用于构造可复现 slice，不能解释为 causal Agent insertion/access order；该实验也不涉及 co-access、future access 或 dynamic reasoning chain。

## 9. 统计与可复现性说明

M1/M2 使用原 v4 aggregate 中保存的 95% confidence intervals。M3 同时报告 240 条 query 的均值和 query-level ECDF。S1 的 heatmap 使用全部 train queries；按 group-count 的 proportion 图使用近似 binomial CI，并显式标注每组样本量。S2 的 posting operations 和 lookup equivalence 是确定性指标；wall time 没有被当作硬件性能结论。

所有主结果必须同时保留 query count、query-type stratification、budget/capacity 设置和 claim boundary。不能只引用最有利的一个 query type 或单一预算点。

## 10. 论文中的推荐组织

主文建议按 M1 -> S1 -> M2 -> M3 排列：先说明长期历史产生系统压力，再说明查询需求不是单一相似度命中，随后展示 flat expansion 的成本局限，最后把结构问题连接到 Tile-granular physical inefficiency。S2 更适合附录；只有当论文明确把“static content maintenance”列为核心贡献背景时，才应进入主文。

可以将主动机段落概括为：

> Long-term agent memories create pressure at both the retrieval and data-movement levels. As histories grow, the host-side memory footprint increases and expensive reranking dominates selection latency. More importantly, a substantial fraction of queries require multiple distinct evidence groups, for which flat similarity retrieval frequently fails to cover all required aspects even at enlarged budgets. Increasing a flat frontier improves agreement with a high-budget reranker reference, but incurs rapidly rising selection cost. The selected memories are also scattered across logical Tiles under chronological or random placement, causing low payload utilization and substantial access amplification. Finally, under continued content insertion, rebuilding an index processes the entire corpus even though only a small fraction of postings changes. Together, these observations motivate a structured, bounded, and incrementally maintainable memory organization.

对应的中文表述为：

> 长期 Agent memory 同时面临检索成本和数据访问效率问题。随着历史规模扩大，host-side memory footprint 上升，且昂贵的 reranking 成为 selection latency 的主要来源。与此同时，相当比例的查询需要覆盖多个不同的 evidence groups；对于这类组合需求，flat similarity retrieval 即使扩大预算也经常遗漏部分 evidence aspects。继续增大 flat frontier 虽然能够提高对高预算 reranker reference 的保真度，却带来快速增长的 selection cost。被选中的 memories 在 chronological 或 random placement 下还会分散到多个逻辑 Tile，造成较低的 payload utilization 和显著的 access amplification。最后，在 memory content 持续增长时，full rebuild 会重复处理大量未改变状态。上述现象共同说明，需要一种结构化、有界且可增量维护的 memory organization。

## 11. Claim boundaries

本组实验允许支持：长期历史的系统压力、多组 evidence demand、flat frontier cost-fidelity tradeoff、逻辑 Tile scattering/over-fetch headroom、static content insertion 的局部维护需求。

本组实验不允许支持：v21 方法优越性、最终 Agent answer-quality 改善、stage-aware retrieval、access-order/co-access learning、causal dynamic update、真实 PIM latency/energy 或硬件 speedup。

