# 方法一动机实验数据汇总

## 1. 数据包位置

当前动机实验版本：

```text
agent-memory-profiling/versions/20260715_method1_motivation_medmemorybench_v5_alignment/
```

主要数据位于 `aggregate/`，论文级 PNG/PDF 位于 `figures/`，实验协议位于 `configs/v5_protocol.json`。

## 2. 基础数据规模

| 数据对象 | 数量 | 来源 |
|---|---:|---|
| turn-pair memories | 42,034 | v3 `memory_items_turn_pair.jsonl` |
| 原始 MedMemoryBench English queries | 1,939 | v3 `queries_medmemorybench_en.jsonl` |
| S1 train-only query plans | 1,355 | QA-basis `query_plan_train.jsonl` |
| M2/M3 workload queries | 240 | 每种 query type 40 条 |
| query types | 6 | Exact、Inference、Multi-hop、Multiple choice、State update、Temporal |

S1 的 query-type 分布：

| Query type | 数量 |
|---|---:|
| Entity exact match | 278 |
| Inference generation | 269 |
| Multi-hop clinical deduction | 116 |
| Multiple choice | 289 |
| State update | 138 |
| Temporal localization | 265 |
| 合计 | 1,355 |

## 3. M1 历史规模数据

直接绘图数据：`aggregate/v5_reused_m1_source.csv`。

### 3.1 样本量

| History condition | 128 | 256 | 512 | 1024 | full |
|---|---:|---:|---:|---:|---:|
| medical_only | 202 | 161 | 88 | - | 240 |
| medical_plus_official_noise | 225 | 212 | 184 | 136 | 240 |

### 3.2 主要均值

| Condition | Scale | Selection latency (ms) | Rerank latency (ms) | ANN latency (ms) | Host bytes (MB) |
|---|---|---:|---:|---:|---:|
| medical_only | 128 | 109.60 | 97.08 | 1.22 | 0.538 |
| medical_only | 256 | 140.70 | 126.61 | 1.77 | 0.937 |
| medical_only | 512 | 137.12 | 125.27 | 1.73 | 1.726 |
| medical_only | full | 132.29 | 118.27 | 1.59 | 1.436 |
| medical + noise | 128 | 104.22 | 92.06 | 1.26 | 0.525 |
| medical + noise | 256 | 137.01 | 123.18 | 1.73 | 0.922 |
| medical + noise | 512 | 135.73 | 123.17 | 1.71 | 1.711 |
| medical + noise | 1024 | 136.81 | 124.29 | 1.79 | 3.284 |
| medical + noise | full | 134.25 | 121.86 | 1.66 | 3.556 |

`host_memory_bytes` 的定义为 metadata bytes、payload bytes 和 candidate-vector bytes 之和。CSV 同时保存 mean、median、p95 和 95% CI。

数据表明 host bytes 随历史规模明显扩大；selection latency 不是严格单调关系。full-history 下 reranking 约占 total selection latency 的 89%–91%。

## 4. S1 Evidence 结构数据

Query-level 数据：`aggregate/v5_s1_evidence_structure_per_query.csv`。

聚合数据：`aggregate/v5_s1_evidence_structure_summary.csv`。

### 4.1 Evidence-group 数量

| Evidence groups | Query 数 | 比例 |
|---:|---:|---:|
| 1 | 1,009 | 74.46% |
| 2 | 238 | 17.56% |
| 3 | 98 | 7.23% |
| 4+ | 10 | 0.74% |

共有 346 条 query 需要多个 evidence groups，占 25.54%。每个 evidence group 平均约有 2 个 acceptable-memory alternatives。

### 4.2 AllGroupsCovered

| Query type | B=4 | B=8 | B=16 | B=32 |
|---|---:|---:|---:|---:|
| Exact match | 75.9% | 84.5% | 91.0% | 92.4% |
| Inference | 23.8% | 32.0% | 41.3% | 50.9% |
| Multi-hop | 1.7% | 5.2% | 10.3% | 27.6% |
| Multiple choice | 15.9% | 20.4% | 29.8% | 38.1% |
| State update | 37.7% | 45.7% | 52.2% | 60.9% |
| Temporal | 63.4% | 71.3% | 80.4% | 89.1% |
| 全体 query | 40.1% | 47.1% | 55.1% | 63.2% |

### 4.3 Mean GroupCoverage

| Query type | B=4 | B=8 | B=16 | B=32 |
|---|---:|---:|---:|---:|
| Exact match | 75.9% | 84.5% | 91.0% | 92.4% |
| Inference | 31.8% | 42.2% | 52.9% | 63.0% |
| Multi-hop | 24.4% | 35.8% | 46.4% | 62.4% |
| Multiple choice | 22.3% | 28.0% | 38.2% | 48.3% |
| State update | 37.7% | 45.7% | 52.2% | 60.9% |
| Temporal | 63.4% | 71.3% | 80.4% | 89.1% |
| 全体 query | 44.9% | 53.3% | 62.3% | 70.7% |

单 group query 的 GroupCoverage 与 AllGroupsCovered 相同。多 group query 可能获得部分 group coverage，但仍无法完成全部 evidence aspects，因此 Multi-hop 的两项指标差异最明显。

## 5. M2 Flat Frontier 数据

直接绘图数据：`aggregate/v5_reused_m2_source.csv`。

Budget 聚合数据：`aggregate/v5_reused_m2_weighted_by_budget.csv`。

| Frontier B | Query 数 | Weighted selection latency | Weighted host bytes | Weighted evidence recall | B=64 reference overlap |
|---:|---:|---:|---:|---:|---:|
| 8 | 240 | 58.55 ms | 3.455 MB | 0.1097 | 0.5542 |
| 16 | 240 | 97.91 ms | 3.489 MB | 0.1100 | 0.7200 |
| 32 | 240 | 143.77 ms | 3.556 MB | 0.1122 | 0.8575 |
| 64 | 240 | 229.23 ms | 3.686 MB | 0.1108 | 1.0000 |

`reranker_oracle_overlap_at_5` 实际表示与 B=64 high-budget reranker reference 的 top-5 overlap。它不是人工 gold 或 answer accuracy。B=64 overlap=1 是由 reference 定义决定的。

从 B=8 扩展到 B=32：reference overlap 增加约 0.303，但 selection latency 增加约 85.22 ms。扩展到 B=64 后 latency 是 B=8 的约 3.92 倍。

## 6. M3 Tile 数据

Query-level 数据：`aggregate/v5_reused_m3_source.csv`，共 2,160 行，即 240 queries × 3 capacities × 3 layouts。

聚合数据：`aggregate/v5_reused_m3_tile_summary.csv`。

### 6.1 Activated Tiles

| Layout | Capacity 8 | Capacity 16 | Capacity 32 |
|---|---:|---:|---:|
| Chronological | 4.50 | 4.21 | 3.84 |
| Random per persona | 4.94 | 4.90 | 4.85 |
| Oracle lower bound | 1.00 | 1.00 | 1.00 |

### 6.2 Access Amplification

| Layout | Capacity 8 | Capacity 16 | Capacity 32 |
|---|---:|---:|---:|
| Chronological | 7.21x | 13.48x | 24.56x |
| Random per persona | 7.90x | 15.68x | 31.04x |
| Oracle lower bound | 1.60x | 3.20x | 6.40x |

### 6.3 Useful Payload Utilization

| Layout | Capacity 8 | Capacity 16 | Capacity 32 |
|---|---:|---:|---:|
| Chronological | 14.22% | 7.78% | 4.52% |
| Random per persona | 12.70% | 6.41% | 3.24% |
| Oracle lower bound | 62.50% | 31.25% | 15.63% |

所有布局使用相同的 top-5 selected memories。Oracle lower bound 是 query-specific 理论下界，不是可部署布局。

## 7. S2 内容维护数据

直接绘图数据：`aggregate/v5_s2_generic_content_maintenance.csv`。

结果清单：`aggregate/v5_s2_generic_content_maintenance_summary.json`。

初始 base memory 数为 33,627。每个 checkpoint 使用 128 个 top-8 lookup probes。

| Growth | Cumulative appended | Active memories | New postings | Full postings | Changed fraction | Incremental time | Rebuild time | Time ratio | Mismatches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 420 | 34,047 | 105,223 | 8,587,986 | 1.23% | 44.85 ms | 4.002 s | 89.22x | 0 |
| 5% | 2,102 | 35,729 | 426,061 | 9,014,047 | 4.73% | 232.14 ms | 3.403 s | 14.66x | 0 |
| 10% | 4,203 | 37,830 | 527,323 | 9,541,370 | 5.53% | 224.35 ms | 3.669 s | 16.35x | 0 |
| 20% | 8,407 | 42,034 | 1,023,831 | 10,565,201 | 9.69% | 429.35 ms | 3.693 s | 8.60x | 0 |

四个 checkpoints 共执行 512 次 incremental-vs-rebuild lookup comparison，mismatch 总数为 0。wall time 会受机器状态影响，论文中应以 posting operation count 和 exact lookup equivalence 为主要证据。

## 8. 图与数据文件映射

| Figure | 数据文件 |
|---|---|
| `Fig_V5_M1_history_pressure` | `v5_reused_m1_source.csv` |
| `Fig_V5_M2_flat_frontier_tradeoff` | `v5_reused_m2_source.csv` |
| `Fig_V5_M3_tile_activation` | `v5_reused_m3_source.csv`、`v5_reused_m3_tile_summary.csv` |
| `Fig_V5_S1_evidence_structure` | `v5_s1_evidence_structure_per_query.csv`、`v5_s1_evidence_structure_summary.csv` |
| `Fig_V5_S2_content_maintenance` | `v5_s2_generic_content_maintenance.csv` |

## 9. 闭环与审计数据

- `aggregate/v5_motivation_chain_status.csv`：五个动机环节的 pass/fail 状态。
- `aggregate/v5_motivation_closure.json`：复用数据、补充实验和 claim boundary 总清单。
- `aggregate/v5_validation.csv`：逐项验证结果。
- `aggregate/v5_validation.json`：机器可读的最终验证报告，当前状态为 `passed`。
- `configs/v5_protocol.json`：输入边界、实验预算和 validation guardrails。

## 10. 数据使用注意事项

1. M1 的 latency 不严格单调，不能写成线性 scaling law。
2. M2 的 B=64 是 reference，不是 gold quality；evidence recall 基本不随 B 增长。
3. M3 是逻辑 Tile 和 over-fetch proxy，不能转换为 PIM latency 或 energy。
4. S1 使用 train-only evidence labels，只能证明 workload structure，不能证明正式方法优越。
5. S1 的 4+ group 子集只有 10 条，必须保留样本量和宽 CI。
6. S2 是通用 inverted-index static insertion stress，不是 v21 dynamic hypergraph benchmark。
7. S2 的 posting counts 和 zero mismatch 是稳定证据；单次 wall time 只能作为辅助结果。

