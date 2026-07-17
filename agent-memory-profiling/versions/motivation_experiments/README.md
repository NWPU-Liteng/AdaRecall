# Method 1 Motivation Experiments

这是 Method 1 的独立动机实验归档，不是 v21 的可部署主线，也不参与
`versions/INDEX.md` 中的 canonical package 选择。这里保留用于支撑论文动机、
工作负载特征和早期系统假设的正式报告、配置、脚本、图表和小型汇总结果。

## 与 v21 的边界

- v21 的可运行实现、输入、OOF 结果和审阅材料位于
  `20260714_method1_fsa_dynamic_hypergraph_v21/`。
- 本目录中的实验不应被当作 v21 的训练输入、校准输入或 formal-test 结果。
- v21 所需的 MedMemoryBench 输入已经复制到 v21 的 `inputs/`，因此 v21 不依赖
  本目录的相对路径；迁移清单中的旧绝对路径只是历史来源记录。
- 本目录中的原始大规模 trace、缓存和模型检查点没有全部保留。保留内容以报告、
  配置、图表、manifest、验证结果和唯一负结果为主；相同 SHA-256 文件只保留一份。

## 实验清单

| 目录 | 作用 | 状态 |
| --- | --- | --- |
| `20260616_phase_aware_real_agent_v3_formal` | 多阶段真实 agent 访问与 rerank 成本剖析 | 正式动机证据 |
| `20260621_real_trace_adaptive_phase_v4_formal` | 真实 trace 的自适应阶段/预算实验 | 正式动机证据 |
| `20260622_memorycd_candidate_frontier_v5_1_formal` | MemoryCD candidate-frontier 质量-代价实验 | 正式动机证据 |
| `20260622_memorycd_native_v5_formal` | MemoryCD 原生任务重跑 preflight | 阻塞记录，不是正式结果 |
| `20260703_method1_motivation` | 动机套件 M1/M2/M3/O1 汇总与图表 | 汇总包 |
| `20260704_method1_motivation_medmemorybench_v3` | MedMemoryBench v3 动机 workload 实验 | 正式/审计材料 |
| `20260705_method1_motivation_medmemorybench_v4_completion` | v3 的 v4 完成与泄漏审计 | 正式/审计材料 |
| `20260715_method1_motivation_medmemorybench_v5_alignment` | v5 对齐、证据歧义和维护成本分析 | 工作负载诊断 |
| `20260716_method1_motivation_medmemorybench_v5p1_controlled` | 固定 cohort 的 v5.1 控制实验 | 工作负载诊断 |

## 推荐阅读顺序

1. `20260716_method1_motivation_medmemorybench_v5p1_controlled/README.md`
2. `20260715_method1_motivation_medmemorybench_v5_alignment/README.md`
3. `20260705_method1_motivation_medmemorybench_v4_completion/README.md`
4. `20260704_method1_motivation_medmemorybench_v3/reports/FINAL_MOTIVATION_SUMMARY.md`
5. 需要追溯早期真实 agent 访问路径时，再阅读 v3/v4 与 MemoryCD 目录。

## 去重与迁移记录

本目录的文件选择和 SHA-256 记录位于
`../20260714_method1_fsa_dynamic_hypergraph_v21/provenance/legacy_versions/SELECTION_INDEX.tsv`
和 `DEDUP_INDEX.tsv`；完整来源到当前路径的映射位于
`../20260714_method1_fsa_dynamic_hypergraph_v21/aggregate/MIGRATION_MANIFEST_20260717.tsv`。
这些清单保留原始版本名作为 provenance，不代表原目录仍存在或仍可直接运行。

