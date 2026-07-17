# Real Trace Adaptive Phase V4 说明

该版本目录用于承接 `codex_real_trace_phase_locality_experiment_guide.md` 指导下的 V4 实验资产。

当前目录包含：

- `manifests/`: 运行时 manifest、memory bank 与 evaluator-only labels
- `runs/`: V4 smoke/formal profiling 输出
- `reports/`: 泄漏审计、tenant 隔离、数据溯源等报告
- `configs_snapshot/`: 本次运行使用的配置快照

约束说明：

- 保持 V3 目录不变，不覆盖 `versions/20260616_phase_aware_real_agent_v3_formal/`
- V4 runtime 不导入 `synthetic_memory_generator.py`
- evaluator-only label 与 runtime manifest 分离保存
- 本次默认使用 GPU 上的本地 Qwen、BGE 和 MiniLM 模型
