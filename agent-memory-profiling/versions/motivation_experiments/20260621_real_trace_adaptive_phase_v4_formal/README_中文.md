# 20260621_real_trace_adaptive_phase_v4_formal 说明

## 1. 版本用途

该版本目录保存的是本轮实验中**最终保留的正式结果**，用于后续阅读、汇报和写作。

本版本已经清理掉了以下中间产物：

- smoke / early plain runs
- tail-heavy exploratory runs
- 失败重试目录
- 不再对应最终结果的旧配置快照

当前保留内容只包含：

- 正式 manifest
- 正式全量运行结果
- 对应配置快照
- 正式日志
- 报告与说明文档

---

## 2. 当前保留的正式实验

### 2.1 LoCoMo

- workload: `official_qa`
- event type: `official_qa`
- runs:
  - `locomo_official_qa_full_shared_once_controller_lite_selection_only`
  - `locomo_official_qa_full_fixed_three_controller_lite_selection_only`
  - `locomo_official_qa_full_adaptive_phase_controller_lite_selection_only`

### 2.2 LongMemEval-S

- workload: `official_qa`
- event type: `official_qa`
- runs:
  - `longmemeval_s_official_qa_full_shared_once_controller_lite_selection_only`
  - `longmemeval_s_official_qa_full_fixed_three_controller_lite_selection_only`
  - `longmemeval_s_official_qa_full_adaptive_phase_controller_lite_selection_only`

### 2.3 MSC

- workload: `replay_turn`
- event type: `replay_turn`
- runs:
  - `msc_replay_turn_full_shared_once_controller_lite_selection_only`
  - `msc_replay_turn_full_fixed_three_controller_lite_selection_only`
  - `msc_replay_turn_full_adaptive_phase_controller_lite_selection_only`

注意：

- `MSC` 没有与 LoCoMo / LongMemEval-S 同构的 `official_qa` 结构，因此这里只能做 replay-turn 工作负载分析，不能把 MSC 结果解释成 benchmark QA 结果。

---

## 3. 当前实验设置

### 3.1 通用设置

- `selection_only = true`
- 使用真实本地 `Qwen` 作为 controller
- 使用真实 embedding + dense retrieve + cross-encoder rerank
- 当前 `selection_only` 模式下会**跳过 output phase**
- 因此当前实验只执行：
  - `planning`
  - `reasoning`

### 3.2 三种模式的含义

- `shared_once`
  - 只在 `planning` 执行一次 retrieval
- `fixed_three`
  - 在当前 selection-only 定义下，等价于前两个 phase 都执行 retrieval
- `adaptive_phase`
  - 每个 phase 先由真实本地 Qwen controller 决定是否 retrieval

注意：

- 因为 `output` phase 已在 selection-only 模式下被跳过，所以这里的 `fixed_three` 实际平均 retrieval 次数为 `2.0`，不是旧定义下的 `3.0`。

---

## 4. 目录结构说明

### 4.1 `manifests/`

保存三个数据集的正式 manifest 与 provenance：

- `manifests/locomo`
- `manifests/longmemeval_s`
- `manifests/msc`

其中包含：

- `runtime_manifest.jsonl`
- `runtime_memory_bank.jsonl`
- `evaluation_labels.jsonl`（MSC 基本无 QA 用途）
- `data_provenance_report.md`

### 4.2 `runs/`

仅保留 9 个正式结果目录：

- LoCoMo 3 个
- LongMemEval-S 3 个
- MSC 3 个

每个目录内通常包含：

- `raw_profile.jsonl`
- `phase_trace.jsonl`
- `raw_profile_frame.json`
- `summary.csv`
- `summary.md`
- `run_manifest.json`

### 4.3 `configs_snapshot/`

仅保留与上述 9 个正式结果完全对应的配置快照。

### 4.4 `logs/full_official_qa/`

保留正式运行日志。

虽然该目录名是 `full_official_qa`，但其中也保存了本次 MSC replay-turn 的正式日志，这是历史沿用的日志目录名，不影响结果解释。

### 4.5 `reports/`

当前保留的报告包括：

- `leakage_audit.md`
- `tenant_isolation_report.md`
- `phase_selection_tail_heavy_progress_20260621.md`

其中 `phase_selection_tail_heavy_progress_20260621.md` 记录了前期 tail-heavy 探索结论，属于过程性说明；正式最终结果仍以 `runs/` 下的 full 结果为准。

---

## 5. 三个数据集的正式结果概览

| 数据集 | 事件类型 | 模式 | runs | retrieval | selection ms | controller ms | total ms | selection 占比 | controller 占比 | rerank / selection |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | official_qa | shared_once | 1986 | 1.0 | 558.93 | 0.00 | 559.74 | 99.86% | 0.00% | 97.15% |
| LoCoMo | official_qa | fixed_three | 1986 | 2.0 | 1092.62 | 0.00 | 1093.72 | 99.90% | 0.00% | 97.34% |
| LoCoMo | official_qa | adaptive_phase | 1986 | 2.0 | 1090.68 | 1253.24 | 2347.84 | 46.45% | 53.38% | 97.32% |
| LongMemEval-S | official_qa | shared_once | 500 | 1.0 | 45.06 | 0.00 | 47.30 | 95.28% | 0.00% | 67.10% |
| LongMemEval-S | official_qa | fixed_three | 500 | 2.0 | 87.92 | 0.00 | 89.78 | 97.93% | 0.00% | 69.81% |
| LongMemEval-S | official_qa | adaptive_phase | 500 | 2.0 | 81.26 | 1056.20 | 1142.24 | 7.11% | 92.47% | 71.81% |
| MSC | replay_turn | shared_once | 2980 | 1.0 | 103.84 | 0.00 | 104.22 | 99.64% | 0.00% | 87.14% |
| MSC | replay_turn | fixed_three | 2980 | 2.0 | 207.66 | 0.00 | 208.17 | 99.76% | 0.00% | 87.27% |
| MSC | replay_turn | adaptive_phase | 2980 | 2.0 | 197.33 | 1130.65 | 1331.15 | 14.82% | 84.94% | 87.57% |

---

## 6. 如何解读这些结果

### 6.1 LoCoMo

LoCoMo 是当前最能支撑“selection bottleneck 动机”的数据集。

理由：

- `adaptive_phase` 下 selection 占比达到 `46.45%`
- `rerank / selection` 达到 `97.32%`
- 说明 selection 路径本身确实很重，且 rerank 是其内部主导成本

适合支持的说法：

- 在真实 official QA 工作负载下，selection path 可以成为主要瓶颈之一
- rerank 是 selection 内部最昂贵部分

### 6.2 LongMemEval-S

LongMemEval-S full official_qa 整体负载偏轻。

表现为：

- `shared_once` / `fixed_three` selection 本身很快
- `adaptive_phase` 中 controller 成本远大于 selection
- selection 占比只有 `7.11%`

适合支持的说法：

- 不同真实数据集之间的 selection 压力差异很大
- 不是所有 benchmark QA workload 都会自然形成重 rerank 瓶颈

### 6.3 MSC

MSC 不适合用来论证 official QA selection bottleneck。

原因：

- 只有 `replay_turn`
- 单个事件可见 memory 很少
- 候选 frontier 本身较小

适合支持的说法：

- 在真实 replay workload 下，adaptive controller 仍然是主要在线开销
- MSC 更适合后续做长期在线行为 / locality / warm-up 相关分析

---

## 7. 建议后续使用方式

### 7.1 如果要写动机实验

优先使用：

- `LoCoMo official_qa full`

谨慎使用：

- `LongMemEval-S official_qa full`
  - 更适合作为“轻 workload 对照”

不建议直接拿来做相同论证：

- `MSC replay_turn full`
  - 它不是 official QA
  - 更适合说明 replay 环境下 selection 与 controller 的相对关系

### 7.2 如果要看正式结果

优先阅读：

- 各 run 目录下的 `summary.md`
- 各 run 目录下的 `summary.csv`

如需看明细：

- `raw_profile.jsonl`
- `phase_trace.jsonl`

### 7.3 如果要继续扩展实验

推荐方向：

- 基于 LoCoMo 继续做 phase locality / hypergraph pruning
- 把 LongMemEval-S 作为对照数据集
- 把 MSC 放在 replay / warm-up / online trace 方向使用

---

## 8. 最终保留原则

本目录中的数据已经过清理，默认可视为当前轮实验的**最终有效版本**。

后续如果继续实验，建议：

- 新建新的版本目录
- 不再覆盖本目录中的正式结果

