# CODEX_PLAN_v2：主实验代码完善执行方案

适用仓库：`agent-memory-profiling`
基线版本：`versions/20261031_mobicom_omni_causal_pim_w12_system_v1`（下称 v1）
目标版本：`versions/20261101_mobicom_omni_causal_pim_w12_system_v2`（下称 v2）

本文件是给编码代理的执行规范。请严格按任务编号顺序执行，每次只完成一个任务。

---

## 第 0 部分：给执行代理的启动指令

每次会话开始时，按以下要求执行：

```text
你是本仓库的编码代理。请先完整阅读 docs/CODEX_PLAN_v2.md 的第 1 部分（全局规则），
再阅读你本次要执行的任务（例如 T1.4）。只执行这一个任务。
开始写代码前，先阅读任务"需阅读的文件"一栏列出的全部文件，
并把你对现有接口的理解写入 NOTES/<任务编号>.md。
如果发现现有代码与本计划的描述不一致，停止编码，在 NOTES 中写明差异并报告，不要猜测。
完成后运行任务要求的全部测试和回归检查，然后按第 1.9 节的格式输出报告。
```

---

## 第 1 部分：全局规则（所有任务都必须遵守）

### 1.1 版本隔离
- 不得修改 v1 目录下的任何文件。
- T0.1 会把 v1 的代码和配置复制到 v2。后续所有修改只发生在 v2。
- 不复制数据、索引和运行输出。v2 通过配置路径引用 v1 已准备好的 prepared 数据和索引。

### 1.2 默认值必须复现 v1
- 所有新功能都放在配置的 `extensions:` 块下，并带开关。
- 所有开关的默认值必须让 v2 的行为与 v1 完全一致。
- 每个任务完成后都必须通过 T0.1 建立的回归检查。

### 1.3 禁止静默回退
- 新代码遇到不满足前提的情况时，必须 `raise`，并给出明确的错误信息。
- 唯一的例外是本计划明确定义了状态字段的情况，例如 `status = "not_executable"`。
- 不得用 try/except 吞掉异常，然后改走旧路径。

### 1.4 评测标签隔离（最高优先级）
Serving 路径中的任何模块都不得读取以下内容：
- `labels.jsonl`
- gold answer
- 题型
- 标注证据

Serving 路径包括：
- `run_proof_mainline.py`
- `run_main_experiment.py`
- `omni_proof_*.py`
- `pim_backend/*`

只有文件名以 `eval_` 或 `fit_` 开头的脚本，以及 `judge_incremental.py`，可以读取标签。

### 1.5 不得改名，只能新增
- 不得重命名现有的文件、函数、类、配置键和输出字段。
- 需要新的语义时，新增字段即可。
- 论文中的术语调整（例如不再使用 "proof"）只在写作时处理，不改代码标识符。

### 1.6 不得编造常数和结论
- 能耗参数、时序参数和文献数值，一律做成配置参数。
- 未知的值填 `null`，并加注释 `# TODO_CITE: 需要用户从文献填写`。
- 代码注释和报告中不得写"近似比为 X"、"节能 Y%"这类未经推导或测量的结论。

### 1.7 新产物必须带哈希
- 新生成的离线产物（calibrator、量化矩阵缓存等）都要写入 sha256。
- 在线加载时校验哈希，延续 v1 的 index 契约风格。

### 1.8 测试
- 测试放在 `tests/` 下，使用 pytest。
- 需要 GPU 或真实模型的测试标记为 `@pytest.mark.gpu`。
- CPU 单元测试必须能在 1 分钟内完成。
- 每个任务都必须新增对应测试，并保证 `pytest -q tests -m "not gpu"` 全部通过。

### 1.9 任务完成报告格式

```text
## 任务 <编号> 完成报告
- 修改/新增文件：<列表>
- 新增配置键及默认值：<列表>
- 新增测试：<列表>，结果：<通过数/总数>
- 回归检查：<通过/失败，及差异说明>
- 与计划不一致之处及原因：<无/说明>
- 需要用户决定或补充的事项：<无/列表>
```

### 1.10 提交
- 每个任务单独提交一次 commit。
- 提交信息格式：`[v2][T<编号>] <一句话描述>`。

---

## 第 2 部分：配置扩展总览

在 v2 的主配置中新增如下 `extensions:` 块。下面给出的值即为默认值，这些默认值必须复现 v1。

```yaml
extensions:
  visibility_protocol: history_end      # history_end | as_of_question
  as_of_rule: session_end_before_question  # 见 T1.4
  operation_executor: off               # off | on
  witness_values: off                   # off | on   （T1.5 的见证 schema 扩展）
  stage_profiler: off                   # off | on
  workload_trace: off                   # off | on
  progressive_topk:
    lm_head: off                        # off | shadow
    lm_head_shadow_every: 8             # 每 N 个 decode token 做一次影子校验
    retrieval: off                      # off | shadow | replace
    bits: 12
    group_size: 128
    plane_schedule: [1,1,1,1,1,1,1,1,1,1,1,1]  # 每轮读取的位平面数，总和必须等于 bits
    eps_rel: 1.0e-9
  candidate_dump: null                  # null 或输出目录路径（T3.1）
  calibration:
    calibrator_path: null
    calibrator_sha256: null
    candidate_policy: fixed48           # fixed48 | conformal
    alpha: 0.1
    min_per_demand: 4
  join_algorithm: dfs_maxmin            # dfs_maxmin | noisy_or_saturate
  noisy_or:
    max_witnesses_per_demand: 3
    levels: 20
    fill_redundancy: false
  vision_feature_cache: off             # off | on
```

配置解析要求：
- `extensions` 下出现未知键时，必须报错。
- 所有键都要做类型检查和取值检查。

---

## Phase 0：基础设施

### T0.1 建立 v2 目录与回归基线

**需阅读的文件：**
- v1 目录结构
- `configs/main_experiment_pilot112_w12.yaml`
- `scripts/run_main_experiment.py`
- `scripts/run_proof_mainline.py` 的输出写入部分

**步骤：**
1. 复制 v1 的 `scripts/`、`pim_backend/`、`configs/` 和其他代码目录到 v2，不复制数据和输出。如果无法判断某个目录是否属于数据，停止并询问用户。
2. 新建 `configs/regression_golden.yaml`，内容与主配置相同，但只运行一个固定的小集合：从 development split 中选 2 个用户，每个用户选 3 个问题。把选中的 question_id 写死在配置中，选择规则是按 question_id 字典序取前 3 个。
3. 用 v1 代码和该配置运行两次，结果保存到 `regression/golden_v1_run{1,2}/`。
4. 新建 `tools/regression_check.py <dir_a> <dir_b>`，逐题比较以下字段：
   - `answer`
   - `accepted_object_ids`（排序后比较）
   - `proof_status`
   - `selection_mode`
   - 每个 demand 的候选 record_id 列表（保持顺序）
   - 查询契约 JSON
5. 先比较 v1 自身的两次运行。如果有差异，说明基线本身存在非确定性。这时把有差异的字段写入 `regression/NONDETERMINISTIC_FIELDS.json`，后续检查对这些字段只报告差异、不判失败。
6. 用 v2 代码在默认配置下运行一次，并与 `golden_v1_run1` 比较，必须一致。

**验收：** `python tools/regression_check.py regression/golden_v1_run1 regression/golden_v2_default` 退出码为 0。

### T0.2 测试框架与标签隔离守卫

**步骤：**
1. 新建 `tests/conftest.py`，注册 `gpu` marker。
2. 新建 `tests/test_label_isolation.py`。对第 1.4 节列出的每个 serving 模块，检查源码中不出现以下任一字符串：
   - `labels.jsonl`
   - `golden_answers`
   - `"question_type"`
   - `"difficulty"`

   如果 v1 中已有合法引用（例如 manifest 校验时读取了字段名），在测试里显式列出白名单，并在 NOTES 中说明理由。
3. 新增一个运行时守卫：在 `run_proof_mainline.py` 启动时，对 `builtins.open` 加一层包装。当打开的路径以 `labels.jsonl` 结尾时，直接 `raise PermissionError`。该守卫在 `run_proof_mainline.py` 中始终开启。如果它导致 v1 行为出现差异，停止并报告。

**验收：** 测试通过；回归检查通过。

### T0.3 配置扩展解析

**需阅读的文件：** 主配置的加载代码（定位 `run_main_experiment.py` 和 `run_proof_mainline.py` 中读取 yaml 的位置）。

**步骤：**
1. 新建 `scripts/extensions_config.py`。
   - 定义 `@dataclass(frozen=True) class Extensions`，字段与第 2 部分一一对应，默认值相同。
   - 实现 `load_extensions(cfg_dict) -> Extensions`：缺省键使用默认值；未知键报错；类型错误报错。
   - 校验 `plane_schedule` 各项之和等于 `bits`。
2. 在 worker 启动时解析 `Extensions`，并把它传给后续模块。
3. 把 `Extensions` 的 JSON 及其 sha256 写入每个用户的输出文件（新增字段 `extensions`）。

**测试：**
- 测试默认值。
- 测试未知键会报错。
- 测试 `plane_schedule` 校验。

**验收：** 回归检查通过。新增的 `extensions` 字段不参与回归比较，需在 `regression_check.py` 中显式忽略。

---

## Phase 1：低成本、高优先级修补

### T1.1 阶段剖析器与工作负载 trace

目的：给出每个阶段的时延、token 数和 PIM 计数，并为周期级仿真和真机重放提供输入。

**需阅读的文件：**
- `run_proof_mainline.py`（三个 Reader 阶段的调用点、`counter_delta()`）
- `omni_proof_runtime.py`（`greedy()`）
- `pim_backend/w12_functional_dispatch.py`

**步骤：**
1. 新建 `scripts/stage_profiler.py`，实现上下文管理器 `with profiler.stage(name):`。
   - 记录 `wall_ms`，用 `torch.cuda.synchronize()` 前后的 `time.perf_counter()` 计算。
   - 记录每个阶段的 W12 计数器差值，复用 `counter_delta()`。
2. 覆盖以下阶段：
   - `contract_parse`
   - `query_embed`
   - `dense_retrieval`
   - `rerank`
   - `witness_construct`
   - `join`
   - `materialize`
   - `answer_prefill`
   - `answer_decode`

   对三个 Reader 阶段，需要把 prefill 和 decode 分开记录。具体做法是在 `greedy()` 内部，第一次 forward 前后单独计时。
3. 每个 Reader 阶段额外记录：
   - `prefill_tokens`：即 `input_ids.shape[1]`
   - `n_images`
   - `decode_tokens`
   - `grammar_forced_tokens`：bitmask 允许集合大小为 1 的步数
   - `grammar_allowed_mean`：允许集合大小的均值。读取允许集合大小需要解包 xgrammar 的 bitmask，实现位置见 T2.2 的说明；本任务只做计数。
4. 当 `workload_trace: on` 时，每个问题写一个 `trace/<user>/<qid>.json`，内容包括：
   - 上述所有字段。
   - `dense_retrieval` 的矩阵形状 `[N, 1024]`。
   - 每个 Reader 阶段中，每个被调用的 Linear 的名称、形状和调用次数（从 dispatcher 计数器汇总）。
5. 当 `stage_profiler: on` 时，在输出行中新增 `stage_profile` 字段。

**测试：**
- CPU 单测：用一个假模型验证上下文管理器能正确累加时间。
- 验证 `grammar_forced_tokens` 的计数逻辑：构造一个 bitmask，只允许 1 个 token。

**验收：**
- 回归检查通过（开关关闭时）。
- 开关打开时，在 golden 集上能产出 trace，并且各阶段 `wall_ms` 之和与该问题总耗时的误差在 5% 以内。

### T1.2 运行统计报告（不读标签）

新建 `scripts/report_run_stats.py <run_dir>`，输出 `run_stats.json` 和一张 markdown 表。内容包括：
- `proof_status` 和 `selection_mode` 的分布。
- `answer_mode = contextual` 的比例。
- `parse_status != ok` 的比例。
- join 是否 `optimal`；未达最优时 `support_gap` 的分布。
- 见证校验丢弃原因的计数。
- `omitted_whole_record_ids_due_to_context` 的平均数量。
- 最终证据的文本 token 数、图片数和 processor token 数的分布。
- 若有 `stage_profile`，给出各阶段时延的均值和 P50/P90。

**测试：** 用一个合成的输出 JSON 验证各项统计。

### T1.3 证据召回评测（eval-only，读标签）

**需阅读的文件：**
- `labels.jsonl` 的实际 schema（先打印 3 条样例写入 NOTES）
- `omni_graph.json` 中 session 和 atom 的字段
- `source.json` 的 object 字段

**步骤：**
1. 新建 `scripts/eval_evidence_recall.py <run_dir> --labels <path>`。
2. 先在 NOTES 中写清楚，标注证据能映射到哪一层级（event、session、atom 或图片）。如果标注证据只能映射到 session 级，就只计算 session 级指标。不要编造更细的映射。
3. 指标：
   - `evidence_recall_session`：标注 session 中，被最终 `accepted_object_ids` 覆盖的比例。
   - `evidence_recall_image`：标注图片中，被最终证据包含的比例。图片按 sha256 匹配。
   - `candidate_recall_session`：标注 session 在 48 个候选池中出现的比例。数据来源依赖 T3.1 的 dump；没有 dump 时跳过该项。
   - 以上指标分别按 `proof_status` 分组，再按 `question_type` 分组报告。
4. 同时输出一个映射失败计数，即无法映射的标注证据数量。

**测试：** 用合成的 labels 和输出验证计算结果。

### T1.4 按提问时刻可见（as_of_question）

**需阅读的文件：**
- `omni_prepare.py`（`visibility_start`/`visibility_end`、`effective_timestamp` 的格式）
- `run_proof_mainline.py` 和 planner 中关于"在线可见性 mask"的现有实现
- `omni_proof_sources.py` 中的 `materialize()`

**规则定义：**
- `as_of_rule = session_end_before_question`：当且仅当 `session.visibility_end <= question.effective_timestamp` 时，该 session 可见。
- 时间解析必须复用 prepare 阶段已有的解析函数。遇到时区不一致或格式不一致时，`raise`。

**步骤：**
1. 新建函数 `visible_sessions(graph, question) -> frozenset[str]`。
2. 对象可见性规则：
   - 文本 object：所属 atom 的 session 可见，即为可见。
   - 图片 object：至少有一个来源 atom 的 session 可见，即为可见。
3. 当 `visibility_protocol = as_of_question` 时：
   - (a) 稠密检索使用可见性 mask，复用现有的 mask 参数。如果现有接口不支持，在 `dense_topk` 中新增可选参数 `row_visible: np.ndarray[bool]`。
   - (b) 见证构造只接收可见候选。
   - (c) `materialize()` 新增参数 `visible_sessions`。任何不可见 object 进入物化时，直接 `raise`。
   - (d) 图片的来源 header 只列出可见的来源 atom。
4. 输出行新增两个字段：`visibility_protocol` 和 `n_visible_sessions`。
5. 某个问题的可见 session 数为 0 时，不调用 Reader，直接写 `answer = ""` 和 `status = "no_visible_history"`。这个状态需要计入统计。
6. 在 `eval_evidence_recall.py` 中增加一个统计：有多少问题的标注证据 session 在 as_of 规则下不可见，即该问题依赖"未来历史"。这项统计只在 eval 脚本中做。

**测试：**
- 合成图测试：给定 session 时间和问题时间，验证可见集合计算正确。
- 验证图片多来源时的可见性。
- 验证 `materialize` 遇到不可见 object 时会 raise。

**验收：** 默认 `history_end` 时回归检查通过；golden 集在 `as_of_question` 下能跑通。

### T1.5 运算 DAG 的确定性执行

目的：让 difference、comparison、selection、aggregation 由程序计算，Reader 只负责措辞。

现状：见证的 `bindings` 只记录跨事实连接键，没有日期、数值等运算输入。因此需要两部分改动：先扩展见证 schema（受 `witness_values` 开关控制），再实现执行器（受 `operation_executor` 开关控制）。

**需阅读的文件：**
- `omni_proof_grounding.py`（xgrammar schema、`expand_compact_sources()`、`validate_witnesses()`）
- `omni_proof_query_contract.py`（operations、`render_operation_plan()`）

#### T1.5a 见证 schema 扩展：`values`

当 `witness_values: on` 时，见证 schema 新增一个可选数组：

```json
"values": [
  {"name": "d0.date", "type": "date|datetime|number|duration|count|text",
   "value": "逐字值", "record_id": "r0", "text_index": 0}
]
```

约束与校验：
- 最多 8 项。
- `value` 必须逐字出现在对应原文中，校验方式与 binding 完全一致，复用同一个函数。
- 校验失败的项丢弃，并记录到 `discarded_optional_fields`。
- 在见证构造 prompt 中加一句说明："若问题需要计算，请在 values 中逐字摘出参与计算的日期/数字/时长，并标注类型。" prompt 文本放在常量中，并在输出中记录其 sha256。

#### T1.5b 类型化值解析：`scripts/typed_values.py`

```python
@dataclass(frozen=True)
class Interval:      # 闭区间，用于表达不确定性
    lo: float
    hi: float
    unit: str        # "day" | "second" | "number" | "count"

def parse_value(v: dict, session_interval: tuple[datetime, datetime]) -> Interval | None
```

解析规则：
- **绝对日期：** 支持 ISO 格式、`YYYY年M月D日`、`M月D日`（年份取 session 区间的年份；若跨年无法确定，返回整年区间）、英文月份名。日期的单位为 day，用 ordinal 表示。
- **相对日期：** 如"昨天"、"上周"，用 session 可见区间 `[visibility_start, visibility_end]` 推出一个区间，不得假设具体时刻。原因是 v1 明确规定时间标签是 session 区间，而不是逐轮时刻。
- **数字：** 支持阿拉伯数字、千分位，以及常见中文数字（零到万）。
- **时长：** 支持"N 天/周/月/年"。月和年换算成天时，使用区间：1 月记为 [28, 31] 天，1 年记为 [365, 366] 天。
- 无法解析时返回 `None`，不得猜测。

#### T1.5c 执行器：`scripts/operation_executor.py`

```python
def execute(operations, witnesses_by_demand, objects, graph) -> dict[str, dict]
# 返回 {op_id: {"status": "ok"|"not_executable", "result": ..., "reason": str}}
```

各运算的规则：
- **difference：** 结果为 `[a.lo - b.hi, a.hi - b.lo]`，单位必须一致。
- **comparison：** 区间不相交时，给出确定的顺序；相交时结果为 `"undetermined"`。
- **selection：** 对输入做 argmax/argmin。默认取 argmax；问题片段中出现最早、最少、最小等词时取 argmin，判定词表写在常量中。只有区间排序唯一确定时，status 才为 ok。
- **aggregation：** 只实现计数。统计该 demand 已接受见证中不同 binding 值的个数，结果为 `{"count_lower_bound": n}`。原因是见证只覆盖检索到的部分，不能声称是精确总数。

通用规则：
- 输入可以是 demand（取该 demand 见证中的 values），也可以是更早 operation 的结果。
- 任一输入缺失或无法解析时，status 为 `not_executable`，并写明原因。

#### T1.5d 渲染

当 `operation_executor: on` 时，在 `render_operation_plan()` 输出之后追加一段：

```text
程序计算结果（由原文定位值确定性计算，区间表示时间粒度带来的不确定性）：
- op0: difference = [12, 14] 天；输入：d0.date=..., d1.date=...
- op1: not_executable（原因：d1 缺少可解析日期）
```

`not_executable` 的运算保持原有的文本计划。

**测试（必须覆盖）：**
- 日期解析，包括中文、英文、跨年和相对日期。
- 区间差。
- comparison 的相交与不相交。
- 输入缺失时为 `not_executable`。
- DAG 中运算结果作为后续运算的输入。

**验收：** 开关关闭时回归检查通过；开关打开时 golden 集能跑通，`report_run_stats` 中新增 operation 状态分布。

---

## Phase 2：核心机制——可证明精确的渐进位平面 top-k

### 背景与数学定义（实现必须严格按此）

**量化与编码。**
- 权重按行、按组量化，组大小 `group_size = 128`。
- 码值为 `q ∈ [-2047, 2047]`，复用 v1 的量化函数。
- 用 12-bit two's complement 存储。plane 0 是符号位，权值为 `-2^11`；plane p（p ≥ 1）的权值为 `2^(11-p)`。

**部分码值。** 读完前 m 个位平面（m ≥ 1）后：

```
q^(m) = -2^11·b_0 + Σ_{p=1}^{m-1} 2^(11-p)·b_p
q = q^(m) + r,   0 ≤ r ≤ R_m = 2^(12-m) - 1
```

**得分与上下界。** 对输入向量 x（使用 float64），每组定义：

```
P_g = Σ_{j∈g} max(0, x_j)
N_g = Σ_{j∈g} min(0, x_j)
```

第 i 行的得分和上下界为：

```
s_i = Σ_g σ_ig Σ_{j∈g} q_ij x_j
S_i^(m) = Σ_g σ_ig Σ_{j∈g} q^(m)_ij x_j
L_i = S_i^(m) + R_m Σ_g σ_ig N_g
U_i = S_i^(m) + R_m Σ_g σ_ig P_g
```

其中 `σ_ig > 0` 是组 scale。

**剪枝规则。**
- 在幸存行中，令 τ_k 为第 k 大的 L_i。
- 若 `U_i < τ_k - eps`，则永久剪除第 i 行。
- 当幸存行数小于等于 k 时，不剪枝。
- `eps = eps_rel · max(1, max_i |U_i|)`。

**正确性。** 真实 top-k 中的任意一行都满足 `U_i ≥ s_i ≥ τ_k`，因此不会被剪除。

**最终排序。** 对幸存行用全部位平面重算精确得分，按 `(-score, index)` 排序后取前 k 个。

**精确性定义。** 结果必须与 `reference_scores()`（同一份量化码、float64、同一计算路径）全量计算后、按相同排序规则得到的 top-k 完全一致，包括索引和分数逐位相等。

### T2.1 参考实现：`pim_backend/progressive_topk.py`

**需阅读的文件：**
- `pim_backend/functional_decode.py`（量化函数）
- `pim_backend/w12_bank_layout.py`（现有位平面编码方式）

先在 NOTES 中写明 v1 的位平面编码是 two's complement 还是符号-幅值。无论哪种，新模块都使用 two's complement plane-major 编码。

```python
@dataclass(frozen=True)
class PlaneMajorMatrix:
    planes: np.ndarray   # uint8 [bits, rows, ceil(cols/8)]，plane 0 = 符号位
    scales: np.ndarray   # float64 [rows, n_groups]
    bits: int
    group_size: int
    cols: int
    codes_sha256: str

def quantize_plane_major(w: np.ndarray, bits=12, group_size=128) -> PlaneMajorMatrix
    # 必须调用 v1 的量化函数得到 q 和 scale，只改变打包方式
def from_codes(codes: np.ndarray, scales: np.ndarray, bits, group_size) -> PlaneMajorMatrix
def decode_codes(m: PlaneMajorMatrix) -> np.ndarray          # int32 [rows, cols]
def reference_scores(m, x, rows=None) -> np.ndarray          # float64
def progressive_topk(m, x, k, row_mask=None,
                     plane_schedule=None, eps_rel=1e-9) -> tuple[np.ndarray, np.ndarray, dict]
```

`stats` 字典必须包含：
- `rounds`：列表，每轮记录 `{planes_in_round, rows_alive_before, rows_alive_after}`
- `plane_row_reads`：Σ 每轮的幸存行数 × 该轮平面数
- `full_plane_row_reads`：rows_masked × bits
- `weight_bytes_read`
- `weight_bytes_full`
- `scale_bytes_read`

计算时把 `row_mask = False` 的行视为不存在，不计入读取。

**测试（`tests/test_progressive_topk.py`，CPU）：**
1. `decode_codes(quantize_plane_major(w))` 等于 v1 量化函数输出的码值。
2. 用 200 组随机种子，覆盖以下维度：
   - 矩阵形状：rows ∈ {1, 7, 300, 5000}，cols ∈ {128, 1024, 2560}
   - k ∈ {1, 5, 48, 80}
   - 分布：高斯、重尾（student-t）
   - x 含负值

   每组都断言 `progressive_topk` 与 reference 全排序的结果完全一致。
3. 平局：构造重复行和全零行，断言排序规则一致。
4. `row_mask` 场景：只允许 1 行；允许 0 行时 raise；允许行数小于 k 时返回全部允许行。
5. 测试多种 `plane_schedule`：全部为 1、`[4,2,2,2,2]`、`[12]`。
6. 断言 `plane_row_reads <= full_plane_row_reads`。
7. 单测：上下界对每轮每个幸存行都成立，即 `L_i <= s_i <= U_i`。

### T2.2 lm_head 影子模式（含语法掩码）

**需阅读的文件：**
- `omni_proof_runtime.py` 中的 `greedy()`（bitmask 应用方式）
- `w12_functional_dispatch.py`（lm_head 包装类）
- 模型中 lm_head 与 embed_tokens 是否共享权重

**步骤：**
1. 在 `W12BankLayout` 中新增只读导出方法 `export_codes_and_scales() -> (codes int16 [rows, cols], scales float32 [rows, groups])`。导出结果必须与 bank 存储逐位一致，并编写测试验证。
2. 每个 worker 进程只构建一次 lm_head 的 `PlaneMajorMatrix`。完成后释放 CPU 侧的中间数组。如果内存不足，在 NOTES 中报告峰值，不要擅自改成降采样实现。
3. 在 lm_head 包装器上注册 forward hook，捕获最后位置的输入 hidden。
4. 当 `progressive_topk.lm_head = shadow` 时，在 `greedy()` 的每第 N 个 decode token（N = `lm_head_shadow_every`），以及每个被语法强制的 token 上执行以下步骤：
   - (a) 从 xgrammar bitmask 解包出 `allowed: bool[vocab_rows]`。无 grammar 时全为 True；超出 tokenizer 词表的填充行设为 False。
   - (b) 用 torch 在 GPU 上以 float64 计算 `progressive_topk(k=1, row_mask=allowed)`。需要提供一个与 numpy 版本逐位一致的 torch 实现 `progressive_topk_torch`，并编写测试，在 CPU 上比较两个版本。
   - (c) 用 `reference_scores` 在 allowed 行上计算 argmax，并断言两者相等。不相等时 raise，这一断言属于定理级保证。
   - (d) 记录 progressive argmax 是否等于实际 BF16 路径选出的 token，计数为 `agree_bf16` 和 `disagree_bf16`。这一项只记录，不断言。原因是 BF16 舍入会造成差异，它反映的是数值误差，不是本机制的错误。
   - (e) 累加 `stats`，并按阶段分别统计：`contract_parse`、`witness_construct`、`answer`。
5. 当 `allowed` 中只有 1 个 True 时，记为 `forced`，此时 lm_head 读取字节为 0，不需要调用 `progressive_topk`。
6. 新增输出字段 `progressive_lm_head_stats`，按阶段给出：
   - 校验的 token 数
   - forced 数
   - `weight_bytes_read / weight_bytes_full`
   - 平均轮数
   - BF16 一致率

**测试：**
- GPU 测试：在 golden 集的 1 个问题上，shadow 模式下断言全部通过。
- 开关关闭时，回归检查通过。
- 开关打开时，`answer` 字段必须与关闭时一致，因为影子模式不得改变输出。

### T2.3 检索的渐进 top-k

**需阅读的文件：** 稠密检索与 48/32 选择逻辑所在函数（planner 或 runtime）。

**步骤：**
1. 用户索引加载后，对 `vectors.npy` 调用 `quantize_plane_major`，得到 `bits=12, group_size=128` 的量化矩阵，并记录 `codes_sha256`。不改动 v1 的索引文件。
2. 等价选择定理：v1 的选择规则是"模态内取 32，再用全体剩余的高分候选补齐到 48"，它最多只需要两部分候选：
   - 全体 top-80
   - 该模态内 top-32

   因此计算这两部分的并集，对并集用 reference 得分执行 v1 原有的选择函数，结果与"全量计算后选择"完全一致。需要编写测试证明这一点：随机构造 1000 组数据比较。
3. 两种模式：
   - `retrieval = shadow`：照常使用 float32 cosine 的 v1 结果；同时计算量化后的 progressive 选择结果和量化后的全量选择结果，断言两者一致；再记录量化候选与 float32 候选的 `overlap@48`。
   - `retrieval = replace`：使用量化后的 progressive 选择结果进入 rerank。这会改变实验结果，需要单独作为一组实验运行。
4. 可见性 mask 通过 `row_mask` 传入。
5. 新增输出字段 `progressive_retrieval_stats`：每个 demand 的轮数、字节比例和 `overlap@48`。

**验收：**
- shadow 模式下断言全部通过。
- 开关关闭时回归检查通过。

### T2.4 时序与容量模型扩展

**需阅读的文件：** `pim_backend/hyperpim_timeline.py`、`pim_backend/hyperpim_sc.py`

**规则：**
- 不得修改任何现有函数的签名和返回值，只能新增函数。
- 新增测试，断言现有函数在固定输入下的输出与 v1 完全相同。做法是先用 v1 代码生成快照，再进行比较。

**新增函数：**

```python
def plane_major_layout_plan(rows, cols, bits=12, group_size=128,
                            row_bytes=2048, tile_rows=16, bank_count=256) -> dict
    # 每个 bank、每个平面占用的 DRAM 行数；全精度读取的行激活数（须与 row-major 对比输出）

def progressive_topk_cycles(layout_plan, rounds: list[dict], timing_profile,
                            host_round_trip_cycles: int | None) -> dict
    # 依据 T2.1 stats 的 rounds 计算：每轮按幸存行涉及的 tile 计行激活，
    # 内部读取周期、bank 内 MAC/界计算周期、每轮一次阈值广播（host_round_trip_cycles）。
    # 输出逐轮与总和，并同时输出同矩阵全精度 GEMV 的周期（调用现有模型）作为对照。

def retrieval_gemv_workload(n_records, dim=1024) -> dict
    # 把检索打分作为 PIM 负载接入现有时序模型

def energy_model(counts: dict, energy_params: dict) -> dict
    # energy_params 的所有项默认为 None；任何项为 None 时返回 {"status": "params_missing", "missing": [...]}
    # 不得填写任何数值。参数名至少包括：
    #   e_act_pj_per_row, e_read_pj_per_byte_internal, e_io_pj_per_byte,
    #   e_mac_bf16_pj, e_bound_update_pj, e_refresh_pj_per_cycle
```

`host_round_trip_cycles` 默认为 `None`，此时必须报错。把它写入 `configs/pim_profile_v2.yaml`，标注 `TODO_CITE`，由用户填写。

**新增脚本：** `scripts/pim_replay.py <trace_dir> <stats_dir>`。读取 T1.1 的 trace 和 T2.2/T2.3 的统计，对每个问题、每个阶段输出四项周期估计：
- 全精度 GEMV
- 仅语法掩码
- 语法掩码 + 渐进
- 检索部分

---

## Phase 3：校准、保形候选宽度与 noisy-or 证据选择

### T3.1 候选导出（serving 侧，不读标签）

当 `candidate_dump` 非空时，每个问题写一个 `<dump>/<user>/<qid>.json`。内容包括：
- question_id、split
- 每个 demand 的 48 个候选，每个候选包含：`record_id`、`object_ids`、`session_ids`、`modality`、`retrieval_cosine`、`rerank_score`、`query_rank`、`retrieval_rank`
- 见证构造的输出：接受的见证及其 `record_ids`

`session_ids` 字段需要从 source 模型中取出每个 object 的 session。

### T3.2 校准器拟合（eval-only）

新建 `scripts/fit_calibration.py --dump <dir> --labels <path> --out <calibrator.json>`。

**数据切分：**
- 只允许使用 split = development 的问题。遇到其他 split 时 raise。
- 输出中写明使用的 user 列表。

**正样本定义：** 候选的 `session_ids` 与该问题标注证据的 session 有交集，或候选的图片 sha256 属于标注证据图片。映射方式复用 T1.3。

**校准方法：**
- 使用 `sklearn.isotonic.IsotonicRegression(increasing=True, out_of_bounds="clip")`。
- 按 modality 分别拟合：text 与 image。
- 输出概率裁剪到 `[1e-4, 1 - 1e-4]`。

**保形阈值。** 对每个校准问题 q，定义：

```
g_q = min_{e ∈ 标注证据单元} max{ rerank_score(c) : c 是 q 的候选且 c 匹配 e }
```

若某个 e 没有任何匹配候选，则 `g_q = -inf`。

对每个 α ∈ {0.05, 0.1, 0.2}：
- 将 g 升序排序，取第 `⌊α(n+1)⌋` 个（从 1 开始计数）作为 `τ_α`。
- 若 `⌊α(n+1)⌋ < 1`，则 `τ_α = -inf`。

**诊断输出：** 写入 calibrator.json 的 `diagnostics` 字段，包括：
- 可靠性分箱（10 箱）与 ECE
- n
- `g_q = -inf` 的比例，即候选池本身漏召回的比例

**产物：** calibrator.json 包含断点、τ 表、diagnostics 和 sha256。

注释中只允许写：覆盖保证以问题可交换为前提，且只针对 48 个候选池。不得写更强的结论。

### T3.3 保形候选策略（serving 侧）

当 `candidate_policy = conformal` 时：
1. 加载 calibrator，并校验 sha256。
2. 对每个 demand，保留 `rerank_score ≥ τ_α` 的候选。
3. 若保留数少于 `min_per_demand`，按分数补足到 `min_per_demand`。
4. 上限仍为 48。
5. 输出字段 `candidate_policy_stats`：每个 demand 保留的候选数，以及见证构造输入 token 数的变化。

新增 eval 脚本 `scripts/eval_conformal_coverage.py`：在 validation split 上计算经验覆盖率，即 `g_q ≥ τ_α` 的比例，并与 1-α 对比。

### T3.4 noisy-or 证据选择：`scripts/omni_proof_join_noisy_or.py`

**需阅读的文件：**
- `omni_proof_join.py`（`_compatible()`、`evaluate()`、输出 plan 的 schema）

**定义：**
- 见证有效概率：`p_w = min_{record ∈ w} cal(rerank_score(record))`。注释中写明这是启发式定义。
- demand 覆盖：`f_d(S) = 1 - Π_{w ∈ S∩W_d} (1 - p_w)`。
- 目标：在 `evaluate(∪O(w))` 可行且见证两两兼容的前提下，最大化 `min_d f_d(S)`。

**算法（SATURATE 式）：**
1. 候选阈值集合 T：取 `levels` 个等分点，即 `{i/levels}` 中小于等于 `max_d max_w p_w` 的值，并加入所有不同的 `p_w` 值。对 T 二分。
2. 对给定 t，令 `F_t(S) = Σ_d min(f_d(S), t)`。从 `S = ∅` 开始做贪心：
   - 在所有未选、与 S 兼容、加入后 `evaluate` 可行、且满足 `max_witnesses_per_demand` 的见证中，选择 `(F_t(S∪w) - F_t(S)) / max(1, Δcost)` 最大的 w。
   - `Δcost` 为精确 processor token 的增量，复用 evaluate 缓存。
   - 平局按 `witness_id` 字典序打破。
   - 当 `F_t = |D|·t` 时成功；没有可增益的候选时失败。
3. 取成功的最大 t 记为 t*，其对应的 S 为解。
4. 若 `fill_redundancy` 为 true，在预算内继续贪心最大化 `Σ_d f_d(S)`。
5. 若任何 t 都不成功，或 `parse_status != ok`，调用 v1 的 `select_retrieval_package()`，与 v1 行为一致。

**输出：** 与 v1 plan 的 schema 相同，并新增以下字段：
- `join_algorithm`
- `t_star`
- `per_demand_coverage`
- `witnesses_per_demand`
- `n_evaluate_calls`

`proof_status`、`selection_mode` 和 `semantic_proof` 沿用 v1 的取值规则。

**测试：**
- 合成实例：小于等于 12 个见证、3 个 demand，成本为对象并集上的加和，并含 binding 冲突。用暴力枚举求出精确最优的 `max min f_d`，然后：
  - 断言算法解可行且两两兼容；
  - 断言算法解不超预算；
  - 记录与最优的比值（只记录，不断言下界）；
  - 断言结果是确定性的。
- 断言 `join_algorithm = dfs_maxmin` 时调用的是 v1 原函数，行为不变。

---

## Phase 4（可选，Phase 1–3 完成并经用户确认后再做）

### T4.1 视觉特征缓存（精确）
1. 先阅读 Reader 模型的 forward，查明能否注入预计算的视觉特征，例如通过 `inputs_embeds` 或视觉模块的输出。
2. 如果没有干净的注入点，只实现每张图片视觉编码耗时与 token 数的测量，然后停止并报告。
3. 如果有注入点：
   - 缓存键为 `sha256(图片字节) + processor 配置哈希 + 模型文件哈希`。
   - 在 3 个问题上断言：开启缓存与关闭缓存的 prefill logits 完全相等，比较方式为 `torch.equal`。

### T4.2 对象复用统计（不读标签）
新建 `scripts/report_object_reuse.py <run_dir>`。按用户统计：
- 对象被选中的频次分布。
- 重复对象的 token 占比。
- 理想对象级 KV 缓存下可省去的 prefill token 上界。
- 容量为 C 的 LRU 缓存下可省去的 token，C 取多个值，按问题顺序模拟。

注意：这是分析估计，不是实现。

### T4.3 真机重放脚手架（由用户在手机上执行）
新建 `device/replay_from_trace.py`，从 T1.1 的 trace 生成每个阶段的 `(prefill_tokens, decode_tokens, n_images)` 列表。

新建 `device/android_bench.sh`，通过 adb 推送并调用 llama.cpp 的基准工具，按 trace 中的长度组合运行。要求：
- 模型路径、二进制路径和线程数都作为参数传入。
- 不要假设特定模型已被 llama.cpp 支持。
- 不得在报告中写任何未经实测的数值。

---

## 第 5 部分：实验运行清单（全部任务完成后）

为每一行新建一个配置文件 `configs/v2_<name>.yaml`。除表中列出的开关外，其余与主配置相同。

| 名称 | 关键开关 |
|---|---|
| `v2_base` | 全部默认（应与 v1 一致） |
| `v2_asof` | `visibility_protocol: as_of_question` |
| `v2_asof_op` | 上一行 + `witness_values: on`、`operation_executor: on` |
| `v2_asof_op_conf` | 上一行 + `candidate_policy: conformal`、`alpha: 0.1` |
| `v2_asof_op_conf_nor` | 上一行 + `join_algorithm: noisy_or_saturate` |
| `v2_full_shadow` | 上一行 + `progressive_topk.lm_head: shadow`、`retrieval: shadow`、`stage_profiler: on`、`workload_trace: on` |
| `v2_full_retq` | 上一行，但 `retrieval: replace` |

每一行运行完成后，依次执行以下脚本：
1. `report_run_stats.py`
2. `eval_evidence_recall.py`
3. `judge_incremental.py`
4. `pim_replay.py`（仅 shadow 行）

calibrator 只用 `v2_asof_op` 在 development split 上的 dump 拟合，所有指标只在 validation split 上报告。

---

## 第 6 部分：停止并询问用户的情形

遇到以下任一情况，必须停止并报告，不要自行决定：

1. 现有代码与本计划的描述不一致，且不一致之处影响接口。
2. 回归检查失败，且原因不在 `NONDETERMINISTIC_FIELDS.json` 中。
3. 定理级断言失败，即 T2.1、T2.2、T2.3 中的精确性断言。
4. 内存或显存不足，需要改变算法才能继续。
5. 标签 schema 无法映射到 session 级证据。
6. 需要填写任何文献常数。
