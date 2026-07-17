# Method 1 FSA-QEHG Overview Figure Design

## Purpose

Create one publication-quality overview figure for Method 1, the frozen
FSA-QEHG sparse semantic hypergraph retrieval path. The figure should explain
the deployable data flow and the evaluation boundary. It must not imply a PIM
hardware implementation or a PIM latency result.

## Recommended Composition

Use a left-to-right pipeline with four visually separated bands:

1. **Frozen data and representation**
2. **Sparse online candidate construction**
3. **Shared-budget selectors and outputs**
4. **OOF evaluation, gates, and locked follow-up**

Place a small top banner above the pipeline:

`Method 1: FSA-QEHG — Frozen Semantic Assignment + Query-conditioned Evidence Hypergraph`

Use solid arrows for deployable information flow, dashed arrows for offline
training/reference inputs, and a lock icon for calibration and formal test
inputs that are not read during development.

## Band 1: Frozen Data and Representation

Draw three source blocks:

- **Memory store**: `42,034 memory items`, frozen memory embeddings, payload
  bytes, persona/session eligibility metadata.
- **Query stream**: frozen query embeddings and train-only session-blocked OOF
  query plans.
- **Neutral layout**: source-order layout, `capacity=16`, `2,628 logical
  Tiles`; label this explicitly as a `host/logical access metric`.

Flow the memory embeddings through:

`UOT-FSA projection -> 64 frozen semantic prototypes -> nonnegative sparse
memory-to-prototype incidence`

Add a small annotation below this block:

`Gate A passed: UOT-FSA; Gate B2 passed: QESA representation validity`

Also show a thin gray note that the original retrieval Gate B negative result
is retained; Gate B2 does not establish an end-to-end retrieval advantage.

## Band 2: Sparse Online Candidate Construction

Use five numbered stages, each as a compact rounded rectangle:

1. **Protected ANN seeds**
   - Preloaded batch-1 ANN prefix.
   - Select five protected eligible memory seeds.
   - No evidence labels or gold bundle input.

2. **QESA r2 prediction**
   - Query-conditioned latent aspect probabilities.
   - Active prototype grid: `4 / 8 / 16`.
   - Mark prediction latency as a measured GPU stage.

3. **Sparse semantic inverted-index lookup**
   - Prototype rank index shape: `[64, 42,034]`, `int32`.
   - Scan only activated rank-list prefixes with eligibility membership tests.
   - Stop after `8 / 16 / 32` eligible members per edge.
   - Record `activated_semantic_edges` and `incidence_entries_visited`.
   - Add a red crossed-out annotation: `No dense eligible-by-prototype incidence scan`.

4. **Bounded semantic bundle beam**
   - Build arity-3 tuples of distinct activated semantic edges.
   - Distinct-member constraint: one memory cannot fill two semantic aspects.
   - Beam grid: `8 / 16 / 32`.
   - Explicitly label tuples as `semantic candidate constructions`, not observed
     evidence bundles or co-access events.

5. **r3 sparse frontier**
   - Base union: `protected ANN seeds + unique retained bundle members`.
   - Bounded fallback: append only unique members already returned by the same
     activated-edge rank-prefix lookups.
   - Never extend a rank prefix and never scan dense incidence.
   - Record `semantic_bundle_unique_members`, `fallback_members_appended`, and
     `budget_infeasible`.

Show the final candidate budgets under the frontier as nested brackets:

`S8 subset S16 subset S32`, with `B = 8 / 16 / 32`.

## Band 3: Shared Frontier, Selectors, and Outputs

Branch the same sparse frontier into four selector blocks. Draw the shared
frontier as a single thick horizontal line before the branch to make the
fairness contract visually obvious.

- **ANN flat**: cosine ranking on the shared frontier.
- **Same-capacity neural flat**: frozen-capacity neural flat ranking.
- **Sparse semantic-bundle hypergraph**: frozen/outer-train QHDS monotone
  greedy selection on sparse incidence.
- **Budgeted pairwise factor graph**: nonnegative unary + symmetric pairwise
  factors trained only from outer-train valid distinct-member bundles projected
  to unordered pairs.

Add a fifth block below or in a pale gray inset:

- **Dense QHDS reference**: dense reference only; explicitly marked
  `reference path, not efficient online path` and `dense_online_scan=true`.

All four sparse selectors must show the same frontier and the same protected
seeds. Connect each selector to a common output panel containing:

- quality: `group coverage`, `all groups covered`, `K`, `K>=2`, `final top-5`
- cost: metadata bytes, UTF-8 payload bytes, logical Tile activations,
  redundant Tile activations
- latency: query encode, ANN, QESA, sparse lookup, bundle build/score,
  selection, payload gather, rerank, end-to-end

Put a small `rerank_executed=false; rerank_ms=0` badge on the current OOF
latency output. Put `embedding_precomputed=true; query_encode_ms=0` beside it.

## Band 4: Evaluation and Closure Boundary

Draw a bottom evaluation rail with these blocks:

`5 seeds x 5 session-blocked outer folds`
`-> fixed count / actual UTF-8 fixed byte / iso-quality / iso-cost`
`-> logical Tile and online latency summaries`
`-> Pareto frontier`
`-> paired query bootstrap + persona/session clustered bootstrap`
`-> Holm correction for preregistered Gate C contrasts`

After the rail, draw a decision diamond:

`Gate C: sparse hypergraph non-degradation and quality/cost/latency advantage
over the budgeted pairwise graph?`

From the diamond, use two branches:

- **Pass/continue**: dynamic data feasibility audit -> dynamic update
  comparators or a static FSA-QEHG fallback -> pre-calibration freeze -> one
  calibration CRC/LTT mapping -> final freeze -> one formal test -> closure
  manifest.
- **Fail**: retain the static FSA-QEHG fallback and all negative results; do
  not fabricate access order or co-access traces.

Place a lock banner over calibration and formal test:

`Development status: calibration access = 0; formal-test runs = 0`

## Negative-Result Callout

Use a clearly separated gray/red callout, not a success arrow:

`Learned DeepSets bundle locality is a retained diagnostic ablation only`

Include the compact evidence:

- `K>=2 all-groups-covered effect = -0.008092`
- clustered 95% CI `[-0.013258, -0.003466]`
- bundle recall effect `-0.023603`
- exact cardinality rate `25.43%`

Do not route this component into the sparse primary branch.

## Visual and Wording Constraints

- Prefer a restrained scientific palette: deep teal for frozen
  representation, blue for sparse runtime, orange for selectors, gray for
  references/negative results, red only for prohibited scans and locks.
- Keep labels short; use icons for ANN, prototype/index, graph, latency,
  Tile, lock, and statistics.
- Avoid claims such as `PIM hardware access`, `PIM latency`, `true evidence
  bundle`, `co-access`, `access sequence`, or `online trace`.
- The Tile label must read `logical Tile / host access metric`.
- Do not draw a fabricated timeline, Hawkes process, LACA/OACH module, access
  order chain, or synthetic co-access trace.
- Do not depict calibration or formal-test examples; show only the lock and
  zero-access counters.

## Figure Caption Draft

**Overview of Method 1 FSA-QEHG.** Frozen UOT-FSA semantic prototypes and
QESA query-conditioned probabilities drive bounded sparse rank-prefix lookup
and arity-3 semantic candidate construction. The r3 frontier combines five
protected ANN seeds with retained sparse bundle members and a fallback drawn
only from already returned edge members. ANN flat, same-capacity neural flat,
sparse QHDS hypergraph, and budgeted pairwise factor-graph selectors consume
the same frontier and nested budgets, while dense QHDS remains a labelled
reference. Quality, UTF-8 payload cost, logical host/Tile locality, and batch-1
latency are evaluated with five-seed session-blocked OOF statistics before any
locked calibration or formal-test access.

## Direct GPT Figure Prompt (Chinese)

生成一张横向、论文风格的 Method 1 overview architecture figure，标题为：
“Method 1 FSA-QEHG：Frozen Semantic Assignment + Query-conditioned Evidence
Hypergraph”。画面采用从左到右的四段式数据流，白色背景、细线、扁平矢量风格、
高可读性无衬线字体，不使用 3D、照片、渐变、发光或装饰性插图。所有文字保持
简短，数字和英文缩写必须按原样保留。

第一段标为“Frozen representation”。画出 Memory store、Query stream、
Neutral layout 三个输入块：42,034 memory items；frozen memory/query
embeddings；capacity=16、2,628 logical Tiles，并把 Tile 明确写成
“logical Tile / host access metric”。Memory embeddings 经过
“UOT-FSA projection”进入 64 个 frozen semantic prototypes，再形成
nonnegative sparse memory-to-prototype incidence。旁边加小标签
“Gate A passed”与“Gate B2 representation validity passed”，并保留一条
灰色注释“original retrieval Gate B: negative result retained”。

第二段标为“Sparse online candidate construction”，按 1 到 5 编号排列：
(1) Protected ANN seeds：preloaded batch=1，5 protected seeds；
(2) QESA r2：query-conditioned prototype probabilities，active prototypes
4/8/16；(3) Sparse semantic inverted index：rank index [64, 42,034]，只扫描
activated rank-list prefix，members per edge 8/16/32，记录
incidence_entries_visited；用红色禁止标记写“No dense eligible-by-prototype
incidence scan”；(4) Bounded semantic bundle beam：arity=3、beam 8/16/32，
distinct activated edges 与 distinct memory members，只称“semantic candidate
construction”，不能称 observed evidence bundle 或 co-access；(5) r3 sparse
frontier：protected ANN seeds 与 retained bundle members 的 union，再从同一批
已经返回的 edge members 做 bounded fallback。标明“no prefix extension / no
dense scan”，并展示 nested budgets S8 ⊆ S16 ⊆ S32、B=8/16/32。

第三段标为“Shared frontier and selectors”。先画一条粗的 shared sparse
frontier 线，再分成四个公平比较分支：ANN flat、same-capacity neural flat、
sparse semantic-bundle hypergraph（QHDS monotone greedy）、budgeted pairwise
factor graph（nonnegative unary + symmetric pairwise）。四个分支必须连接到同一
frontier 和同一 protected seeds。另画一个灰色的 Dense QHDS reference，注明
“reference path, not efficient online path”。所有分支汇聚到统一输出面板，列出
quality、UTF-8 payload/metadata bytes、logical Tile activations、batch=1
latency。输出面板加两个状态标签：“embedding_precomputed=true;
query_encode_ms=0”和“rerank_executed=false; rerank_ms=0”。

第四段标为“OOF evaluation and closure”。画出流程：5 seeds × 5
session-blocked outer folds → fixed-count / fixed-byte (actual UTF-8) /
iso-quality / iso-cost → logical Tile + CPU/GPU latency → Pareto frontier →
paired query bootstrap + session-clustered bootstrap → Holm correction →
Gate C。Gate C 之后再连接 dynamic data feasibility audit、dynamic update
comparators 或 static FSA-QEHG fallback、pre-calibration freeze、一次
calibration CRC/LTT、final freeze、一次 formal test、closure manifest。校准和
formal test 上方放锁形图标与横幅：“calibration access=0; formal-test
runs=0 during development”。

右下角单独放灰红色负结果 callout，不接入 primary branch：
“DeepSets learned bundle locality: diagnostic ablation only”；列出
K>=2 all-groups-covered effect=-0.008092，clustered 95% CI
[-0.013258,-0.003466]，bundle recall effect=-0.023603，exact cardinality
rate=25.43%。

配色：teal 表示 frozen representation，blue 表示 sparse runtime，orange
表示 selectors，gray 表示 reference/negative result，red 只表示禁止扫描和
锁定边界。实线表示 deployable online flow，虚线表示 offline training 或
reference input。禁止绘制 PIM hardware、PIM latency、access sequence、
Hawkes、LACA/OACH、伪造 co-access trace、gold bundle 或 calibration/formal
test 样例；不得把 logical Tile 写成硬件结果。
