# Omni main-experiment closure: active target version

## W12 bank closure entry (2026-10-29)

The active PIM path is now W12A16: a uniform 12-bit bit-sliced, row-packed,
single-copy bank-resident decoder shared by parser, grounding, and Reader. The
192-request development gate and the 300-request sealed quality gate pass under
the frozen local 8B Judge protocol. A real-activation functional bank smoke also
passes: 249 Language Linears, 10,458 decode bank calls, zero decode fallback,
shadow argmax 10,450/10,458, and max relative L2 0.4087%.

The dispatcher is now connected to the real three-stage mainline. Six
development questions executed parser, grounding, and final Reader from one
shared layout with 374,247 bank calls and zero decode fallback. This exposed a
numerical-contract failure that the single-stage smoke could not see:
`fp32_bank` changed parser decisions on 1/6 requests, grounding and selected
evidence on 3/6, and final answers on 1/6. `bf16_reference` restored parser
parity to 6/6 but grounding/evidence remained 4/6 and answers 5/6. The mainline
default is therefore `bf16_reference`; `fp32_bank` is diagnostic only. The next
architecture gate is a bank-local BF16 reconstruction datapath whose cost is
charged in the target model. Details:
`reports/HYPERPIM_SC_W12_MAINLINE_STAGE_20261029_ZH.md`.

The authoritative entry is
`artifacts/hyperpim_sc_w12_closure_20261029/closure_manifest.json` with hashes in
`sha256sums.txt`. The current Chinese report is
`reports/HYPERPIM_SC_W12_BANK_CLOSURE_20261029_ZH.md`. The request-level target
projection still passes the Host-control lower bound in the joint-worst case
(`1.352x` service, `1.354x` TTFT), but physical PIM, bank prefill, bank-replace
quality on the sealed 300 requests, BF16-reconstruction overhead, and measured
end-to-end latency remain
unverified. Functional Torch wall time is never PIM latency. The old mixed
traffic statement (`391 TB / 575x`) is retired; the audited BF16/W12/PIM-external
ledger is `781 TB / 586 TB / 680 GB`, with `1,148.76x` and `861.57x` reductions
respectively.

## Historical Reader/lm_head PIM decision (2026-10-28; superseded by W12)

The current Reader/`lm_head` PIM integration is **no-go for the 200-question
main path**. Its 4-request functional replacement preserves generated outputs
but has measured server functional-emulation ratios of 0.9865x for service and
0.9832x for TTFT; this is not physical PIM latency. A separate, deliberately
favorable request-DAG sensitivity projection reports 1.019x service and 1.165x
TTFT, but assumes all decode/TTFT is eligible and zeroes PIM boundary, transfer,
and contention costs. Because its phase model is uncalibrated, this is neither
a measurement nor a strict hardware upper bound. It still does not establish
the required 1.05x complete-service gain.

No PIM route is enabled in the 200-question runner, and the existing answer,
evidence, and memory-organization path remains unchanged. Preserve prior PIM
artifacts as negative qualification evidence; do not expand capacity, rerun the
200 questions, or continue Reader-backend tuning. Reopen only for a distinct
workload with observed memory-bound critical-path evidence, a validated target
PIM model, and a credible path to positive end-to-end benefit. Details:
`reports/PIM_BACKEND_NO_GO_20261028_ZH.md`. The existing-trace audit found no
independent workload passing Gate 1; see
`reports/PIM_NEW_WORKLOAD_QUALIFICATION_20261028_ZH.md`.

## Unified hybrid-state PIM qualification (2026-10-27)

The current candidate treats full-attention KV and linear recurrent/conv state
as one capacity tier. Bank-local KV QK/AV and recurrent updates are modeled;
projections, softmax, cross-bank reduction, MLPs, nonlinearities, and semantic
Reader work remain Host/GPU-owned. The latest qualification is
`artifacts/hybrid_state_pim_unified_qualification_20261027_v5.json`. Frozen trace
residency replays are `artifacts/hybrid_state_pim_unified_tiers_grouped_20261027_v3.json`
and `artifacts/hybrid_state_pim_unified_tiers_round_robin_20261027_v3.json`.
The Chinese report is `reports/HYBRID_STATE_PIM_REFACTOR_20261027_ZH.md`.

This version fails its current 24K/32-agent capacity gate: 32 complete states
need about 26.8 GB, while the assumed 2 GiB pool holds two; the frozen grouped
and round-robin replays produce zero PIM-served accesses and substantial
promotion/writeback traffic. Candidate E2E latency and speedup remain unknown.
The design is not connected to the 200-question path, which has no cross-request
model-native persistent-state identity.

This is a label-free capacity/functional/phase-model qualification, not measured
PIM E2E speedup. A new stateful serving workload is required before any paired
latency experiment.

## Weight-resident microbatch PIM gate (2026-10-27)

The next candidate is weight-resident decode microbatching, not cross-request KV
reuse. It requires naturally co-ready requests in one observed online scheduler
queue. The 200-question replay has no online arrival/ready-queue snapshots, so its
request order and service durations cannot qualify concurrency. The strict audit
is implemented in `pim_backend/microbatch_opportunity.py`; the existing replay
is recorded as `artifacts/pim_microbatch_legacy_trace_audit_20261027.json` and
does not allow a paired PIM smoke. No PIM route or speedup is enabled. See
`reports/PIM_MICROBATCH_QUALIFICATION_20261027_ZH.md` for the trace contract and
next gates.

The 200-question runner assigns separate user jobs/model instances to GPUs and
processes questions serially within each shard; it does not expose a shared online
decode queue. Re-running that offline evaluation cannot qualify this design. A
real serving runtime and observed scheduler snapshots are prerequisites; do not
infer batching from GPU-level parallelism or synthesize arrival times.
The project `omni` Conda environment can run the existing Transformers inference
path, but no vLLM/SGLang shared-weight serving scheduler or observed online
request queue is available. The default Python environment has a separate
Transformers/`huggingface-hub` version conflict; that does not block offline
server-side closure work in `omni`. Real scheduler traces remain a prerequisite
for microbatch qualification; no PIM microbatch performance run is authorized.

## Evidence-state prefilter boundary (2026-10-27)

PIM is not used as a dense-retrieval accelerator. The opt-in
`--evidence-state-prefilter-shadow` path widens a qualification candidate pool,
checks exact provenance/binding metadata before Host grounding, and fails open
when the structural contract is incomplete. It does not replace semantic
grounding or alter the default answer path. No PIM speedup is claimed until the
shadow path demonstrates baseline witness recall and a net end-to-end benefit.

## Direction decision (2026-10-25)

The current candidate does **not** establish method effectiveness or a three-module
closure. Keep one research hypothesis: source-located, semantically sufficient
complementary-evidence compilation with AND/OR witness search, unique-object cost,
and exact Reader budgeting. The fixed 12-query post-run shape review matches 8/12;
four semantic mismatches remain, and this review is not independent accuracy evidence.
Stop further prompt tuning on this same query set; do not expand this candidate to
the full 192 requests. Lifecycle remains accounting support, while PIM is excluded
from the primary claim because all 192 requests used Host and Reader decode is only
about 8.7% of the measured service time. See
`reports/PROOF_DIRECTION_PILOT12_REVIEW_20261025_ZH.md` and
`artifacts/proof_direction_completion_audit_v1_20261025.json`.

The v3 query-only diagnostic used registered development questions only and did not
read history, answer labels, validation or holdout, or launch Reader answers, Judge,
comparison, ablation or a full run. This does not retroactively certify the earlier
Reader pilot: its entry point loaded merged all-user history before filtering, and
its Judge scanned/hashed merged manifest/label bytes. The strict isolation caveat is
recorded in `artifacts/proof_direction_pilot12_v1_20261025/input_isolation_review.json`.
`method_effectiveness_established=false` and `main_experiment_closed=false` remain.

## Atom/region recovery result (2026-10-25)

The staged candidate in `artifacts/proof_atom_region_v4_schema_20261025/` freezes
query-only logical decomposition, atom/region retrieval units, source-located
AND/OR witnesses, unique physical-object costs, and an explicitly nonproof
retrieval package when no complete witness join exists. It does not classify
evidence availability in the query parser or turn a failed join into a top-1
proof. The previous candidate and all its negative results remain preserved.

The 50 focused proof/provenance/Reader/evaluation/reporting tests pass. The first six
requests complete without label or Judge access: all structured generations
finish as JSON; five requests use source-located joins and one explicitly uses
a nonproof package after rejecting duplicate query obligations. The full
192-request frozen run and unchanged 8B Judge complete at **61/192 (31.77%)**,
with zero invalid judgments and all 192 Reader generations ending at EOS.
This is a descriptive recovery from the rejected 49/192 candidate, not a
matched causal improvement or a positive paper-level closure. The source
validator accepts joins for 163 requests; 29 use explicit nonproof packages.
Only 13 valid query graphs contain multiple demands, eight reach a located
join, and no query relation survives validation. Source location is not
semantic entailment, and multi-record selection is not method sufficiency.
The earlier v3 smoke was stopped on parser repetition before any full run or
Judge; its artifacts and failure report remain preserved. Bounded schema
decoding is a correctness facility, not an algorithmic contribution.
No new baseline, ablation, validation/holdout, or physical PIM run is enabled.
Final text/image/input budgets pass for every request. The authoritative
post-run review is `artifacts/proof_atom_region_v4_schema_20261025/scientific_review.json`;
see `reports/PROOF_ATOM_REGION_V4_RESULTS_20261025_ZH.md` for cost and claim limits.
`main_experiment_closed=false`; no causal E2E or PIM speedup is claimed.

## Recovery candidate (2026-10-25)

The frozen recovery run and fixed 8B Judge are complete: **49/192 (25.52%)**,
with no invalid Judge responses. This candidate is rejected, not a positive
main-experiment closure. All query graphs lack hard demands and relations;
181 requests fall back to one source record, while 10 of the 11 nonfallback
witnesses contain multiple records. Selected annotated image-reference recall
is zero. Source location is not semantic proof. Full results, costs and the
remaining algorithm/system gaps are documented in
`reports/PROOF_RECOVERY_RESULTS_20261025_ZH.md`; the authoritative post-run audit
is `artifacts/proof_recovery_dev3x64_contract_v2_20261025/scientific_review_corrected.json`.

The historical recovery candidate is implemented in its frozen snapshot and is
registered in `artifacts/proof_recovery_dev3x64_contract_v2_20261025/`. It separates
source containers, source-located text/image records, and unique Reader
materialization objects; visual locators are built offline without QA inputs,
and the Reader receives the original image rather than a generated caption.
The online path is query parsing, frozen text retrieval/reranking, source
grounding, bounded proof joining with shared-object costs, exact processor
budget validation, Reader generation, and lifecycle/backend event logging.

The candidate remains development-only and `main_experiment_closed=false`.
No baseline, ablation, validation/holdout access, physical PIM execution, or
end-to-end PIM speedup claim is enabled. The recovery protocol and boundaries
are recorded in `reports/PROOF_RECOVERY_IMPLEMENTATION_20261025_ZH.md`.
The first, incomplete `proof_recovery_dev3x64_v1_20261025` serving attempt was
stopped with 62 unscored requests after detecting extraction-template echoes
and an overlong witness serialization contract. Its raw extraction and partial
outputs remain preserved; the corrected run does not select between scores.

## Previous evidence-view execution status (2026-10-25)

The registered v2 evidence-view comparison is complete on the frozen 192-request
development queue. The 8B proxy Judge scores **83/192 reference versus 66/192
candidate** (paired delta -8.85 pp; user-conditional 95% bootstrap interval
[-16.67, -0.52] pp; exact McNemar p=0.043). One candidate Judge response is invalid
and counted wrong. Candidate image-reference recall rises over v1, but stays below
reference. A post-run audit shows **zero image views among 12,288 winning top-64
views**: the recall change came from text-only views of image-bearing exchanges,
not image-vector retrieval. Candidate planning takes 306.6 cumulative seconds
versus 16.1 s for reference; the shared-host service ratio is only screening.
The outcome rejects this candidate; it is not an independent validation result.
See reports/ACTIVE_MAINLINE_20261024_ZH.md and
reports/MOBICOM_REVIEW_AND_RECOVERY_PLAN_20261025_ZH.md. The main experiment
remains open; validation/holdout stay sealed and PIM remains modeled, not measured.

The 2026-10-23 quota, token-cost, and functional PIM results below are historical
mechanism results; none qualifies as a positive end-to-end contribution.

## Historical execution status (2026-10-23)

The 384-request prospective development cohort is complete: strict paired judge
scores are **155/384 control versus 157/384 candidate image quota**, with a
paired difference of +0.52 pp (within-user conditional bootstrap 95% interval
[-1.82, +2.86] pp). The quota does not qualify as a quality improvement.
Exact query-local atom token-cost reuse preserves all 384 evidence plans,
processor inputs and generated token sequences. Its planner cost falls from
1705.89 s to 38.79 s; the concurrent-host 1.270x service ratio is **not causal
E2E evidence**. Free removal of that planner stage bounds this server cohort
at 1.260x, so the mechanism becomes an optimized baseline rather than a claimed
standalone 1.3x contribution.

The new `pim_backend/pimnast_functional.py` packs explicit CR/CRO-1 16x16
INT8 tiles into the pinned 128-bank, 256-byte-interleaved PIMnast reference.
Five full Reader projections yield **276,480 exact INT32 outputs** matching
independent INT64 and the CUDA shared-primary contract. A complete natural
development request then produces **247 identical greedy tokens and the same
EOS**, with all 80 selected layouts exercised. See
`artifacts/pimnast_int8_reader_functional_v3_20261023.json` for frozen source,
model/processor identity, raw integer tensors and full token sequences.
This is functional emulation, not hardware instructions or a latency model.
The 2,144,337,920-byte padded layout duplicates Host primary storage in this
functional implementation; target aliasing and complete capacity remain
unqualified. A duplicate-aware 12-GiB reference projection reports about
9.557–9.558 GB declared payload and 10.631–10.632 GB projected peak under
1-GiB OS plus 1-GiB runtime reserves; this is not a phone capacity result.
The GeniePIM 1.7661x number remains a **known-linear conditional model ratio**,
not measured phone/PIM or full-system speedup.

A 192-request, three-development-user natural-order functional service replay
completed at `artifacts/pimnast_functional_dev3x64_v1_20261023/`. The paired
audit found zero differences in registered inputs, processor digests, answer
tokens, or stop states; the pinned reference Judge therefore transfers 81/192
correct answers (79/192 when EOS is required), with all 10 truncated requests
retained in the denominator. The replay exercised 3,775,360 bank-layout
replacement calls across 80 layouts per user. This is a correctness and
functional-accounting result, not PIM latency: the emulator duplicates
2,144,337,920 bytes of Host primary storage and reports no hardware speedup.
An aligned Host-only control with query-local token reuse is complete. On the
same RTX3090 screening contract, Host-only totals 10,862.47 s versus
12,339.94 s for the functional backend (functional/Host = 1.136x); p50 is
51.56 s versus 59.35 s and p95 is 107.20 s versus 122.50 s. This is emulator
overhead, not PIM performance, because the functional path duplicates Host
weights and uses Torch bank-layout arithmetic. Validation users 3/4/5 and
holdout remain sealed.
`main_experiment_closed = false`. Current results and limitations are recorded
in `reports/MAINLINE_DECISION_20261022_ZH.md`.

## Historical execution status (2026-10-22)

The joint-visual screening has completed under
`artifacts/joint_visual_closure_dev3x16_v3_sharded_20261021/`, including the
Qwen3-VL-8B judge and paired audit. Under the matched 4-image/512-output budget,
baseline scores **15/48** and candidate **16/48** (3 wins, 2 losses, zero invalid
judgments). This does not establish a quality improvement: the exploratory
question-IID difference interval is -6.25 to +10.42 percentage points, with only
three development users. Candidate summed service time is **16138.579 s** versus
**2265.072 s**, about 7.12x slower in shared-host screening. Additional index
production costs **10914.792 s**; no amortization crossover exists at the observed
online costs. These are negative qualification results, not successful gains.

Online cost instrumentation and a three-request, label-free phase diagnostic
are registered in `artifacts/joint_visual_cost_profile_dev3x1_v1_20261022/`.
Its eight-output-token cap and synchronization overhead prohibit comparing its
E2E time or quality with the 48-request screening. See
`reports/JOINT_VISUAL_EXECUTION_20261021_ZH.md`. Main experiment remains open;
no measured PIM E2E speedup or device feasibility is established.

A same-request reranker batch-size probe did not qualify batch=4: scores changed
(maximum absolute difference 0.109375), top-16 order changed, and synchronized
forward time increased from 198.455 s to 217.940 s. Although the final evidence
and generated tokens matched for that one request, it is not promoted as an
optimization. Batch-4 vision encoding accounts for 216.988 s of 217.940 s
forward time in this single diagnostic. This motivates an exact feature-level
reuse opportunity probe, which is now complete for the frozen two-image prefix:
all pooled and DeepStack outputs differ between original-batch and per-image
encoding (pooled max absolute difference up to 1.90625). Exact per-image feature
reuse is therefore rejected, and approximate reuse has no quality qualification.
Controlled patch projection checks now reproduce a severe implementation-path
cost: warmed Conv3d takes roughly 19--20 seconds for the frozen two-image prefix,
whereas its algebraic Linear form takes under a millisecond. A full-vision check
falls from 19.746 seconds to 0.047 seconds, but pooled and DeepStack outputs drift
in BF16. This is not promoted, not an algorithmic/PIM contribution, and not an E2E
speedup. The next gate is a numerically requalified, equally optimized baseline
and candidate, not cache tuning or PIM projection against the pathological path.

The completed diagnostic, numerical rejection, and next execution gates are
documented in `reports/JOINT_VISUAL_COST_DIAGNOSIS_20261022_ZH.md`.

On October 22, a frozen-prefix cuDNN-autotune control still measures 19.103 s
for Conv3d versus 0.000592 s for Linear, with unequal BF16 visual outputs.
This excludes autotuning alone as a fix for that prefix, not all alternative
Conv implementations. A fresh full-budget 48-request-per-policy screen has
completed in `artifacts/joint_linear_requalification_dev3x16_v1_20261022` using
`scripts/run_linear_joint_requalification.sh`: unchanged Reader and embedding
index, Linear only in the candidate reranker, fresh baseline outputs, and the
same Qwen3-VL-8B judge. No state reuse or PIM is enabled. All 96 judgments are
valid: baseline 15/48 versus candidate 16/48, three wins and two losses.
Candidate service time remains 26.8% higher in this shared-host screen; neither
quality noninferiority nor E2E acceleration is established. Independent
protocol checks pass in `scripts/audit_linear_joint_requalification.py`.
See `reports/LINEAR_REQUALIFICATION_CLOSURE_20261022_ZH.md` for costs,
numerical drift, evidence failures, orchestration recovery, and next gates.

See `reports/MAIN_EXPERIMENT_EXECUTION_20261021_ZH.md` for the preceding target-version
text-only reranking execution results. The valid seven-category, label-independent 48-request
development cohort scored **14/48 baseline vs 11/48 text-reranked candidate**
with the explicitly selected Qwen3-VL-8B proxy judge (zero invalid labels).
Its 1.642x concurrent-screening service ratio is **not a successful speedup**:
quality regressed, fixed-GPU repeats are not qualified, and PIM E2E is unknown.
Multimodal relevance is an explicit qualification candidate, not a proven
deployment improvement. Reader remains Qwen3.5-4B; no training or fine-tuning.
`main_experiment_closed = false`. The older sections below are historical
development context; their results must not override this status.

## Active Protocol (2026-10-21)

The active scope is 16 users and 7,415 original Omni questions, reconstructed
by retaining every question for the upstream filtered release's user set.
All seven published category counts match. Prior development/validation/holdout
user boundaries are preserved. See
`artifacts/omni_paper_scope_v3_20261021/protocol_manifest.json`.

The Reader remains Qwen3.5-4B without training or quantization. The explicitly
user-selected judge is Qwen3-VL-8B-Instruct with the Omni evidence rubric;
this is a proxy score, not the official Qwen3-14B score. Invalid judgments
remain visible and are included in the conservative full denominator.

`scripts/omni_provenance_optimizer.py` uses structural provenance closure,
frozen Qwen3-Reranker-0.6B scores, actual processor image-grid token costs,
and integer constrained optimization. It does not use question-type routes,
keyword answer rules, dataset-fitted weights, or fine-tuning. Its optimality
applies only to the registered relevance surrogate, not task correctness.

The six-worker pilot is a correctness/quality screen, not an isolated final
latency benchmark. State reuse and PIM are not enabled in this screen.
Historical results below are copied provenance only and do not establish
target-version gains. The authoritative design is
`reports/MOBICOM_MAIN_EXPERIMENT_CLOSURE_BLUEPRINT_20261021_ZH.md`.

## Historical Source-Version Notes (Not Target Results)

## Current Research Priority (2026-10-21)

The user's 12 GB device specification only removes the obsolete 4 GiB
deployment restriction. It is not the workload definition, optimization
objective, or a main-experiment axis. Historical capacity artifacts remain
diagnostics; no capacity sweep or device migration is currently prioritized.
The active mainline is expensive-computation reuse, evidence quality, and
an execution-coupled PIM backend within one dependency-aware system.
See `reports/MAINLINE_REFOCUS_AND_CRITICAL_PATH_20261020_ZH.md`.
The latest exact-producer full-service pair and reverse-order repeat are
recorded in `reports/OMNI_EXACT_PRODUCER_E2E_CLOSURE_20261021_ZH.md`.
Exact producer provenance and store lifecycle pass for all 48 requests, with
2,181 vectors reused. Direct hyperedge embedding time falls by about 13 seconds
per cohort, but the first 1.089x full-service ratio is not reproduced by the
reverse-order run (1.0007x); no causal end-to-end speedup is claimed. The new
canonical numerical reference scores 37/48 under the local judge, so earlier
38/48 quality results are not inherited. PIM remains modeled-only and the main
experiment remains open.
The current canonical 48-request vision opportunity gate is recorded in
`reports/OMNI_VISION_OPPORTUNITY_GATE_20261021_ZH.md`. All reconstructed
processor tensors match the serving reference. Only one whole ordered image
batch recurs naturally; even free batch reuse bounds this cohort's full-service
gain at 1.01285x. Per-image partition output equivalence is unqualified. Visual
cache tuning is therefore not the next mainline optimization; the next execution
replay binds current evidence and producer outputs to exact processor operands
before rebuilding the residual Reader/PIM graph.
That canonical replay and independent route reconstruction now pass for all
48 requests; see `reports/OMNI_CANONICAL_RESIDUAL_PIM_CLOSURE_20261021_ZH.md`.
The current full-decode ledger covers 8,819 invocations and 2,195,931 Linear
calls. A horizon-free per-weight admission replay yields a 1.29961x conditional
known-Linear ratio, not a measured or full-system speedup. Its elementary
rent-or-buy certificate is an existing optimization tool, not an innovation
claim; actual Reader/PIM dispatch and complete unknown-work costs remain open.
The fixed three-user execution-graph closure is recorded in
`reports/OMNI_EXECUTION_GRAPH_COHORT_CLOSURE_20261020_ZH.md` and
`artifacts/omni_execution_graph_dev3x16_v1_20261020/cohort_audit.json`.
All `48/48` complete greedy outputs match their frozen references. Natural
input-prefix overlap is sparse; shared-input deduplication leaves the much
larger decode weight footprint unchanged. The first-decode graph is diagnostic
and may omit opaque kernel internals; PIM routing and cross-request Reader
state restoration remain unexecuted.

The first functional bank-layout Reader dispatch qualification is recorded in
`reports/OMNI_FUNCTIONAL_BANK_QUALIFICATION_20261021_ZH.md`. The synchronous
GPU bank emulation passes its FP32 numerical envelope but does not reproduce
the original GPU BF16 MLP outputs: all six fixed-prefix replacement token
sequences diverge, and the same-family local proxy scores 3/6 versus 5/6 for
the matched frozen answers. A full-matrix FP64 diagnosis shows that disabling
reduced-precision reduction substantially changes the original host MLP
outputs toward the FP64 reference; this does not explain all generation or
quality differences. The registered matched-arithmetic candidate disables
reduced-precision BF16 reduction only for selected decode Linears, with exact
state restoration and unchanged prefill/unknown-work paths. Only 1/6 complete
token sequences match the matched-host reference; local same-family quality is
3/6 versus 4/6, with one lost and no gained successes. The small fixed cohort
is not a population estimate, but it fails the observed nonregression gate, so
replacement expansion is stopped. No physical PIM or comparable end-to-end
speedup was measured; `end_to_end_speedup` remains null. See
`artifacts/omni_matched_bank_quality_v2_20261021/summary.json` and the updated
qualification report for the closure and next gate.

This version is an isolated experiment, not a replacement for earlier results.
The frozen text development serving path is dense retrieval → FP32 reranker →
observable dialogue closure → shared immutable materialization → one Reader.
Neither labels, question types, lexical rules, nor inferred factual relations
enter the planner. Dialogue provenance is not a claim of world-state causality.

## Fixed execution contract

- Python: `../../.venvs/flamr-qwen35/bin/python` (Python 3.11 environment).
- Reader: Qwen3.5-4B BF16; embedder: the cached Qwen3-Embedding-0.6B
  sentence-transformer (last-token pooling); reranker: the separate frozen
  Qwen3-Reranker-0.6B. The stale project embedding directory is not used.
- Evidence budget: 6000 tokens; output limit: 256 tokens; no truncation.
- Current candidate retrieval: dense-ranked event identities (at most 64), all
  visible windows for each event; 16 reranked window seeds. Earlier v1 artifacts
  used 64 individual windows and remain separately frozen.
- Search: at most 10000 nodes, with feasible incumbent and cardinality bound.
- Prerequisites follow the recorded child→parent direction, not undirected
  connected components. No future reply is added merely because it shares a parent.
- Duplicate closures count once. The current objective maximizes the sum of frozen
  reranker yes scores over unique closures (maximum seed score per duplicate),
  then minimizes union cost; frozen rank resolves exact ties. Exact integer units
  preserve the binary scores for search bounds. These are uncalibrated relevance
  scores, not answer-success probabilities. Earlier cardinality runs stay frozen.
- All over-budget closures remain explicitly rejected; no partial evidence tile.
- One chronological device stream, no cold sharding of a reuse experiment.
- Reranker prefix forks are checked against full FP32 scores on every request;
  an error or ranking disagreement substitutes the full scores.
- Reader prefix reuse is exact-token-prefix only, under a bounded state store.
  Cold requests serve the unsplit full-prefill greedy reference; checkpoint
  construction is separately charged and never supplies that answer. Every
  actual resumed candidate checks complete greedy tokens/EOS against an
  independent unsplit reference. Direct-reference requests are reported
  separately, not counted as equivalence passes.
- PIM routing charges copy, commands, fences and capacity for the current known
  primitive only. No assumed future query reuse or output-length oracle.

The PIM implementation is a functional NumPy backend plus an **uncalibrated
phase-cost model**. Actual answers are computed by the GPU Reader. Modeled PIM
cycles must never replace measured wall time or be claimed as device speedup.

The execution-driven route compiler is `scripts/compile_omni_pim_routes.py`.
It binds routes to actual parameter allocations and preserves every observed
event, dependency, and cache writeback barrier. Its output is a known-Linear
subgraph scenario only; opaque kernels remain unknown, and `end_to_end_speedup`
is required to remain `null` until a real Reader/PIM execution path is validated.
The fixed-cohort route closure and independent reconstruction audit are under
`artifacts/omni_pim_execution_route_dev3x16_v2_20261020/`. After declared shared
index/store reservations, 18 physical matrices are placed. Conservative modeled
known-Linear scenarios are 1.299887x warm and 0.984688x cold-first-step, not E2E
results. See `reports/OMNI_PIM_EXECUTION_ROUTE_CLOSURE_20261020_ZH.md` for the exact
optimization, retained unknowns, reproducibility commands and next mainline gates.
The subsequent opaque-boundary replay covers all 48 frozen requests with 2,304
inclusive callable completion markers and preserves 48/48 frozen outputs/EOS. Full
service baseline timing and diagnostic replay timing are reported separately in
`reports/OMNI_OPAQUE_BOUNDARY_AND_E2E_LEDGER_20261020_ZH.md`; measured and modeled
end-to-end speedups remain `null` until a matched complete-serving candidate exists.

## Omni raw-image closure status (2026-10-20)

The Omni adapter and label isolation are implemented in `scripts/omni_prepare.py`.
The development raw-image Reader smoke is `scripts/run_omni_raw_image_smoke.py`.
It now supports a label-free local dialogue-hyperedge evidence planner in
`scripts/omni_evidence_planner.py`: adjacent observable turns are selected as
indivisible text/image context units under explicit token and image budgets.
This is a local evidence heuristic, not factual causality or a global optimum.

The latest image-frontier cohort and remaining gates are documented in
`reports/OMNI_IMAGE_FRONTIER_CLOSURE_20261020_ZH.md`: 48/48 EOS, strict local
proxy 38/48 (79.17%), empirical server service p50/p95 30.54/43.03 seconds,
candidate/final image-reference recall 0.70/0.34286. This is development
progress, not significant improvement: tail latency increased and the
candidate-to-selection gap remains. The optional unique-image max-coverage
objective (`--image-novelty-weight`, default 0) has only a two-request verified
pilot; it did not change those requests' selected images or answers, and is
not part of this frozen 48-request result. Run in the Python 3.11 project
environment; the current visual-reuse additions are covered by the unit suite.

The preceding frozen cohort, audits and remaining gates are documented in
`reports/OMNI_DEV48_CLOSURE_20261020_ZH.md` (earlier smoke history remains in
`reports/OMNI_RAW_IMAGE_AND_DECODE_PIM_PROGRESS_20261020_ZH.md`). Three development
users completed 16 raw-image requests each, with 48/48 EOS under a uniformly
registered 512-token output ceiling. The same-family local judge returned
37/48 Yes, also 37/48 under strict Yes-and-EOS scoring. Empirical server service
p50/p95 are 31.55/38.65 seconds (nearest-rank); these are development proxies,
not official accuracy, population significance, or device results.

The separate Omni path uses dense session retrieval, bounded local hyperedge
selection and one raw-image Reader; it does not yet reuse Reader KV/recurrent
state or use the frozen text reranker. Its bounded embedding-state store reuses
immutable edge vectors in that frozen cohort. The registered decode-PIM trace gives a 1.31x
linear-only scenario; reserving the full declared embedding-state budget and
dense index in the same pool reduces it to about 1.30x. Both use uncalibrated
LPDDR5 assumptions with `end_to_end_speedup=null`, not measured PIM acceleration.
The whole Qwen3.5-4B model is not qualified for the current 4 GiB full-device
residency assumption. Larger user-disjoint multimodal evaluation, exact
multimodal state reuse, shared capacity accounting, and calibrated hardware
validation remain required before the main experiment can be called closed.

The opt-in exact visual-state reuse implementation and its mandatory gates are
in `scripts/omni_visual_feature_store.py` and
`reports/OMNI_VISUAL_STATE_PROTOCOL_20261020_ZH.md`. Cache entries represent
ordered whole-image batches and exact processor pixels/grid, checkpoint and
processor identities; no language KV/recurrent state is reused. The registered
16-question natural-stream validation completed under
`artifacts/omni_visual_exact_user0_16_v2_20261020/`: 16/16 independent token and
bitwise visual-feature equivalence, 1/16 natural ordered-batch hits, 13/16 local
judge successes, and 1.28731x shared-capacity linear-only PIM projection.
See `reports/OMNI_VISUAL_EXACT_REUSE_CLOSURE_20261020_ZH.md` and use
`closure_audit_v2.json`; the original `closure_audit.json` has a stale judge
producer hash after a duplicate evaluation incident documented in
`evaluation_execution_note.json`. No Reader output or serving source changed.
A first attempt is retained as a failed gate in
`artifacts/omni_visual_exact_user0_16_v1_20261020/failure_note.json`. Hit-path
timings versus a later uncached generation are diagnostic only, not causal
speedup or end-to-end PIM evidence. The enabled registered-cohort evaluator
requires the visual budget, independent verification, and shared PIM-capacity
reservation to match the pre-execution plan.

## Run

Use fresh output directories; scripts refuse to overwrite frozen artifacts.

```bash
PYTHON=../../.venvs/flamr-qwen35/bin/python
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" scripts/mobilemem_prepare.py \
  --input data/mobilemem/text/mobilemem_data.json \
  --output artifacts/mobilemem_causal_prepare_v1
GPUS=0 LIMIT=16 SAMPLE_COUNT=0 CROSS_QUERY_PREFIX=1 \
  RUN_ROOT=artifacts/causal_dev16_evidence_v1_20261019 \
  bash scripts/run_mobilemem_development.sh
"$PYTHON" scripts/merge_mobilemem_development.py \
  --run-root artifacts/causal_dev16_evidence_v1_20261019 \
  --prepared artifacts/mobilemem_causal_prepare_v1
```

Use `score_mobilemem.py`, `judge_mobilemem_local.py`,
`audit_mobilemem_development.py`, `diagnose_evidence_flow.py`, and
`snapshot_mobilemem_run.py` to close each run before changing serving sources.
Chinese whitespace TokenF1 is diagnostic, not task success. The local semantic
judge is a development proxy, not official accuracy or interactive mobile task
completion. Sealed551 remains unauthorized.

## Scope and stop conditions

See `reports/CLOSURE_PROTOCOL_20261018_ZH.md`. A passing execution audit only
certifies that the path was exercised correctly. It does not establish research
novelty, quality improvement, PIM speedup, or publication readiness.
