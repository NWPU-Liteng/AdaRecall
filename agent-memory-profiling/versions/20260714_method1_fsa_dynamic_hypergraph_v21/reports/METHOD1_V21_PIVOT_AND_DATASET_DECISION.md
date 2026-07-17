# Method 1 v21 Pivot And Dataset Decision

## Decision

v21 is an independent Method 1 version named **FSA-QEHG** (Fuzzy Semantic
Association and Query-conditioned Evidence Hypergraph). It retains UOT-FSA, a
query-conditioned compositional evidence-set association, a hyperedge-first
budgeted selector, risk-controlled adaptive budgets, and content-causal
structural overlays. It does not include LACA/OACH in a main claim, a model
input, or a selector feature.

This is a scope correction, not a claim that the MedMemoryBench-derived corpus
is generally unsuitable. The corpus contains train-only evidence groups,
frozen embeddings, a fixed chronological/session-aware split, and enough
content and evidence structure for semantic association and retrieval
selection. It does not identify the proposed causal ordered-access mechanism.

## Reproducible Evidence For The Pivot

The frozen v20-r1 primary trace is a real local-agent prompt-payload trace:
1,355 successful invocations and 1,263 sequences of length at least three. It
contains no gold/evidence field and was collected without opening calibration
or formal test. Its primary payload sequence is nevertheless induced by the
frozen retrieval/reranking/context serialization policy, not by an explicit
agent `read(memory_id)` action after observing a previous memory result.

In session-heldout OACH evaluation every exact next memory target was unseen in
the training trace (`unseen_target_rate = 1.0`). Full-vocabulary Recall@1/5/10
was 0/0/0 and order-shuffle accuracy was 0.520586. This fails the original
order-aware headroom gate. A neural model cannot repair a target-space split in
which no evaluated target recurs, and treating serialization order as demand
would create a retrieval-policy feedback claim rather than an access-demand
claim.

The corresponding train-only UOT-FSA gate did pass. Across the five frozen
seeds, session-blocked OOF evidence-group co-location improved over matched
cosine kNN by +0.056396 to +0.069496, with every paired-bootstrap 95% lower
bound positive. These results are retained as a historical pilot observation;
v21 reruns its own training and does not import those artifacts or results as a
v21 finding.

## Dataset Suitability Audit

| Candidate | Long-lived content/state | Independently observed ordered reads | Cross-session recurring memory IDs | Fits FSA | Fits original LACA | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| Current MedMemoryBench-derived corpus | Yes | No; the available order is serialized payload order | No in heldout primary trace | Yes | No | v21 primary semantic corpus |
| LoCoMo | Multi-session dialogue histories | No memory-tool read log | Not an access-log protocol | Yes | No | External semantic retrieval validation candidate |
| LongMemEval / MemoryAgentBench | Incremental chunks and repeated queries | No independent memory-read event log in released benchmark API | Not established as access recurrence | Yes | No | External semantic and dynamic-content validation candidate |
| WebLINX | Real multi-turn web demonstrations | Yes for browser actions/turns | Not persistent long-term memory | No direct match | Only after changing claim to web-artifact chains | Do not use as Method 1 primary data |
| tau-bench / tau2-bench | Stateful tool environment | Agent actions can be logged during a run | No released long-term-memory access corpus | No direct match | Only after a separate collection protocol | Not a direct replacement |
| MemoryArena | Public interdependent multi-session environment | The supplied client returns a wrapped prompt, not item-level read IDs | Requires new eligibility audit | Potentially | Potentially after instrumentation | Controlled collection candidate, not a drop-in dataset |

Primary sources inspected: LoCoMo (`https://github.com/snap-research/locomo`),
LongMemEval (`https://github.com/xiaowu0162/LongMemEval`), MemoryAgentBench
(`https://github.com/HUST-AI-HYZ/MemoryAgentBench`), WebLINX
(`https://github.com/McGill-NLP/weblinx`), tau-bench
(`https://github.com/sierra-research/tau-bench`), and MemoryArena
(`https://github.com/ZexueHe/MemoryArena`). In particular, the inspected
MemoryArena `memory/client.py` calls `wrap_user_prompt(question)` and returns
only a prompt; its server endpoint returns no retrieved memory-item IDs. It is
therefore not currently an item-level LACA trace source.

## v21 Scientific Claim

The v21 claim is limited to: train-only learned, overlapping semantic
hyperedges organize fuzzy evidence groups more effectively than matched hard or
pairwise semantic structures; query-conditioned hierarchical evidence paths
can select bundles that cover labeled evidence aspects under a fixed
candidate/payload budget; and a dynamic semantic overlay may approach a
content-only full rebuild at lower update cost.

The set-of-sets relation has a precise name: `query -> labeled evidence aspect
-> alternative acceptable memory set` during training, and `query -> learned
latent aspect -> activated FSA hyperedge -> adaptive memory bundle` during
inference. It is a hierarchical association path, not a temporal chain. v21
must not claim that ordered access chains are useful, that FSA and LACA are
complementary, that a co-evidence group is observed co-access, or that a
payload serialization trace represents actual future access demand.

## Execution Sequence

1. Re-run UOT numerical stability and session-blocked nested OOF FSA selection
   under v21-only artifact names.
2. Audit and materialize the train-only `query -> evidence group -> eligible
   memory` incidence; implement the matched pairwise-clique,
   flattened-memory, and hyperedge-first evidence-bundle selectors.
3. Run five fixed seeds at B = 8/16/32 with a small-pool exact oracle.
4. Freeze architecture, then and only then use calibration for CRC/LTT.
5. Replay content arrival with Online UOT/BOSI semantic overlays and compare
   with append-only and full rebuild.
6. Keep LACA as a separately preregistered extension that first must pass the
   data-eligibility audit in `METHOD1_V21_LACA_EXTENSION_DATA_AUDIT.md`.

No calibration or formal-test path was read while making this decision.
