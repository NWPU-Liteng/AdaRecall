# Method 1 v21 LACA Extension Data Audit

## Status

`not_eligible_for_v21_main_results`. This document is a preregistered
eligibility contract for a future LACA extension, not evidence that the
extension has passed.

## Why The Existing Trace Is Ineligible

The v20-r1 trace records items serialized into a prompt by a frozen retrieval
and reranking policy. It lacks an explicit agent memory-read tool call and has
an all-unseen session-heldout next-target vocabulary. It may be reported as a
payload construction audit but cannot be relabeled as actual access demand.

## Required Event Schema

Every event must contain `event_id`, `session_id`, `query_or_task_id`, a
strictly increasing causal `step_id` or timestamp, `memory_id`, `access_type`,
and a provenance pointer. The access types `seed_retrieved`,
`candidate_expanded`, `payload_serialized`, `reranker_selected`,
`agent_tool_read`, and `agent_consumed` must be distinct. Only an observed
`agent_tool_read` (or a validated agent-consumed event) can train the primary
LACA target. Gold evidence and future task success may never be encoded as an
online read.

## Required Data Eligibility Checks

Before OACH implementation, a candidate corpus must publish: (1) chronological
train/heldout split construction; (2) target recurrence and unseen-target
rate; (3) number and share of length-at-least-three chains; (4) event time-gap
and type completeness; (5) recurrence by session block; (6) an audit proving
that memory reads are not retrieval-policy output; and (7) a heldout comparison
where Markov/global-frequency/OACH all predict the same legal target space.
The preregistered headroom criterion is a positive paired CI over the strongest
frequency/Markov baseline together with a non-chance order-shuffle decrement.
No threshold may be changed after inspecting this result.

## Public Candidates And Constraints

WebLINX provides public real multi-turn web demonstrations and can evaluate
ordered *web artifact* action chains, but it is not an agent long-term-memory
corpus. tau-bench/tau2-bench provide stateful tool environments, not a released
corpus of recurrent memory reads. MemoryArena is the closest public
interdependent multi-session environment, but its current memory client only
returns a prompt wrapper. A future collection can add an explicit
`search_memory/read_memory` tool that logs returned and selected stable IDs;
the data must be described as **instrumented frozen-agent trajectories**, not
as independently observed human demand.

## Future Extension Protocol

1. Freeze at least one behavior policy and the environment version before
   collection; record model, prompt, tool schema, and random seed hashes.
2. Give the agent an explicit item-level memory tool and log every returned,
   selected, and read ID separately with step/time metadata.
3. Preserve persistent item IDs across chronological blocks and split by
   session/time before fitting OACH.
4. Run this eligibility audit before selecting model architecture or negative
   sampling details. If the audit fails, report it and keep FSA-DyHG semantic
   only.
5. Keep trajectory collection, LACA training, and deployment evaluation
   policies separate enough to expose retrieval-policy feedback loops.

This extension is intentionally not a prerequisite for completing v21's
semantic claims.
