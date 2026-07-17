# Stage Controller Audit

Stage label source: `controller_fsm_v1`

## Trace Sample 1

- query_id: `session_10_eem_1`
- stage: `ground`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Current-state grounding evidence: Identify chronic metabolic disease from past medical history
- selected_memory_ids: `medmmb_en_p1_s7_x8,medmmb_en_p1_s1_x14,medmmb_en_p1_s1_x5,medmmb_en_p1_s1_x16,medmmb_en_p1_s2_x5`

## Trace Sample 2

- query_id: `session_10_eem_1`
- stage: `plan`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Longitudinal planning evidence: Identify chronic metabolic disease from past medical history
- selected_memory_ids: `medmmb_en_p1_s1_x14,medmmb_en_p1_s1_x16,medmmb_en_p1_s2_x2,medmmb_en_p1_s4_x2,medmmb_en_p1_s2_x5`

## Trace Sample 3

- query_id: `session_10_eem_1`
- stage: `reason`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Causal temporal reasoning evidence: Identify chronic metabolic disease from past medical history
- selected_memory_ids: `medmmb_en_p1_s7_x8,medmmb_en_p1_s1_x14,medmmb_en_p1_s1_x16,medmmb_en_p1_s7_x16,medmmb_en_p1_s2_x5`

## Trace Sample 4

- query_id: `session_10_eem_1`
- stage: `reflect`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Update contradiction safety-check evidence: Identify chronic metabolic disease from past medical history
- selected_memory_ids: `medmmb_en_p1_s7_x8,medmmb_en_p1_s1_x14,medmmb_en_p1_s1_x16,medmmb_en_p1_s2_x10,medmmb_en_p1_s8_x11`

## Trace Sample 5

- query_id: `session_10_ig_1`
- stage: `ground`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Current-state grounding evidence: Identify recent patient state and key constraints to determine if medication dose adjustment is indicated.
- selected_memory_ids: `medmmb_en_p1_s10_x6,medmmb_en_p1_s7_x6,medmmb_en_p1_s10_x0,medmmb_en_p1_s2_x12,medmmb_en_p1_s5_x6`

## Trace Sample 6

- query_id: `session_10_ig_1`
- stage: `plan`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Longitudinal planning evidence: Identify chronic conditions, long-term goals, prior episodes, medications, and relevant preferences to inform medication dose adjustment.
- selected_memory_ids: `medmmb_en_p1_s2_x12,medmmb_en_p1_s2_x15,medmmb_en_p1_s8_x15,medmmb_en_p1_s4_x8,medmmb_en_p1_s3_x9`

## Trace Sample 7

- query_id: `session_10_ig_1`
- stage: `reason`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Causal temporal reasoning evidence: Retrieve evidence that supports or refutes the current medical hypothesis regarding the need to increase the dose of blood sugar–lowering medication.
- selected_memory_ids: `medmmb_en_p1_s2_x11,medmmb_en_p1_s2_x5,medmmb_en_p1_s2_x2,medmmb_en_p1_s2_x14,medmmb_en_p1_s4_x8`

## Trace Sample 8

- query_id: `session_10_ig_1`
- stage: `reflect`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Update contradiction safety-check evidence: Check for recent updates, contradictions, traps, safety constraints, and stale facts related to the need to increase the dose of blood sugar–lowering medication.
- selected_memory_ids: `medmmb_en_p1_s4_x8,medmmb_en_p1_s1_x14,medmmb_en_p1_s8_x11,medmmb_en_p1_s2_x11,medmmb_en_p1_s5_x8`

## Trace Sample 9

- query_id: `session_20_mcd_1`
- stage: `ground`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Current-state grounding evidence: Identify recent symptoms and medication adherence to assess current clinical status and determine if medication efficacy is compromised.
- selected_memory_ids: `medmmb_en_p1_s2_x11,medmmb_en_p1_s2_x14,medmmb_en_p1_s19_x4,medmmb_en_p1_s11_x2,medmmb_en_p1_s2_x18`

## Trace Sample 10

- query_id: `session_20_mcd_1`
- stage: `plan`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Longitudinal planning evidence: Identify chronic conditions, long-term goals, prior episodes, medications, and relevant preferences to inform ongoing management strategies.
- selected_memory_ids: `medmmb_en_p1_s19_x1,medmmb_en_p1_s19_x4,medmmb_en_p1_s19_x16,medmmb_en_p1_s19_x10,medmmb_en_p1_s19_x13`

## Trace Sample 11

- query_id: `session_20_mcd_1`
- stage: `reason`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Causal temporal reasoning evidence: Retrieve evidence that supports or refutes the current medical or temporal hypothesis. Seek causal, temporal, comparative, and multi-hop supporting memories.
- selected_memory_ids: `medmmb_en_p1_s15_x14,medmmb_en_p1_s11_x8,medmmb_en_p1_s15_x20,medmmb_en_p1_s13_x16,medmmb_en_p1_s11_x17`

## Trace Sample 12

- query_id: `session_20_mcd_1`
- stage: `reflect`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Update contradiction safety-check evidence: Check for recent medication changes, allergies, or contradictions that may affect blood sugar control.
- selected_memory_ids: `medmmb_en_p1_s4_x8,medmmb_en_p1_s1_x14,medmmb_en_p1_s3_x8,medmmb_en_p1_s11_x2,medmmb_en_p1_s5_x11`

## Trace Sample 13

- query_id: `session_10_mq_1`
- stage: `ground`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Current-state grounding evidence: Identify recent symptoms and current clinical issue to determine appropriate antibiotic use.
- selected_memory_ids: `medmmb_en_p1_s4_x1,medmmb_en_p1_s4_x4,medmmb_en_p1_s5_x1,medmmb_en_p1_s9_x2,medmmb_en_p1_s5_x4`

## Trace Sample 14

- query_id: `session_10_mq_1`
- stage: `plan`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Longitudinal planning evidence: Identify chronic conditions, long-term goals, prior episodes, medications, and preferences to inform antibiotic selection.
- selected_memory_ids: `medmmb_en_p1_s9_x2,medmmb_en_p1_s1_x1,medmmb_en_p1_s1_x4,medmmb_en_p1_s4_x1,medmmb_en_p1_s2_x14`

## Trace Sample 15

- query_id: `session_10_mq_1`
- stage: `reason`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Causal temporal reasoning evidence: Retrieve evidence that supports or refutes the current medical hypothesis regarding the appropriateness of antibiotic use for a sore throat.
- selected_memory_ids: `medmmb_en_p1_s4_x4,medmmb_en_p1_s4_x1,medmmb_en_p1_s5_x1,medmmb_en_p1_s4_x10,medmmb_en_p1_s4_x13`

## Trace Sample 16

- query_id: `session_10_mq_1`
- stage: `reflect`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Update contradiction safety-check evidence: Check for recent updates, contradictions, and safety constraints related to antibiotic use for sore throat.
- selected_memory_ids: `medmmb_en_p1_s4_x1,medmmb_en_p1_s4_x4,medmmb_en_p1_s1_x1,medmmb_en_p1_s9_x2,medmmb_en_p1_s4_x13`

## Trace Sample 17

- query_id: `session_10_sua_1`
- stage: `ground`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Current-state grounding evidence: Determine if the patient is currently taking empagliflozin.
- selected_memory_ids: `medmmb_en_p1_s6_x2,medmmb_en_p1_s6_x5,medmmb_en_p1_s7_x0,medmmb_en_p1_s3_x5,medmmb_en_p1_s6_x14`

## Trace Sample 18

- query_id: `session_10_sua_1`
- stage: `plan`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Longitudinal planning evidence: Determine if the patient is currently taking empagliflozin.
- selected_memory_ids: `medmmb_en_p1_s6_x2,medmmb_en_p1_s6_x5,medmmb_en_p1_s6_x14,medmmb_en_p1_s3_x2,medmmb_en_p1_s3_x5`

## Trace Sample 19

- query_id: `session_10_sua_1`
- stage: `reason`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Causal temporal reasoning evidence: Determine if the patient is currently taking empagliflozin.
- selected_memory_ids: `medmmb_en_p1_s6_x2,medmmb_en_p1_s6_x5,medmmb_en_p1_s3_x2,medmmb_en_p1_s3_x5,medmmb_en_p1_s2_x5`

## Trace Sample 20

- query_id: `session_10_sua_1`
- stage: `reflect`
- prompt includes retrieved memory before intent: `false`
- retrieval_goal: Update contradiction safety-check evidence: Determine if the patient is currently taking empagliflozin.
- selected_memory_ids: `medmmb_en_p1_s6_x2,medmmb_en_p1_s6_x5,medmmb_en_p1_s7_x0,medmmb_en_p1_s6_x14,medmmb_en_p1_s4_x8`

## Validity Checks

- Stage is assigned by fixed FSM before retrieval.
- Every stage uses the same retrieval budget.
- Stage prompts contain query text/type/session and prior controller summaries only.
- Stage prompts do not include gold answers, source key points, or retrieved memory text before intent generation.
