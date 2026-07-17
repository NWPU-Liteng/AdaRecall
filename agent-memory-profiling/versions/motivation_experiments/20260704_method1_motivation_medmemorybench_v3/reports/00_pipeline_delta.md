# Pipeline Delta for MedMemoryBench v3

| Original component | v3 component | Reason | Expected impact | Invalidates old-run comparison |
|---|---|---|---|---|
| LoCoMo/LongMemEval runtime manifest rows | MedMemoryBench schema-driven adapter | MedMemoryBench has different dialogue/query/evidence fields | Enables official healthcare memory workload | Yes, old absolute results are not directly comparable |
| Raw turn/session memory units from previous adapters | Chronological patient-doctor turn-pair memory item | Required primary granularity for v3 | More coherent retrieval unit than single turns; finer than session | Yes |
| `retrieval_top_r` and `pruner_top_b` naming | `frontier_budget` for reranker frontier plus ANN overfetch multiplier | Aligns with motivation question | Makes occupancy auditable | Naming change only |
| Adaptive/old phase controller | Fixed Ground -> Plan -> Reason -> Reflect controller FSM | MedMemoryBench source fields are not used as O1 execution-stage labels | Makes O1 auditable as controller-defined stage locality | Yes |
| Prior tile analysis from historical selected sets | Fresh M2 budget-64 top-5 selected IDs | Required by instruction | Avoids stale synthetic target sets | Yes |
| Optional heuristic reranker | Formal-only cross-encoder reranker | Formal v3 requires real reranker | Prevents embedding-similarity-only claims | No if old run also used same cross-encoder; otherwise yes |
| Existing QA normalization | Raw answer plus deterministic normalized answer | Preserve answer hygiene | Avoids reference-based rewriting | No |
| O1 controller JSON parser and prompt | Prompt avoids copyable schema literals; parser scans for the first valid JSON object with `retrieval_goal`; controller cap raised from 80 to 120 tokens | Pilot showed truncation and copied template fragments at 80 tokens | Raises O1 parse reliability before formal stage-locality analysis | Yes for O1 smoke/pilot traces created before 2026-07-05 12:39 local time |

No model replacement is intended. The default v3 configs point to:

- controller/answer model: `../../../dir`
- embedding model: `../../models/bge-base-en-v1.5`
- reranker: `../../models/cross-encoder-ms-marco-MiniLM-L6-v2`

## GPU selection note

The package instruction prefers `CUDA_VISIBLE_DEVICES` for physical GPU selection. In this server/sandbox combination, setting `CUDA_VISIBLE_DEVICES=<idx>` made `torch.cuda.is_available()` return false inside the managed command environment, while CUDA was available when the variable was unset. For the completed O1 smoke, pilot, and formal runs on 2026-07-05, the run command therefore left `CUDA_VISIBLE_DEVICES` unset and used `LOCAL_QWEN_CUDA_DEVICE=6`, which is consumed by the local Qwen, embedding, and reranker clients. The manifests record both fields:

- `cuda_visible_devices: null`
- `local_qwen_cuda_device: "6"`

M1 and M2 formal runs were completed earlier with `CUDA_VISIBLE_DEVICES=3`.
