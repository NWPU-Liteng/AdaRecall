# Method 1 v21 QHDS Ablation Interface Audit

The `v21_qhds_oof_r1_20260714` primary QHDS, flat-cosine, QESA-association, and without-UOT fixed-budget selections are valid for their declared inference inputs. The run completed 25/25 OOF units and remains immutable.

The first `qhds_without_latent_aspects` result is not a clean latent-aspect ablation. Its concave coverage term used the query-independent prototype prior, but its learned relevance features retained the QESA-conditioned incidence-association value. It therefore removes QESA from one utility path but not from the complete selector input.

This does not invalidate the primary QHDS versus flat comparison. It does prevent attributing the QHDS effect to latent aspects from this ablation. The existing row is retained and relabeled conceptually as `QHDS with prior coverage weights but QESA-conditioned relevance`.

A corrected evaluation revision must use the same frozen QHDS model and heldout queries, replace both the coverage distribution and incidence-association relevance feature with their query-independent-prior counterparts, and write a new run/revision. It must not retrain the model or overwrite the original metrics. Gate C remains unrun until the corrected ablation and matched pairwise factor-graph experiments are complete.

Calibration and formal test remain locked and unread.
