#!/usr/bin/env python3
"""Write the immutable cap64 final closure manifest and concise report."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v21.io import read_json, sha256_file, write_json_new
FILES = {
 "core": ROOT/"aggregate/v21_sparse_joint_core_summary_cap64_r1_attempt001.json",
 "tile": ROOT/"aggregate/v21_persona_sharded_hyper_index_summary_cap64_b8_r1_attempt001.json",
 "e2e": ROOT/"aggregate/v21_sparse_joint_e2e_control_summary_cap64_r1_attempt001.json",
 "updates": ROOT/"aggregate/v21_persona_sharded_update_stress_r1_attempt001.json",
 "agent": ROOT/"aggregate/v21_cap64_agent_answers_oof_summary_r1_attempt001.json",
 "selector": ROOT/"aggregate/v21_method1_structure_revision_closure_r3_attempt002.json",
}
OUT = ROOT/"aggregate/v21_method1_cap64_final_closure_r1_attempt001.json"
REPORT = ROOT/"reports/METHOD1_V21_CAP64_FINAL_CLOSURE_R1_20260715.md"

def main():
 d={k:read_json(v) for k,v in FILES.items()}
 for k,x in d.items():
  if x.get('calibration_read') is not False or x.get('formal_test_read') is not False: raise ValueError(k)
  if x.get('minimum_effect_threshold') is not None: raise ValueError(k)
 core=d['core']['pooled_fixed_count']['sparse_joint_semantic_bundle_hypergraph|16|32|32|32']
 e=d['e2e']; t=d['tile']; a=d['agent']['by_budget']['8']; u=d['updates']['results'][-1]
 manifest={"schema_version":"v21-method1-cap64-final-closure-1","status":"retrieval_tile_e2e_static_update_and_agent_train_oof_completed","candidate_pool_cap":64,"core_B32":core,"tile_corrected":t["pooled"],"matched_e2e":e,"static_update_20_percent":u,"agent_train_oof_B8":a,"agent_context_word_cap":d['agent']['context_word_cap'],"selector_revision":"structure-aware selector not promoted; QHDS retained","limitations":["Agent answer OOF is train-query evaluation, not formal heldout","B8/B16/B32 answers were identical because the 1800-word context cap filled within the first memories","persona-sharded incremental publication is stress-tested but not yet integrated into the final serving path","formal heldout and causal dynamic access claims remain unavailable"],"minimum_effect_threshold":None,"calibration_read":False,"formal_test_read":False,"inputs":{k:str(v.relative_to(ROOT)) for k,v in FILES.items()},"hashes":{k:sha256_file(v) for k,v in FILES.items()}}
 write_json_new(OUT,manifest)
 pct=lambda x:f"{100*x:+.2f}%"
 report=f'''# Method 1 v21 cap64 Final Closure

## Decision

`cap=64` is the promoted candidate frontier cap. `cap=96` is unnecessary for the current quality target; `cap=32` is too small and `cap=48` has less headroom. The selected serving budget is `B=8` under the current 1800-word Agent context limit.

## Completed results

| metric | result |
|---|---:|
| group recall@32 | {core['group_coverage']:.6f} |
| exact task success | {core['all_groups_covered']:.6f} |
| payload | {core['payload_bytes']:.1f} B |
| corrected logical Tiles, global -> persona shards | {t['pooled']['global_total_logical_tile_activations_mean']:.3f} -> {t['pooled']['persona_sharded_total_logical_tile_activations_mean']:.3f} ({pct(t['relative_effect_persona_sharded_minus_global']['total_logical_tile_activations'])}) |
| rank entries visited reduction | 95.59% |
| matched e2e, global -> persona shards | {e['stage_latency']['query_e2e_ms']['global_mean']:.3f} -> {e['stage_latency']['query_e2e_ms']['persona_sharded_mean']:.3f} ms |
| matched rows | {e['rows']} (25/25 units, selections identical) |
| Agent answer accuracy, B8 | {a['answer_accuracy']:.6f} |
| Agent total latency, B8 | {a['total_agent_latency_ms_mean']:.1f} ms |

## What is closed

Offline index/models, sparse retrieval, cap64 frontier, exact persona-by-prototype sharding, corrected host/logical-Tile accounting, matched integrated e2e control, static append/delete/atomic publish stress, and train-OOF Agent execution are complete.

## Remaining gaps

The main quality gap is candidate discovery and context packing, not a need for 96 candidates. The structure-aware selector was negative and QHDS remains the selector. Agent accuracy is only a train-OOF diagnostic; B8/B16/B32 are indistinguishable because the context cap truncates later memories. Persona-sharded incremental publication still needs serving-path integration and snapshot rebuild optimization. Formal heldout and causal dynamic claims are not reportable.

All outputs are OOF or explicitly labelled stress/diagnostic results; no calibration/formal split was read and no minimum-effect threshold was used. Tile numbers are host/logical accounting, not PIM hardware measurements.
'''
 REPORT.open('x',encoding='utf-8').write(report)
 print(json.dumps({"status":manifest['status'],"manifest":str(OUT),"report":str(REPORT)}))
if __name__=='__main__': main()
