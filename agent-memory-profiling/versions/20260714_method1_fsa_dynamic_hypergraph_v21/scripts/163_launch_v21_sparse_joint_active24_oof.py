#!/usr/bin/env python3
"""Launch active24 OOF units across available GPUs."""
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='v21_sparse_joint_cap64_active24_oof_r1.json'); p.add_argument('--run-id',required=True); p.add_argument('--attempt',default='attempt_001'); p.add_argument('--devices',default='cuda:0,cuda:2,cuda:3,cuda:6'); a=p.parse_args()
 cfg=json.loads((ROOT/'configs'/a.config).read_text()); devices=[x for x in a.devices.split(',') if x]; jobs=[(int(s),f) for s in cfg['fixed_seeds'] for f in range(cfg['outer_folds'])]; root=ROOT/'artifacts/v21_sparse_joint'/a.run_id; (root/'launcher_logs').mkdir(parents=True,exist_ok=True)
 def one(i,s,f):
  unit=root/f'seed_{s}'/f'fold_{f}'/'attempts'/a.attempt; manifest=unit/'v21_sparse_joint_unit_manifest.json'
  if manifest.exists(): raise FileExistsError(str(manifest))
  cmd=[sys.executable,'-u',str(ROOT/'scripts/162_run_v21_sparse_joint_active24_oof.py'),'--config',a.config,'--run-id',a.run_id,'--seed',str(s),'--fold',str(f),'--attempt',a.attempt,'--device',devices[i%len(devices)]]
  r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True); stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); log=root/'launcher_logs'/f'unit_seed_{s}_fold_{f}_{stamp}.log'; log.write_text(r.stdout+'\n'+r.stderr)
  if r.returncode: raise RuntimeError(f'failed {s}/{f}: {log}')
  return {'seed':s,'fold':f,'status':'completed','log':str(log.relative_to(ROOT))}
 results=[]; failures=[]
 with ThreadPoolExecutor(max_workers=len(devices)) as ex:
  fs={ex.submit(one,i,s,f):(s,f) for i,(s,f) in enumerate(jobs)}
  for future in as_completed(fs):
   s,f=fs[future]
   try: row=future.result(); results.append(row)
   except Exception as e: row={'seed':s,'fold':f,'status':'failed','error':str(e)}; failures.append(row)
   print(json.dumps(row,sort_keys=True),flush=True)
 summary={'status':'completed' if not failures else 'failed','run_id':a.run_id,'attempt':a.attempt,'results':results,'failures':failures,'calibration_read':False,'formal_test_read':False}; (root/'launcher_logs'/f'launcher_summary_{a.attempt}.json').write_text(json.dumps(summary,indent=2))
 if failures: raise RuntimeError(f'{len(failures)} failed')
 print(json.dumps({'status':'completed','units':len(results)}))
if __name__=='__main__': main()
