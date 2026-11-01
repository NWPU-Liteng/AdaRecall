"""Archive only sources matching the recorded execution hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text())
    output = args.result.parent / 'source_snapshot'
    if output.exists():
        raise ValueError('fresh snapshot directory required')
    sources = {}
    for name, expected in payload['provenance']['code_sha256'].items():
        directory = 'pim_backend' if name in {'lpddr_bank_pim.py', 'agent_memory_pim.py'} else 'scripts'
        path = ROOT / directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'cannot archive modified execution source: {name}')
        sources[path.relative_to(ROOT)] = path
    profile = ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json'
    if hashlib.sha256(profile.read_bytes()).hexdigest() != payload['provenance']['profile_sha256']:
        raise ValueError('profile changed')
    sources[profile.relative_to(ROOT)] = profile
    for name in ('scripts/score_mobilemem.py', 'scripts/judge_mobilemem_local.py',
                 'scripts/audit_mobilemem_development.py', 'scripts/merge_mobilemem_development.py',
                 'scripts/system_audit.py', 'scripts/freeze_mobilemem_development.py',
                 'scripts/mobilemem_prepare.py', 'scripts/trajectory_hypergraph.py',
                 'scripts/state_reuse.py', 'scripts/run_mobilemem_development.sh',
                 'scripts/reader_single_pass.py', 'scripts/residual_profile.py',
                 'scripts/diagnose_evidence_flow.py',
                 'scripts/diagnose_dense_contract.py',
                 'scripts/probe_reader_execution_contract.py',
                 'scripts/snapshot_mobilemem_run.py',
                 'scripts/audit_reranker_state_reuse.py',
                 'scripts/summarize_materialization.py',
                 'scripts/model_identity.py',
                 'README.md',
                 'reports/CLOSURE_PROTOCOL_20261018_ZH.md',
                 'reports/EVIDENCE_OBJECTIVE_REVISION_20261019_ZH.md',
                 'reports/READER_REFERENCE_EXECUTION_REVISION_20261019_ZH.md',
                 'reports/ONLINE_SHARED_INT8_CONTRACT_20261019_ZH.md',
                 'pim_backend/__init__.py'):
        sources[Path(name)] = ROOT / name
    for path in sorted((ROOT / 'tests').glob('test_*.py')):
        sources[path.relative_to(ROOT)] = path
    manifest = {}
    for name, source in sources.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[str(name)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'snapshot': str(output), 'files': len(manifest)}))


if __name__ == '__main__':
    main()
