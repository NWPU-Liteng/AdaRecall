"""Diagnostic only: test atom-fragment token identity, never change serving plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from audit_omni_smoke_closure import digest
from omni_evidence_planner import render_atoms


def fragment_tokens(atoms, tokenizer):
    ordered = sorted((atom for atom in atoms if atom.get('content_type') == 'text'
                      and atom.get('content')), key=lambda atom: atom['ordinal'])
    plain, followed = {}, {}
    for atom in ordered:
        fragment = render_atoms([atom])
        plain[atom['atom_id']] = tokenizer.encode(fragment, add_special_tokens=False)
        followed[atom['atom_id']] = tokenizer.encode(fragment + '\n', add_special_tokens=False)
    return ordered, plain, followed


def compose_tokens(ordered, plain, followed):
    return [token for index, atom in enumerate(ordered)
            for token in (plain if index + 1 == len(ordered) else followed)[atom['atom_id']]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-user', type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists() or args.samples_per_user < 1:
        raise ValueError('fresh output and positive label-free sample count required')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    for name in ('omni_graph.json', 'split.json'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared source changed')
    split = json.loads((args.prepared / 'split.json').read_text())
    from transformers import AutoTokenizer
    model = Path('/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B')
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    graphs = json.loads((args.prepared / 'omni_graph.json').read_text())
    randomizer = random.Random(20261023)
    users = []
    for graph in graphs:
        if graph['person_id'] not in split['development_people']:
            continue
        atoms = [atom for atom in graph['atoms'] if atom.get('content_type') == 'text' and atom.get('content')]
        started = time.perf_counter()
        ordered, plain, followed = fragment_tokens(atoms, tokenizer)
        prepare_seconds = time.perf_counter() - started
        failures = []
        full_seconds = compose_seconds = 0.
        for index in range(args.samples_per_user):
            count = randomizer.randint(1, min(96, len(ordered)))
            sample = sorted(randomizer.sample(ordered, count), key=lambda atom: atom['ordinal'])
            started = time.perf_counter()
            actual = tokenizer.encode(render_atoms(sample), add_special_tokens=False)
            full_seconds += time.perf_counter() - started
            started = time.perf_counter()
            composed = compose_tokens(sample, plain, followed)
            compose_seconds += time.perf_counter() - started
            if composed != actual:
                failures.append({'sample': index, 'atom_ids': [atom['atom_id'] for atom in sample],
                                 'actual_tokens': len(actual), 'composed_tokens': len(composed)})
        users.append({'person_id': graph['person_id'], 'atom_fragments': len(ordered),
                      'samples': args.samples_per_user, 'failed_token_sequences': failures,
                      'all_history_fragment_preparation_seconds': prepare_seconds,
                      'full_tokenization_seconds': full_seconds, 'composed_seconds': compose_seconds})
        print(json.dumps({key: value for key, value in users[-1].items()
                          if key != 'failed_token_sequences'} | {'failed': len(failures)}), flush=True)
    result = {'complete': True, 'per_user': users,
              'all_tested_token_sequences_exact': all(not user['failed_token_sequences'] for user in users),
              'tokenizer_files_sha256': {path.name: digest(path) for path in model.iterdir()
                                         if path.name.startswith('tokenizer') or path.name == 'added_tokens.json'},
              'prepared_audit_sha256': digest(args.prepared / 'audit.json'),
              'script_sha256': digest(Path(__file__)), 'evaluation_label_files_read': [],
              'boundary': 'label-free random atom unions, not serving requests or a general BPE guarantee; preparation over all history is diagnostic, not an online cache or free work; no serving change authorized by this artifact alone'}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
