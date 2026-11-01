"""Label-free, fixed-prefix arithmetic diagnosis; never a serving benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualify_omni_functional_pim import checked, make_inputs, verify_registration
from audit_omni_smoke_closure import digest


def compare(reference, candidate):
    import torch

    error = candidate.double().reshape(-1) - reference
    return {'elements': reference.numel(),
        'max_absolute_error': float(error.abs().max()),
        'relative_l2_error': float(torch.linalg.vector_norm(error) /
                                  torch.linalg.vector_norm(reference).clamp_min(1e-30)),
        'correctly_rounded_bf16_equal_elements': int((candidate.bfloat16().reshape(-1) ==
                                                     reference.bfloat16()).sum())}


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import zipfile
    from pim_backend.functional_decode import BankOwnedLinear
    from omni_execution_input_contract import execution_input_contract

    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'status': 'reserved_before_execution'}, handle)
    registration = json.loads(args.registration.read_text())
    verify_registration(registration)
    plan = checked(Path(registration['execution_plan']), registration['execution_plan_sha256'])
    routes = checked(Path(registration['routes']), registration['routes_sha256'])
    job = plan['jobs'][0]
    smoke = checked(Path(job['smoke']), job['smoke_sha256'])
    original = smoke['rows'][0]
    manifests_path = Path(plan['prepared']) / 'manifests.jsonl'
    if digest(manifests_path) != plan['manifests_sha256']:
        raise ValueError('serving manifests changed')
    manifests = {row['question_id']: row for row in
                 map(json.loads, manifests_path.read_text().splitlines())}
    checkpoint = Path(plan['reader'])
    for name, expected in {**plan['checkpoint_sha256'], **plan['processor_files_sha256']}.items():
        if digest(checkpoint / name) != expected:
            raise ValueError('Reader checkpoint or processor changed')
    if digest(Path(plan['archive'])) != plan['archive_sha256']:
        raise ValueError('image archive changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.image_processor.size['longest_edge'] = smoke['image_max_pixels']
    reader = AutoModelForImageTextToText.from_pretrained(str(checkpoint), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    aliases = {}
    for name, parameter in reader.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append('parameter:' + name)
    selected = []
    for name, module in reader.named_modules():
        if isinstance(module, torch.nn.Linear):
            identity = '||'.join(sorted(aliases[id(module.weight)]))
            if identity in routes['placement']:
                selected.append((name, module))
    targets = [next(item for item in selected if item[0].endswith('gate_proj')),
               next(item for item in selected if item[0].endswith('up_proj')),
               next(item for item in selected if item[0].endswith('lm_head'))]
    rows, handles = [], []
    state = {'observe': False}

    def hook(name):
        def capture(module, arguments, host):
            if not state['observe']:
                return
            activation = arguments[0]
            if activation.numel() != module.in_features:
                raise ValueError('single decode row required')
            weight = module.weight
            flat = activation.reshape(-1)
            reference = torch.cat([torch.mv(weight[start:start + 1024].double(), flat.double())
                for start in range(0, module.out_features, 1024)])
            if module.bias is not None:
                reference.add_(module.bias.double())
            layout = BankOwnedLinear(weight)
            bank = layout(activation, module.bias)
            dense_fp32 = torch.nn.functional.linear(activation.float(), weight.float(),
                None if module.bias is None else module.bias.float())
            settings = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            try:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                host_no_reduced = torch.nn.functional.linear(activation, weight, module.bias)
            finally:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = settings
            rows.append({'module': name, 'shape': list(weight.shape),
                'host_bf16': compare(reference, host),
                'bank_fp32': compare(reference, bank),
                'bank_bf16': compare(reference, bank.bfloat16()),
                'dense_fp32': compare(reference, dense_fp32),
                'host_reduced_precision_disabled': compare(reference, host_no_reduced),
                'host_equals_reduced_precision_disabled': bool(torch.equal(host, host_no_reduced)),
                'host_equals_bank_bf16': bool(torch.equal(host.reshape(-1), bank.bfloat16())),
                'bank_layout_bytes': layout.resident_bytes})
        return capture

    with zipfile.ZipFile(plan['archive']) as archive, torch.inference_mode():
        inputs = make_inputs(processor, original, manifests[original['question_id']], archive)
        contract = execution_input_contract(inputs, original)
        for name, module in targets:
            handles.append(module.register_forward_hook(hook(name)))
        try:
            output = reader(**inputs.to('cuda:0'), use_cache=True, logits_to_keep=1)
            prefix = []
            for invocation in range(2):
                token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                prefix.append(int(token.item()))
                state['observe'] = invocation == 1
                output = reader(input_ids=token, past_key_values=output.past_key_values,
                                use_cache=True, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
    if len(rows) != len(targets) or prefix != original['answer_token_ids'][:2]:
        raise ValueError('complete fixed targets and unmodified host prefix required')
    verify_registration(registration)
    result = {'complete': True, 'question_id': original['question_id'], 'rows': rows,
        'input_contract': contract, 'unchanged_host_prefix_tokens': prefix,
        'diagnostic_script_sha256': digest(Path(__file__)),
        'registration_sha256': digest(args.registration), 'evaluation_label_files_read': [],
        'end_to_end_speedup': None, 'actual_physical_pim_executed': False,
        'reference': 'chunked full FP64 dot products of original BF16 operands, not exact real arithmetic',
        'boundary': 'three predetermined complete matrices at second decode of first canonical request; '
                    'all outputs returned by original host; no quality or timing claim'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'complete': True, 'matrices': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
