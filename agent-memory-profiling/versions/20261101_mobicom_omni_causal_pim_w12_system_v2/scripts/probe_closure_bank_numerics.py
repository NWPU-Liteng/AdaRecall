"""Dataset-independent live-activation roundoff probe, never a latency benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.functional_decode import BankOwnedLinear


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument('--reader', type=Path, default=Path(
        '/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh diagnostic output required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'checkpoint_sha256': {path.name: digest(path) for path in sorted(args.reader.iterdir())
                    if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt')},
        'script_sha256': digest(Path(__file__)),
        'bank_source_sha256': digest(ROOT / 'pim_backend/functional_decode.py'),
        'input': 'Compute one plus one and respond briefly.', 'input_source': 'fixed synthetic; no benchmark data',
        'selection': 'largest executed language Linear for each distinct physical weight shape, top three by bytes',
        'boundary': 'functional GPU emulation only, no PIM performance, QA noninferiority or E2E claim'}
    args.output.write_text(json.dumps({'complete': False, 'provenance': provenance}, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(str(args.reader), local_files_only=True)
    reader = AutoModelForImageTextToText.from_pretrained(
        str(args.reader), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda:0').eval()
    executed = {}
    handles = []
    for name, module in reader.named_modules():
        if isinstance(module, torch.nn.Linear):
            def observe(current, arguments, output, identity=name):
                executed[identity] = current
            handles.append(module.register_forward_hook(observe))
    messages = [{'role': 'user', 'content': provenance['input']}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        initial = reader(**inputs, use_cache=True, logits_to_keep=1)
    for handle in handles:
        handle.remove()
    by_shape = {}
    for name, module in sorted(executed.items()):
        shape = tuple(module.weight.shape)
        by_shape.setdefault(shape, (name, module))
    targets = sorted(by_shape.values(), key=lambda item: (-item[1].weight.numel(), item[0]))[:3]
    rows = []
    handles = []

    def check(name):
        def capture(module, arguments, host):
            activation = arguments[0]
            if activation.numel() != module.in_features:
                raise ValueError('one decode row required')
            layout = BankOwnedLinear(module.weight)
            bank = layout(activation, module.bias)
            flat = activation.reshape(-1).double()
            reference = torch.cat([torch.mv(module.weight[begin:begin + 1024].double(), flat)
                                   for begin in range(0, module.out_features, 1024)])
            magnitude = torch.cat([torch.mv(module.weight[begin:begin + 1024].double().abs(), flat.abs())
                                   for begin in range(0, module.out_features, 1024)])
            if module.bias is not None:
                reference.add_(module.bias.double())
                magnitude.add_(module.bias.double().abs())
            eps = torch.finfo(torch.float32).eps / 2
            steps = module.in_features + (module.in_features + 255) // 256 + 2
            gamma = steps * eps / (1 - steps * eps)
            bound = gamma * magnitude + torch.finfo(torch.float32).tiny
            error = (bank.double() - reference).abs()
            rounded_reference = reference.bfloat16()
            setting = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            try:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                matched_host = torch.nn.functional.linear(activation, module.weight, module.bias)
            finally:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = setting
            rows.append({'module': name, 'inputs': module.in_features, 'outputs': module.out_features,
                'weight_bytes': module.weight.numel() * module.weight.element_size(),
                'bank_max_absolute_fp64_error': float(error.max()),
                'bank_forward_error_bound_pass': bool(torch.isfinite(bank).all() and (error <= bound).all()),
                'elements': module.out_features,
                'default_host_correctly_rounded_elements': int((host.reshape(-1) == rounded_reference).sum()),
                'matched_host_correctly_rounded_elements': int((matched_host.reshape(-1) == rounded_reference).sum()),
                'bank_correctly_rounded_elements': int((bank.bfloat16() == rounded_reference).sum()),
                'default_host_bank_bitwise_equal': bool(torch.equal(host.reshape(-1), bank.bfloat16())),
                'matched_host_bank_bitwise_equal': bool(torch.equal(matched_host.reshape(-1), bank.bfloat16())),
                'arithmetic_contract': 'BF16 operands, tile-256 FP32 partial accumulation, final BF16; FP64 diagnostic reference'})
        return capture

    for name, module in targets:
        handles.append(module.register_forward_hook(check(name)))
    token = initial.logits[:, -1, :].argmax(-1, keepdim=True)
    with torch.inference_mode():
        reader(input_ids=token, past_key_values=initial.past_key_values, use_cache=True, logits_to_keep=1)
    for handle in handles:
        handle.remove()
    complete = len(rows) == len(targets) and bool(rows)
    result = {'complete': complete, 'rows': rows, 'provenance': provenance,
        'roundoff_qualified': complete and all(row['bank_forward_error_bound_pass'] for row in rows),
        'replacement_quality_qualified': False, 'performance_qualified': False,
        'exact_host_equivalence': complete and all(row['default_host_bank_bitwise_equal'] for row in rows)}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
