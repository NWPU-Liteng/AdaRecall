from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_service_pair import (audit_pair, embedding_namespace_equivalence,
                                    function_source_identity, row_differences)
from test_omni_latency_accounting import serving_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pair_fixture(root):
    source = b'frozen serving source'
    sources = {'scripts/frozen.py': hashlib.sha256(source).hexdigest()}
    reference_root, candidate_root = root / 'reference', root / 'candidate'
    plan_fields = {field: 'frozen' for field in ('prepared_audit_sha256', 'serving_manifest_sha256',
        'image_archive_sha256', 'max_new_tokens', 'image_edges_per_session', 'image_novelty_weight')}
    extra = {field: 'frozen' for field in ('reader_config_sha256', 'embedder_config_sha256',
        'system_prompt_sha256', 'candidate_sessions', 'max_images', 'session_text_tokens',
        'total_text_tokens', 'image_max_pixels', 'max_new_tokens', 'archive_sha256',
        'prepared_audit_sha256')}
    frozen_run = serving_run()
    frozen_run.update(extra)
    for row in frozen_run['rows']:
        row.update({'input_tokens': 100, 'visual_patches': 20, 'answer_sha256': 'answer'})
    frozen_path = root / 'frozen.json'
    write(frozen_path, frozen_run)
    original = {'person_id': '0', 'question_ids': [row['question_id'] for row in frozen_run['rows']],
                'gpu': '3', 'output': str(frozen_path)}
    frozen_plan_path = root / 'frozen_plan.json'
    write(frozen_plan_path, {**plan_fields, 'jobs': [original]})
    for run_root, scale in ((reference_root, 1.), (candidate_root, .5)):
        output = run_root / 'user_0.json'
        run = deepcopy(frozen_run)
        run['source_files_sha256'] = sources
        run['embedding_state_reuse'] = {'hits': 1}
        for row in run['rows']:
            for field in ('service_seconds', 'retrieval_seconds', 'prefill_seconds', 'decode_seconds'):
                row[field] *= scale
        write(output, run)
        snapshot = run_root / 'sources'
        (snapshot / 'scripts').mkdir(parents=True)
        (snapshot / 'scripts/frozen.py').write_bytes(source)
        write(output.with_suffix('.registration.json'), {'source_files_sha256': sources,
            'source_snapshot_directory': str(snapshot), 'question_ids': original['question_ids']})
        write(run_root / 'cohort_plan.json', {**plan_fields, 'source_files_sha256': sources,
            'jobs': [{**original, 'output': str(output)}]})
        write(run_root / 'worker_status.json', {'sources_unchanged': True, 'all_workers_exited_zero': True})
    return reference_root, candidate_root, frozen_plan_path


def encoder_fixture(root, suffix, *, function='def token_vector(value):\n    return value\n', schema='legacy_whole_runner'):
    snapshot = root / suffix
    directory = snapshot / 'scripts'
    directory.mkdir(parents=True)
    source = function + f'\nOTHER_SETTING = {suffix!r}\n'
    names = {'scripts/run_omni_raw_image_smoke.py': source,
             'scripts/omni_embedding_producer.py': 'producer source'}
    for name, content in names.items():
        (snapshot / name).write_text(content)
    hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in names.items()}
    source_digest = (hashes['scripts/run_omni_raw_image_smoke.py'] if schema == 'legacy_whole_runner'
                     else hashlib.sha256(function.encode()).hexdigest())
    files = {'model': 'weights', 'tokenizer': 'tokenizer'}
    payload = {'files': files, 'dtype': 'torch.bfloat16', 'attention': 'sdpa',
               'pooling': 'last_nonpad_token_then_fp32_l2_normalize',
               'token_vector_source_sha256': source_digest}
    run = {'source_files_sha256': hashes, 'embedder_identity_files_sha256': files,
           'runtime_environment': {'runtime': 'fixed'}, 'embedding_producer_policy': 'canonical32',
           'embedding_execution_identity': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
    return run, {'source_snapshot_directory': str(snapshot)}


class ServicePairTests(unittest.TestCase):
    def test_known_namespaces_require_identical_encoder_and_model_contract(self):
        with tempfile.TemporaryDirectory() as name:
            first, first_registration = encoder_fixture(Path(name), 'first')
            second, second_registration = encoder_fixture(Path(name), 'second', schema='encoder_function')
            proof = embedding_namespace_equivalence(first, second, first_registration, second_registration)
            self.assertTrue(proof['equivalent'])
            self.assertFalse(proof['namespace_equal'])
            self.assertEqual(proof['schemas'], ['legacy_whole_runner', 'encoder_function'])
            second['embedding_execution_identity'] = 'unknown'
            with self.assertRaises(ValueError):
                embedding_namespace_equivalence(first, second, first_registration, second_registration)

    def test_known_namespace_does_not_allow_changed_encoder_or_weights(self):
        with tempfile.TemporaryDirectory() as name:
            first, first_registration = encoder_fixture(Path(name), 'first')
            second, second_registration = encoder_fixture(
                Path(name), 'second', function='def token_vector(value):\n    return value + 1\n')
            with self.assertRaises(ValueError):
                embedding_namespace_equivalence(first, second, first_registration, second_registration)
            second['embedder_identity_files_sha256']['model'] = 'other'
            with self.assertRaises(ValueError):
                embedding_namespace_equivalence(first, second, first_registration, second_registration)

    def test_embedding_function_identity_ignores_unrelated_runner_edits(self):
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / 'run.py'
            source.write_text('def token_vector(x):\n    return x\n\ndef other():\n    return 1\n')
            snapshot = Path(name) / 'snapshot'
            (snapshot / 'scripts').mkdir(parents=True)
            target = snapshot / 'scripts/run_omni_raw_image_smoke.py'
            target.write_text(source.read_text())
            first = {'source_files_sha256': {'scripts/run_omni_raw_image_smoke.py': 'first'}}
            registration = {'source_snapshot_directory': str(snapshot)}
            identity = function_source_identity(first, registration, 'token_vector')
            target.write_text('def token_vector(x):\n    return x\n\ndef other():\n    return 2\n')
            self.assertEqual(identity, function_source_identity(first, registration, 'token_vector'))
            target.write_text('def token_vector(x):\n    return x + 1\n')
            self.assertNotEqual(identity, function_source_identity(first, registration, 'token_vector'))

    def test_complete_independent_pair_reports_whole_cohort_ratio(self):
        with tempfile.TemporaryDirectory() as name:
            result = audit_pair(*pair_fixture(Path(name)))
        self.assertEqual(result['requests'], 4)
        self.assertEqual(result['execution_integrity'], 'pass')
        self.assertEqual(result['measured_server_end_to_end_speedup'], 2.)

    def test_failure_is_retained_without_success_subset_speedup(self):
        with tempfile.TemporaryDirectory() as name:
            roots = pair_fixture(Path(name))
            output = roots[1] / 'user_0.json'
            run = json.loads(output.read_text())
            run['rows'][0]['answer_token_ids'] = [9]
            write(output, run)
            result = audit_pair(*roots)
        self.assertEqual(result['requests'], 4)
        self.assertEqual(result['execution_integrity'], 'failed_pair_equivalence_gate')
        self.assertIsNone(result['measured_server_end_to_end_speedup'])
        self.assertEqual(len(result['pair_mismatches']), 1)

    def test_changed_source_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            roots = pair_fixture(Path(name))
            (roots[1] / 'sources/scripts/frozen.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                audit_pair(*roots)

    def test_diagnostics_not_operands_may_change(self):
        before = {'question_id': 'q', 'question_sha256': 'query', 'images': ['image'],
            'answer_token_ids': [2, 3], 'stop_reason': 'eos', 'input_tokens': 100,
            'visual_patches': 20, 'answer_sha256': 'answer',
            'hypergraph_evidence_plan': {'text': 'evidence', 'accepted_atom_ids': ['atom'],
                'selected_hyperedges': [{'edge_id': 'edge', 'atom_ids': ['atom'], 'marginal_gain': .3}],
                'embedding_store': {'hits': 1}}}
        after = deepcopy(before)
        after['hypergraph_evidence_plan']['selected_hyperedges'][0]['marginal_gain'] = .4
        after['hypergraph_evidence_plan']['embedding_store'] = {'hits': 2}
        self.assertEqual(row_differences(before, after), [])
        after['hypergraph_evidence_plan']['text'] = 'changed evidence'
        self.assertEqual(row_differences(before, after), ['semantic_evidence_plan'])
        after['answer_token_ids'] = [9]
        self.assertIn('answer_token_ids', row_differences(before, after))

    def test_processor_input_hash_changes_fail(self):
        before = {'question_id': 'q', 'question_sha256': 'query', 'images': [],
            'answer_token_ids': [2], 'stop_reason': 'eos', 'input_tokens': 1,
            'visual_patches': 1, 'answer_sha256': 'answer', 'hypergraph_evidence_plan': None,
            'processor_inputs_sha256': 'pixels/grid'}
        after = {**before, 'processor_inputs_sha256': 'different'}
        self.assertIn('processor_inputs_sha256', row_differences(before, after))

    def test_producer_identity_change_is_not_an_output_mismatch_and_still_fails(self):
        before = {'question_id': 'q', 'question_sha256': 'query', 'images': [],
            'answer_token_ids': [2], 'stop_reason': 'eos', 'input_tokens': 1,
            'visual_patches': 1, 'answer_sha256': 'answer', 'hypergraph_evidence_plan': None,
            'embedding_producers': [{'producer_sha256': 'old', 'edge_ids': ['edge'],
                                     'output_sha256': 'same'}]}
        after = deepcopy(before)
        after['embedding_producers'][0]['producer_sha256'] = 'new'
        self.assertEqual(row_differences(before, after), ['embedding_producer_identity'])
        after['embedding_producers'][0]['output_sha256'] = 'changed'
        self.assertEqual(row_differences(before, after),
                         ['embedding_producer_outputs', 'embedding_producer_identity'])


if __name__ == '__main__':
    unittest.main()
