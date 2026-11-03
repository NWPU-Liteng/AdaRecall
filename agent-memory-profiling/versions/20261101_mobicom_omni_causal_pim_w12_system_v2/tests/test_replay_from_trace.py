import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'device'))

import replay_from_trace


def trace_payload(person_id, question_id, workloads, *, image_count=0):
    stages = {stage: dict(values) for stage, values in workloads.items()}
    return {
        'schema': replay_from_trace.TRACE_SCHEMA,
        'person_id': person_id,
        'question_id': question_id,
        'question_sha256': hashlib.sha256(question_id.encode()).hexdigest(),
        'image_count': image_count,
        'stages': stages,
    }


def write_trace(root, person_id, question_id, workloads, *, image_count=0):
    path = Path(root) / person_id / f'{question_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace_payload(
        person_id, question_id, workloads, image_count=image_count)))
    return path


class ReplayManifestTests(unittest.TestCase):
    def test_manifest_extracts_reader_workloads_and_combines_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory) / 'trace'
            write_trace(trace_dir, '1', '1_q_1', {
                'contract_parse': {
                    'prefill_tokens': 10, 'decode_tokens': 2, 'n_images': 0},
                'witness_construct': {
                    'prefill_tokens': 30, 'decode_tokens': 4, 'n_images': 0},
                'answer_prefill': {
                    'prefill_tokens': 20, 'decode_tokens': 0, 'n_images': 0},
                'answer_decode': {
                    'prefill_tokens': 20, 'decode_tokens': 5, 'n_images': 1},
            }, image_count=1)
            write_trace(trace_dir, '0', '0_q_1', {
                'contract_parse': {
                    'prefill_tokens': 10, 'decode_tokens': 2, 'n_images': 0},
                'answer_prefill': {
                    'prefill_tokens': 7, 'decode_tokens': 0, 'n_images': 0},
                'answer_decode': {
                    'prefill_tokens': 7, 'decode_tokens': 2, 'n_images': 0},
            })

            manifest = replay_from_trace.build_manifest(trace_dir)

        self.assertEqual(manifest['schema'], 'device-replay-manifest-v1')
        self.assertEqual(manifest['trace_schema'], 'omni-serving-workload-trace-v1')
        self.assertEqual(manifest['trace_count'], 2)
        self.assertEqual(manifest['stage_entry_count'], 7)
        self.assertEqual(manifest['unique_combination_count'], 6)
        self.assertEqual(manifest['image_combination_count'], 1)
        self.assertEqual(manifest['stage_workloads'][0], {
            'person_id': '0',
            'question_id': '0_q_1',
            'question_sha256': hashlib.sha256(b'0_q_1').hexdigest(),
            'stage': 'contract_parse',
            'prefill_tokens': 10,
            'decode_tokens': 2,
            'n_images': 0,
        })
        duplicate = next(value for value in manifest['combinations']
                         if (value['prefill_tokens'], value['decode_tokens'],
                             value['n_images']) == (10, 2, 0))
        self.assertEqual(duplicate['occurrences'], 2)
        self.assertEqual(duplicate['sources'], [
            {'person_id': '0', 'question_id': '0_q_1', 'stage': 'contract_parse'},
            {'person_id': '1', 'question_id': '1_q_1', 'stage': 'contract_parse'},
        ])
        self.assertEqual(set(manifest['trace_files_sha256']), {
            '0/0_q_1.json', '1/1_q_1.json'})
        self.assertEqual(len(manifest['script_sha256']), 64)
        self.assertEqual(manifest['measurements'], [])

    def test_reader_stage_requires_all_workload_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory) / 'trace'
            write_trace(trace_dir, '0', '0_q_1', {
                'contract_parse': {'prefill_tokens': 10, 'decode_tokens': 2},
            })
            with self.assertRaisesRegex(ValueError, 'missing workload fields'):
                replay_from_trace.build_manifest(trace_dir)

    def test_non_reader_stage_rejects_workload_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory) / 'trace'
            write_trace(trace_dir, '0', '0_q_1', {
                'query_embed': {
                    'prefill_tokens': 10, 'decode_tokens': 0, 'n_images': 0},
            })
            with self.assertRaisesRegex(ValueError, 'non-Reader stage'):
                replay_from_trace.build_manifest(trace_dir)

    def test_bad_schema_duplicate_identity_and_empty_trace_dir_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory) / 'trace'
            trace_dir.mkdir()
            with self.assertRaisesRegex(ValueError, 'no workload traces'):
                replay_from_trace.build_manifest(trace_dir)

            path = write_trace(trace_dir, '0', '0_q_1', {
                'contract_parse': {
                    'prefill_tokens': 10, 'decode_tokens': 2, 'n_images': 0},
            })
            payload = json.loads(path.read_text())
            payload['schema'] = 'wrong-schema'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'unsupported workload trace schema'):
                replay_from_trace.build_manifest(trace_dir)

        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory) / 'trace'
            workload = {'contract_parse': {
                'prefill_tokens': 10, 'decode_tokens': 2, 'n_images': 0}}
            path = write_trace(trace_dir, '0', '0_q_1', workload)
            (path.parent / 'duplicate.json').write_text(path.read_text())
            with self.assertRaisesRegex(ValueError, 'duplicate trace identity'):
                replay_from_trace.build_manifest(trace_dir)

    def test_cli_refuses_to_overwrite_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'manifest.json'
            output.write_text('{"existing": true}\n')
            argv = ['replay_from_trace.py', str(Path(directory) / 'trace'),
                    '--output', str(output)]
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(
                    ValueError, 'refusing to overwrite'):
                replay_from_trace.main()


class AndroidBenchScriptTests(unittest.TestCase):
    def _manifest(self, directory):
        trace_dir = Path(directory) / 'trace'
        write_trace(trace_dir, '0', '0_q_1', {
            'contract_parse': {
                'prefill_tokens': 5, 'decode_tokens': 0, 'n_images': 0},
            'witness_construct': {
                'prefill_tokens': 10, 'decode_tokens': 4, 'n_images': 2},
            'answer_decode': {
                'prefill_tokens': 5, 'decode_tokens': 3, 'n_images': 0},
        }, image_count=2)
        manifest = replay_from_trace.build_manifest(trace_dir)
        path = Path(directory) / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return path

    def _run(self, directory, manifest, output_dir, *extra):
        model = Path(directory) / 'model.gguf'
        binary = Path(directory) / 'llama-bench'
        fake_bin = Path(directory) / 'fake-bin'
        fake_bin.mkdir()
        model.write_bytes(b'model-under-test')
        binary.write_bytes(b'android-binary-under-test')
        adb_log = Path(directory) / 'adb.log'
        adb = fake_bin / 'adb'
        adb.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$*" >>"$FAKE_ADB_LOG"\n'
            'if [[ "$1" == "shell" ]]; then printf "{\\"raw\\":true}\\n"; fi\n')
        adb.chmod(0o755)
        env = dict(os.environ)
        env['PATH'] = f'{fake_bin}:{env["PATH"]}'
        env['FAKE_ADB_LOG'] = str(adb_log)
        result = subprocess.run([
            'bash', str(ROOT / 'device' / 'android_bench.sh'),
            '--model', str(model),
            '--binary', str(binary),
            '--threads', '2',
            '--manifest', str(manifest),
            '--output-dir', str(output_dir),
            '--serial', 'fake-device',
            '--repetitions', '2',
            *extra,
        ], env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return adb_log, output_dir

    def test_default_skips_image_encoder_and_records_raw_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(directory)
            output_dir = Path(directory) / 'results'
            adb_log, output_dir = self._run(
                directory, manifest, output_dir)

            adb_lines = adb_log.read_text().splitlines()
            self.assertTrue(any('push' in line and 'model.gguf' in line for line in adb_lines))
            self.assertTrue(any('-p 5' in line and '-n 3' in line for line in adb_lines))
            self.assertEqual(len(list(output_dir.glob('bench_*.json'))), 2)
            commands = [line.split('\t') for line in
                        (output_dir / 'commands.tsv').read_text().splitlines()[1:]]
            self.assertEqual([row[5] for row in commands], [
                'ok', 'skipped_image_encoder', 'ok'])
            metadata = json.loads((output_dir / 'run_metadata.json').read_text())
            self.assertEqual(len(metadata['script_sha256']), 64)
            self.assertEqual(metadata['combinations_total'], 3)
            self.assertEqual(metadata['combinations_ok'], 2)
            self.assertEqual(metadata['combinations_skipped_image_encoder'], 1)
            self.assertEqual(metadata['measurements'], [])

    def test_image_text_only_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(directory)
            output_dir = Path(directory) / 'results'
            _, output_dir = self._run(
                directory, manifest, output_dir, '--include-image-text-only')

            self.assertEqual(len(list(output_dir.glob('bench_*.json'))), 3)
            metadata = json.loads((output_dir / 'run_metadata.json').read_text())
            self.assertTrue(metadata['include_image_text_only'])
            self.assertEqual(metadata['combinations_ok_image_text_only'], 1)
            self.assertEqual(metadata['combinations_skipped_image_encoder'], 0)


if __name__ == '__main__':
    unittest.main()
