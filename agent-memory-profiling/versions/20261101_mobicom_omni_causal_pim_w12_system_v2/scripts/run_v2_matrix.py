#!/usr/bin/env python3
"""Run the frozen Phase 5 validation arms sequentially with hard GPU guards."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
READER_USERS = ('4', '5', '6', '7')
READER_GPUS = ('0', '1', '3', '5')
JUDGE_GPU = '6'
FORBIDDEN_GPUS = {'2', '4'}
ARMS = (
    'v2_asof_op',
    'v2_asof_op_conf',
    'v2_base',
    'v2_asof',
    'v2_asof_op_conf_nor',
    'v2_full_shadow',
    'v2_full_retq',
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def wait_for_judge_complete(judge_process: subprocess.Popen, run_root: Path,
                            expected_questions: int) -> None:
    marker = run_root / 'judge_complete.json'
    while True:
        if marker.is_file():
            try:
                payload = json.loads(marker.read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
            if (payload.get('complete') is True
                    and payload.get('questions') == expected_questions):
                code = judge_process.wait()
                if code:
                    raise RuntimeError(f'incremental Judge failed: {code}')
                return
        code = judge_process.poll()
        if code is not None:
            raise RuntimeError(
                f'incremental Judge exited before completion marker: {code}')
        time.sleep(2)


def load_config(arm: str) -> dict:
    path = ROOT / 'configs' / f'{arm}.yaml'
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f'configuration must be a mapping: {path}')
    return config


def validate_gpu_contract(config: dict) -> tuple[tuple[str, ...], str]:
    execution = config['execution']
    readers = tuple(str(value) for value in execution['reader_gpus'])
    judge = str(execution['judge_gpu'])
    if readers != READER_GPUS:
        raise ValueError(f'Reader GPUs must be {READER_GPUS}, got {readers}')
    if judge != JUDGE_GPU:
        raise ValueError(f'Judge GPU must be {JUDGE_GPU}, got {judge}')
    if FORBIDDEN_GPUS & (set(readers) | {judge}):
        raise ValueError('GPU 2/4 are forbidden')
    return readers, judge


def run_logged(command: list[str], *, log_path: Path, env: dict | None = None,
               check: bool = True) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as log:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, text=True)
    if check and completed.returncode:
        raise RuntimeError(
            f'command failed ({completed.returncode}): {" ".join(command)}')
    return completed


def terminate_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def validate_reader_outputs(config: dict, cohort: dict) -> list[Path]:
    run_root = Path(config['paths']['run_root'])
    paths = []
    for user in READER_USERS:
        path = run_root / f'user_{user}_shard_0.json'
        if not path.is_file():
            raise FileNotFoundError(f'Reader output missing: {path}')
        payload = json.loads(path.read_text())
        if payload.get('complete') is not True:
            raise ValueError(f'Reader output incomplete: {path}')
        rows = payload.get('rows')
        if not isinstance(rows, list):
            raise ValueError(f'Reader rows must be a list: {path}')
        expected = cohort['question_ids_by_user'][user]
        observed = [row.get('question_id') for row in rows]
        if observed != expected:
            raise ValueError(f'Reader question order differs: {path}')
        if any(row.get('person_id') != user for row in rows):
            raise ValueError(f'Reader person id differs: {path}')
        if any(row.get('evaluation_label_files_read') != [] for row in rows):
            raise ValueError(f'Reader read labels: {path}')
        paths.append(path)
    return paths


def build_validation_links(run_root: Path) -> Path:
    validation = run_root / 'validation_only'
    validation.mkdir(parents=True, exist_ok=True)
    for user in READER_USERS:
        source = run_root / f'user_{user}_shard_0.json'
        destination = validation / source.name
        if destination.exists():
            if destination.stat().st_ino != source.stat().st_ino:
                raise ValueError(f'validation link is not the completed output: {destination}')
            continue
        os.link(source, destination)
    return validation


def judge_rows(run_root: Path, cohort: dict) -> list[dict]:
    rows = []
    expected = {
        identity for user in READER_USERS
        for identity in cohort['question_ids_by_user'][user]
    }
    for user in READER_USERS:
        for path in sorted((run_root / 'judge_rows' / user).glob('*.json')):
            rows.append(json.loads(path.read_text()))
    identities = {row['question_id'] for row in rows}
    if identities != expected or len(rows) != len(expected):
        raise ValueError(
            f'Judge rows do not match validation cohort: {len(rows)} vs {len(expected)}')
    return rows


def write_judge_summary(run_root: Path, cohort: dict) -> dict:
    rows = judge_rows(run_root, cohort)
    by_type = {}
    for question_type in sorted({row.get('question_type') for row in rows}):
        selected = [row for row in rows if row.get('question_type') == question_type]
        by_type[str(question_type)] = {
            'n': len(selected),
            'correct': sum(
                row.get('parse_valid') is True and row.get('judge_label') == 'CORRECT'
                for row in selected),
        }
    correct = sum(
        row.get('parse_valid') is True and row.get('judge_label') == 'CORRECT'
        for row in rows)
    invalid = sum(row.get('parse_valid') is not True for row in rows)
    summary = {
        'n': len(rows),
        'correct': correct,
        'wrong': len(rows) - correct - invalid,
        'invalid': invalid,
        'by_type': by_type,
    }
    atomic_write(run_root / 'validation_judge_summary.json', summary)
    return summary


def postprocess(config: dict, cohort: dict, validation: Path) -> dict:
    run_root = Path(config['paths']['run_root'])
    paths = config['paths']
    prefixes = config['extensions'].get('progressive_topk', {})
    calibration = config['extensions'].get('calibration') or {}
    candidate_dump = config['extensions'].get('candidate_dump')
    command = [
        PYTHON, str(ROOT / 'scripts/report_run_stats.py'), str(validation)]
    run_logged(command, log_path=run_root / 'postprocess.log')
    recall_command = [
        PYTHON, str(ROOT / 'scripts/eval_evidence_recall.py'), str(validation),
        '--labels', str(paths['prepared'] + '/labels.jsonl'),
        '--index-root', str(paths['index_root']),
    ]
    if candidate_dump:
        recall_command.extend(['--candidate-dump', str(candidate_dump)])
    run_logged(recall_command, log_path=run_root / 'postprocess.log')
    outputs = ['run_stats.json', 'run_stats.md', 'evidence_recall.json']
    if calibration.get('candidate_policy') == 'conformal':
        run_logged([
            PYTHON, str(ROOT / 'scripts/eval_conformal_coverage.py'),
            '--dump', str(candidate_dump),
            '--labels', str(paths['prepared'] + '/labels.jsonl'),
            '--calibrator', str(calibration['calibrator_path']),
            '--index-root', str(paths['index_root']),
            '--out', str(run_root / 'validation_conformal_coverage.json'),
        ], log_path=run_root / 'postprocess.log')
        outputs.append('validation_conformal_coverage.json')
    if config['extensions'].get('workload_trace') == 'on':
        replay_output = run_root / 'pim_replay.json'
        if replay_output.exists():
            raise ValueError(f'stale replay output requires quarantine: {replay_output}')
        run_logged([
            PYTHON, str(ROOT / 'scripts/pim_replay.py'),
            str(run_root / 'trace'), str(validation),
            '--output', str(replay_output),
        ], log_path=run_root / 'postprocess.log')
        outputs.append('pim_replay.json')
    if not prefixes:
        raise ValueError('progressive_topk configuration missing')
    return {
        'outputs': outputs,
        'candidate_dump': candidate_dump,
    }


def arm_complete(arm: str, config: dict, cohort: dict, reader_paths: list[Path],
                 judge_summary: dict, postprocess_result: dict,
                 readers: tuple[str, ...], judge_gpu: str,
                 started: float) -> None:
    run_root = Path(config['paths']['run_root'])
    atomic_write(run_root / 'matrix_arm_complete.json', {
        'schema': 'mobilemem-v2-matrix-arm-complete-v1',
        'complete': True,
        'arm': arm,
        'finished_at': now(),
        'elapsed_seconds': time.time() - started,
        'users': list(READER_USERS),
        'reader_gpus': list(readers),
        'judge_gpu': judge_gpu,
        'forbidden_gpus': sorted(FORBIDDEN_GPUS),
        'reader_outputs': {
            path.name: {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in reader_paths
        },
        'judge': judge_summary,
        'postprocess': postprocess_result,
        'physical_pim_executed': False,
        'boundary': (
            'validation-only quality run; fixed local Qwen3-VL-8B Judge; '
            'no physical phone, PIM, energy, thermal, or latency measurement'),
    })


def run_arm(arm: str) -> None:
    config = load_config(arm)
    readers, judge_gpu = validate_gpu_contract(config)
    run_root = Path(config['paths']['run_root'])
    if (run_root / 'matrix_arm_complete.json').is_file():
        print(json.dumps({'arm': arm, 'status': 'already_complete'}), flush=True)
        return
    cohort = json.loads((ROOT / config['paths']['cohort']).read_text())
    started = time.time()
    processes: list[subprocess.Popen] = []
    judge_process = None
    try:
        for user, gpu in zip(READER_USERS, readers):
            environment = dict(os.environ)
            environment['CUDA_VISIBLE_DEVICES'] = gpu
            environment['OMP_NUM_THREADS'] = '2'
            environment['MKL_NUM_THREADS'] = '2'
            environment['OPENBLAS_NUM_THREADS'] = '2'
            environment['TOKENIZERS_PARALLELISM'] = 'false'
            log_path = run_root / 'matrix' / f'serve_user_{user}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open('a')
            process = subprocess.Popen(
                [PYTHON, str(ROOT / 'scripts/run_main_experiment.py'),
                 '--config', str(ROOT / 'configs' / f'{arm}.yaml'),
                 '--mode', 'serve', '--user-id', user, '--gpu', gpu],
                cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            process._matrix_log = log
            processes.append(process)
        judge_environment = dict(os.environ)
        judge_environment['CUDA_VISIBLE_DEVICES'] = judge_gpu
        judge_log = (run_root / 'matrix' / 'judge.log').open('a')
        judge_process = subprocess.Popen(
            [PYTHON, str(ROOT / 'scripts/judge_incremental.py'),
             '--config', str(ROOT / 'configs' / f'{arm}.yaml'),
             '--users', *READER_USERS],
            cwd=ROOT, env=judge_environment, stdout=judge_log,
            stderr=subprocess.STDOUT, start_new_session=True)
        reader_failures = []
        for process, user in zip(processes, READER_USERS):
            code = process.wait()
            process._matrix_log.close()
            if code:
                reader_failures.append((user, code))
        if reader_failures:
            raise RuntimeError(f'Reader jobs failed: {reader_failures}')
        reader_paths = validate_reader_outputs(config, cohort)
        validation = build_validation_links(run_root)
        postprocess_result = postprocess(config, cohort, validation)
        expected_judge_questions = sum(
            len(cohort['question_ids_by_user'][user]) for user in READER_USERS)
        wait_for_judge_complete(judge_process, run_root, expected_judge_questions)
        judge_log.close()
        judge_summary = write_judge_summary(run_root, cohort)
        if judge_summary['invalid'] != 0:
            raise ValueError(f"Judge produced invalid rows: {judge_summary['invalid']}")
        arm_complete(
            arm, config, cohort, reader_paths, judge_summary,
            postprocess_result, readers, judge_gpu, started)
        print(json.dumps({
            'arm': arm,
            'status': 'complete',
            'elapsed_seconds': time.time() - started,
            'judge': judge_summary,
        }), flush=True)
    except Exception as error:
        for process in processes:
            terminate_group(process)
        if judge_process is not None:
            terminate_group(judge_process)
        atomic_write(run_root / 'matrix_arm_failed.json', {
            'schema': 'mobilemem-v2-matrix-arm-failed-v1',
            'complete': False,
            'arm': arm,
            'failed_at': now(),
            'error': repr(error),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    run_root = ROOT / 'results'
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / '.v2_matrix.lock'
    lock_handle = lock_path.open('a+')
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'another matrix runner owns {lock_path}')
    for arm in args.arms:
        run_arm(arm)


if __name__ == '__main__':
    main()
