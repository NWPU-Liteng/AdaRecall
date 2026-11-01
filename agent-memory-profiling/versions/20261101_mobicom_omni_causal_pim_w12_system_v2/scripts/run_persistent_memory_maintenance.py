"""Replay label-free session arrivals through a Host memory-maintenance engine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def load_index(index_root):
    records = json.loads((index_root / 'records.json').read_text())
    vectors = np.load(index_root / 'vectors.npy', mmap_mode='r')
    if len(records) != len(vectors):
        raise ValueError('records/vectors mismatch')
    return records, vectors


def compact(base_indices, hot_indices, vectors):
    indices = np.asarray(base_indices + hot_indices, dtype=np.int64)
    copied = np.ascontiguousarray(vectors[indices], dtype=np.float32)
    return indices.tolist(), copied


def read_progress(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=0.02)
    parser.add_argument('--delta-fraction', type=float, default=0.10)
    parser.add_argument('--idle-timeout', type=float, default=30.0)
    args = parser.parse_args()
    if args.delta_fraction <= 0 or args.poll_seconds <= 0:
        raise ValueError('positive maintenance parameters required')
    records, vectors = load_index(args.index)
    session_indices = {}
    for index, record in enumerate(records):
        session_indices.setdefault(str(record.get('session_id')), []).append(index)
    base_indices, hot_indices = [], []
    seen_sessions = set()
    seen_progress = None
    last_change = time.perf_counter()
    compactions = []
    append_bytes = 0
    compact_read_bytes = 0
    compact_write_bytes = 0
    maintenance_seconds = 0.0
    started = time.perf_counter()
    while True:
        progress = read_progress(args.progress)
        if progress is not None and progress.get('runtime_labels_read') is not False:
            raise ValueError('progress file is not label-separated')
        if progress is not None and progress.get('complete') is True:
            break
        if progress is not None and progress != seen_progress:
            seen_progress = progress
            last_change = time.perf_counter()
            visible_sessions = set(map(str, progress.get('visible_session_ids', [])))
            new_sessions = sorted(visible_sessions - seen_sessions)
            for session_id in new_sessions:
                additions = session_indices.get(session_id, [])
                hot_indices.extend(additions)
                append_bytes += len(additions) * vectors.shape[1] * vectors.dtype.itemsize
                seen_sessions.add(session_id)
            threshold = max(1, int(max(len(base_indices), 1) * args.delta_fraction))
            if len(hot_indices) >= threshold:
                compact_started = time.perf_counter()
                old_count = len(base_indices)
                base_indices, compacted = compact(base_indices, hot_indices, vectors)
                hot_count = len(hot_indices)
                hot_indices = []
                elapsed = time.perf_counter() - compact_started
                maintenance_seconds += elapsed
                bytes_moved = int((old_count + hot_count) * vectors.shape[1] * vectors.dtype.itemsize)
                compact_read_bytes += bytes_moved
                compact_write_bytes += int(compacted.nbytes)
                compactions.append({
                    'checkpoint_time': progress.get('checkpoint_time'),
                    'old_records': old_count,
                    'appended_records': hot_count,
                    'new_records': len(base_indices),
                    'bytes_read': bytes_moved,
                    'bytes_written': int(compacted.nbytes),
                    'seconds': elapsed,
                    'runtime_labels_read': False,
                })
        if progress is not None and time.perf_counter() - last_change > args.idle_timeout:
            break
        time.sleep(args.poll_seconds)
    if hot_indices:
        compact_started = time.perf_counter()
        old_count = len(base_indices)
        base_indices, compacted = compact(base_indices, hot_indices, vectors)
        elapsed = time.perf_counter() - compact_started
        maintenance_seconds += elapsed
        bytes_moved = int((old_count + len(hot_indices)) * vectors.shape[1] * vectors.dtype.itemsize)
        compact_read_bytes += bytes_moved
        compact_write_bytes += int(compacted.nbytes)
        compactions.append({
            'checkpoint_time': 'final_flush',
            'old_records': old_count,
            'appended_records': len(hot_indices),
            'new_records': len(base_indices),
            'bytes_read': bytes_moved,
            'bytes_written': int(compacted.nbytes),
            'seconds': elapsed,
            'runtime_labels_read': False,
        })
    result = {
        'schema': 'persistent-memory-maintenance-host-v1',
        'person_id': str(records[0]['record_id']).split(':')[1] if records else None,
        'records': len(records),
        'vector_bytes': int(vectors.nbytes),
        'append_bytes': append_bytes,
        'compact_read_bytes': compact_read_bytes,
        'compact_write_bytes': compact_write_bytes,
        'maintenance_seconds': maintenance_seconds,
        'wall_seconds': time.perf_counter() - started,
        'compactions': compactions,
        'runtime_labels_read': False,
        'status': 'pass',
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'status': result['status'],
        'compactions': len(compactions),
        'compact_read_bytes': compact_read_bytes,
        'maintenance_seconds': maintenance_seconds,
        'output': str(args.output),
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
