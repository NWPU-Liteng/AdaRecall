"""Runtime guard that prevents serving code from opening evaluation labels."""
from __future__ import annotations

import builtins
import os


LABEL_FILENAME = 'labels.jsonl'
_ORIGINAL_OPEN = builtins.open


def _normalized_path(file) -> str | None:
    if not isinstance(file, (str, bytes, os.PathLike)):
        return None
    try:
        raw = os.fspath(file)
    except TypeError:
        return None
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    return raw.replace('\\', '/')


def is_label_path(file) -> bool:
    path = _normalized_path(file)
    if path is None:
        return False
    return path == LABEL_FILENAME or path.endswith('/' + LABEL_FILENAME)


def _guarded_open(file, *args, **kwargs):
    if is_label_path(file):
        raise PermissionError(
            f'runtime label isolation forbids opening {LABEL_FILENAME}: {file!s}')
    return _ORIGINAL_OPEN(file, *args, **kwargs)


def install_label_read_guard() -> None:
    """Install the process-wide guard once; later calls are no-ops."""
    if builtins.open is _guarded_open:
        return
    builtins.open = _guarded_open
