from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .causal_tile_layout import TileLayout


@dataclass(frozen=True)
class TileRange:
    tile_id: int
    offset: int
    length: int
    payload_bytes: int
    embedding_bytes: int
    metadata_bytes: int
    padding_bytes: int


class PhysicalTileStore:
    ALIGNMENT = 4096

    def __init__(self, root: Path, ranges: dict[int, TileRange]) -> None:
        self.root = root
        self.data_path = root / "tiles.bin"
        self.ranges = ranges

    @classmethod
    def materialize(
        cls,
        root: Path,
        layout: TileLayout,
        memory_ids: Sequence[str],
        payloads: Sequence[str],
        embeddings: np.ndarray,
    ) -> "PhysicalTileStore":
        root.mkdir(parents=True, exist_ok=False)
        data_path = root / "tiles.bin"
        ranges: dict[int, TileRange] = {}
        with data_path.open("wb") as handle:
            for tile_id in sorted(layout.members_by_tile):
                start = handle.tell()
                padding = (-start) % cls.ALIGNMENT
                if padding:
                    handle.write(b"\0" * padding)
                    start += padding
                rows = layout.members_by_tile[tile_id]
                payload_blob = bytearray()
                embedding_blob = bytearray()
                metadata = []
                for local in rows:
                    payload = str(payloads[local]).encode("utf-8")
                    vector = np.asarray(embeddings[local], dtype=np.float32).tobytes(order="C")
                    metadata.append(
                        {
                            "memory_id": str(memory_ids[local]),
                            "payload_offset": len(payload_blob),
                            "payload_length": len(payload),
                            "embedding_offset": len(embedding_blob),
                            "embedding_length": len(vector),
                        }
                    )
                    payload_blob.extend(payload)
                    embedding_blob.extend(vector)
                metadata_blob = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
                header = struct.pack("<QQQ", len(payload_blob), len(embedding_blob), len(metadata_blob))
                handle.write(header)
                handle.write(payload_blob)
                handle.write(embedding_blob)
                handle.write(metadata_blob)
                length = len(header) + len(payload_blob) + len(embedding_blob) + len(metadata_blob)
                ranges[tile_id] = TileRange(tile_id, start, length, len(payload_blob), len(embedding_blob), len(metadata_blob) + len(header), padding)
        manifest = {
            str(tile): {
                "offset": value.offset,
                "length": value.length,
                "payload_bytes": value.payload_bytes,
                "embedding_bytes": value.embedding_bytes,
                "metadata_bytes": value.metadata_bytes,
                "padding_bytes": value.padding_bytes,
            }
            for tile, value in ranges.items()
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return cls(root, ranges)

    def read_tiles(self, tile_ids: Sequence[int], *, cold: bool = False) -> dict[str, float | int]:
        unique = sorted(set(int(value) for value in tile_ids))
        if cold and hasattr(os, "posix_fadvise"):
            with self.data_path.open("rb", buffering=0) as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        started = time.perf_counter_ns()
        total = 0
        with self.data_path.open("rb", buffering=0) as handle:
            for tile_id in unique:
                row = self.ranges[tile_id]
                handle.seek(row.offset)
                payload = handle.read(row.length)
                if len(payload) != row.length:
                    raise IOError(f"short physical read for tile {tile_id}")
                total += len(payload)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return {
            "unique_tiles": len(unique),
            "payload_bytes": sum(self.ranges[tile].payload_bytes for tile in unique),
            "embedding_bytes": sum(self.ranges[tile].embedding_bytes for tile in unique),
            "metadata_bytes": sum(self.ranges[tile].metadata_bytes for tile in unique),
            "padding_bytes": sum(self.ranges[tile].padding_bytes for tile in unique),
            "total_fetched_bytes": total,
            "read_latency_ms": elapsed_ms,
            "cold_requested": cold,
        }
