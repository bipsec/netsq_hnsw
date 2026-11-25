"""Readers for the public ANN formats (texmex .fvecs/.ivecs/.bvecs, big-ann .fbin).

These parse untrusted downloads with numpy only. Each reader validates the per-record
dimension field so a truncated or malformed file fails loudly instead of being reinterpreted.
"""

from pathlib import Path

import numpy as np


def _vecs(path: Path, dtype, elem: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"{path}: empty")
    d = int(raw[:4].view(np.int32)[0])
    rec = 4 + d * elem
    if d <= 0 or raw.size % rec:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of record {rec} (d={d})")
    n = raw.size // rec
    rows = raw.reshape(n, rec)
    dims = rows[:, :4].copy().view(np.int32).ravel()
    if not np.all(dims == d):
        raise ValueError(f"{path}: inconsistent dimension fields")
    return rows[:, 4:].copy().view(dtype).reshape(n, d)


def read_fvecs(path) -> np.ndarray:
    return _vecs(Path(path), np.float32, 4)


def read_ivecs(path) -> np.ndarray:
    return _vecs(Path(path), np.int32, 4)


def read_bvecs(path) -> np.ndarray:
    return _vecs(Path(path), np.uint8, 1)


def read_fbin(path, n_max: int | None = None) -> np.ndarray:
    """big-ann .fbin: int32 n, int32 d, then n*d float32. ``n_max`` reads a prefix; a file that
    was range-downloaded (header says 1B, body holds fewer) is accepted when n_max is given and
    the body holds at least n_max rows."""
    path = Path(path)
    with open(path, "rb") as f:
        n, d = (int(v) for v in np.fromfile(f, dtype=np.int32, count=2))
    body = (path.stat().st_size - 8) // (4 * d)
    take = n if n_max is None else min(n_max, n)
    if body < take:
        raise ValueError(f"{path}: header n={n}, d={d}; body holds {body} rows < {take}")
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(take, d))
