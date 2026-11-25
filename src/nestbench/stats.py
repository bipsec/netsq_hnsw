"""Paired statistics for per-query metrics — copied from fewer_bits
(annstudy.analysis.relevance_stats): paired bootstrap mean CI, two-sided paired sign-flip
randomization test, Holm correction. Used for MRR@10 (E12/C5) and calibration (E9)."""

import numpy as np

B = 10_000

def _boot_idx(n, b=B, seed=0, chunk=1000):
    rng = np.random.default_rng(seed)
    for s in range(0, b, chunk):
        yield rng.integers(0, n, size=(min(chunk, b - s), n))


def boot_mean_ci(x: np.ndarray, b=B, seed=0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    means = np.concatenate([x[i].mean(axis=1) for i in _boot_idx(len(x), b, seed)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def randomization_p(diff: np.ndarray, b=B, seed=1) -> float:
    """Two-sided paired sign-flip test on per-query differences."""
    diff = np.asarray(diff, float)
    obs = abs(diff.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(0, b, 1000):
        k = min(1000, b - s)
        signs = rng.choice([-1.0, 1.0], size=(k, len(diff)))
        hits += int((np.abs((signs * diff).mean(axis=1)) >= obs - 1e-15).sum())
    return (hits + 1) / (b + 1)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


