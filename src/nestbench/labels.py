"""Per-dataset labels for the benchmark suite: rotations, energy spectra, C25, density, splits, LID.

**Orders.** Every set gets three coordinate orders, so the ordering ablation (E7) and the
D/C25 analysis (E5) can use any of them:
- ``native``: the stored coordinates (the MRL prefix order for MS MARCO);
- ``pca``: principal axes of the base set, sorted by decreasing variance;
- ``random``: a seeded uniformly random rotation (the control).

All three are orthogonal maps (plus a translation for PCA), so L2 distances, and therefore the
ground truth, are the same in every order.

**Energy.** NestSQ's bound prunes on the unread tail of the *difference* q - x. The relevant
per-axis energy is therefore the variance (centred second moment) along each axis of the order.
For two independent draws, E[(q_d - x_d)^2] = 2 Var_d. ``tail[p]`` is the share of total variance
in axes p..D-1, and ``C25`` = 1 - tail[D/4] is the share in the first 25% of axes (v4 A2/C4).
The uncentred version (share of the squared norm) is recorded too, for comparison with how v4
first worded C25.

**Density** (E6). Per query: the distance to its 10th true neighbour (sqrt of the GT squared L2),
plus its quartile (0 = densest).

**Splits** (E9). ``n_splits`` seeded 50/50 calibration/test resamples of the query set, stored
as a (n_splits, nq) int8 mask (1 = calibration).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def covariance(base: np.ndarray, chunk: int = 200_000, device: str = "cuda") -> tuple:
    """Mean (float64) and covariance (float64, D x D) of ``base``.

    Gram blocks are computed on the GPU in fp32 without TF32 and accumulated in float64 on the
    centred data, using a float64 mean from a first pass.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    N, D = base.shape
    s = np.zeros(D)
    for a in range(0, N, chunk):
        s += np.asarray(base[a:a + chunk], dtype=np.float64).sum(0)
    mean = s / N
    mu = torch.from_numpy(mean.astype(np.float32)).to(device)
    G = torch.zeros((D, D), dtype=torch.float64, device=device)
    for a in range(0, N, chunk):
        x = torch.from_numpy(np.ascontiguousarray(base[a:a + chunk], dtype=np.float32))
        x = x.to(device) - mu
        G += (x.T @ x).double()
    cov = G.cpu().numpy() / (N - 1)
    del G
    torch.cuda.empty_cache()
    return mean, (cov + cov.T) / 2


def pca(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending) and components (rows = principal axes, matching order)."""
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    # deterministic sign: largest-magnitude entry of each axis positive
    sign = np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])
    sign[sign == 0] = 1
    return np.clip(w, 0, None), (V * sign).T


def random_rotation(D: int, seed: int) -> np.ndarray:
    """Haar-uniform random orthogonal matrix (rows = new axes)."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((D, D)))
    return (Q * np.sign(np.diag(R))).T


def axis_variance(cov: np.ndarray, rot: np.ndarray | None) -> np.ndarray:
    """Variance along each row-axis of ``rot`` (identity if None): diag(R C R^T)."""
    if rot is None:
        return np.diag(cov).copy()
    return np.einsum("ij,jk,ik->i", rot, cov, rot)


def tail_curve(var: np.ndarray) -> np.ndarray:
    """tail[p] = share of total variance in axes p..D-1, for p = 0..D (tail[0]=1, tail[D]=0)."""
    c = np.concatenate([[0.0], np.cumsum(var)])
    return 1.0 - c / c[-1]


def c25(var: np.ndarray) -> float:
    D = len(var)
    return float(1.0 - tail_curve(var)[D // 4])


def second_moment_share(base: np.ndarray, rot: np.ndarray | None, mean: np.ndarray | None,
                        frac: float = 0.25, n_sample: int = 100_000, seed: int = 0) -> float:
    """Uncentred variant: share of E|x|^2 in the first ``frac`` of axes (for the record)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(base), min(n_sample, len(base)), replace=False))
    x = np.asarray(base[idx], dtype=np.float64)
    if mean is not None:  # PCA order is defined on centred data
        x = x - mean
    if rot is not None:
        x = x @ rot.T
    e = (x ** 2).mean(0)
    return float(e[: int(len(e) * frac)].sum() / e.sum())


def density(gt_dist: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(np.maximum(gt_dist[:, k - 1].astype(np.float64), 0))
    q = np.quantile(r, [0.25, 0.5, 0.75])
    return r, np.searchsorted(q, r, side="right").astype(np.int8)


def splits(nq: int, n_splits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros((n_splits, nq), dtype=np.int8)
    for i in range(n_splits):
        m[i, rng.permutation(nq)[: nq // 2]] = 1
    return m


def lid_mle(base: np.ndarray, n_sample: int = 10_000, k: int = 20, seed: int = 42) -> tuple:
    """MLE local intrinsic dimension (Levina-Bickel), as in fewer_bits (annstudy.truncate),
    but with neighbours taken from the FULL base set (exact, GPU) rather than from the sample."""
    from nestbench.gt import exact_knn

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(base), min(n_sample, len(base)), replace=False))
    r = exact_knn(base, np.asarray(base[idx]), k=k + 1, extra=32)
    d = np.sqrt(np.maximum(r["dist"][:, 1:].astype(np.float64), 1e-30))  # drop self
    lid = -k / np.log(d[:, :-1] / d[:, -1:]).sum(1)
    lid = lid[np.isfinite(lid)]
    return float(lid.mean()), float(lid.std())
