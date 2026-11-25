"""Exact top-k ground truth (squared L2) on the GPU, with a per-query exactness certificate.

Method:
1. **Candidate pass.** On the GPU, distances are computed in fp32 as ``|q|^2 + |x|^2 - 2 q.x``
   (TF32 disabled), keeping the ``k + extra`` smallest per query over base chunks.
2. **Re-score.** The candidates are re-scored exactly in float64 on the CPU, as the sum of
   squared differences, and re-sorted by (distance, id).
3. **Certificate.** Let ``e`` be a bound on the pass's error for any single distance. The
   re-scored top-k is exact if the k-th exact distance is at most the (k+extra)-th pass
   distance minus 2e: then no point outside the candidate list can be closer than the k-th
   returned point.
4. **Fallback.** Queries that fail the certificate are re-run with the candidate pass in float64
   on the GPU (error bound ~1e-13 relative), then re-scored and certified again. Queries that
   still fail (only exact ties spanning the whole candidate list) go to an exact CPU float64
   scan that keeps every point tied at the k-th distance.

The error bound uses the standard dot-product bound ``|fl(q.x) - q.x| <= gamma_D |q||x|`` with
``gamma_D = D u / (1 - D u)``. The norms add ``gamma_D`` terms of their own, giving
``e = gamma_{D+2} (|q|^2 + |x|^2 + 2|q||x|)``, bounded using the maximum base norm.

Distances are returned as float32 values of the exact float64 distances; ids are int64.
"""

import logging
import math
import time

import numpy as np

logger = logging.getLogger(__name__)

UNIT = {"float32": 2.0 ** -24, "float64": 2.0 ** -53}


def _gamma(n: int, u: float) -> float:
    return n * u / (1 - n * u)


def _candidates(base, queries, kk, dtype, base_chunk, q_chunk, device):
    """GPU pass: (k+extra) smallest expanded distances per query in ``dtype``."""
    import torch

    tdt = getattr(torch, dtype)
    N = len(base)
    nq = len(queries)
    q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(device).to(tdt)
    qn = (q * q).sum(1)
    best_d = torch.full((nq, kk), float("inf"), device=device, dtype=tdt)
    best_i = torch.full((nq, kk), -1, dtype=torch.int64, device=device)
    max_xn = 0.0
    for s in range(0, N, base_chunk):
        xb = torch.from_numpy(np.ascontiguousarray(base[s:s + base_chunk], dtype=np.float32))
        xb = xb.to(device).to(tdt)
        xn = (xb * xb).sum(1)
        max_xn = max(max_xn, float(xn.max()))
        for a in range(0, nq, q_chunk):
            d = qn[a:a + q_chunk, None] + xn[None, :] - 2.0 * (q[a:a + q_chunk] @ xb.T)
            dv, di = torch.topk(d, min(kk, d.shape[1]), dim=1, largest=False)
            cd = torch.cat([best_d[a:a + q_chunk], dv], 1)
            ci = torch.cat([best_i[a:a + q_chunk], di + s], 1)
            o = torch.topk(cd, kk, dim=1, largest=False)
            best_d[a:a + q_chunk] = o.values
            best_i[a:a + q_chunk] = torch.gather(ci, 1, o.indices)
        del xb, xn
    out = (best_i.cpu().numpy(), best_d.cpu().double().numpy(), qn.cpu().double().numpy(),
           max_xn)
    del q, best_d, best_i
    torch.cuda.empty_cache()
    return out


def _rescore(base, queries, cand, k):
    """Exact float64 distances (sum of squared differences) of the candidates; top-k by
    (distance, id)."""
    q64 = np.asarray(queries, dtype=np.float64)
    nq = len(cand)
    ids = np.empty((nq, k), dtype=np.int64)
    dist = np.empty((nq, k), dtype=np.float64)
    for a in range(0, nq, 256):
        c = cand[a:a + 256]
        uniq = np.unique(c)
        x = np.asarray(base[uniq], dtype=np.float64)
        for r in range(len(c)):
            dd = ((x[np.searchsorted(uniq, c[r])] - q64[a + r]) ** 2).sum(1)
            order = np.lexsort((c[r], dd))[:k]
            ids[a + r], dist[a + r] = c[r][order], dd[order]
    return ids, dist


def _certify(dist_k, d_kk, qn, max_xn, D, dtype):
    err = _gamma(D + 2, UNIT[dtype]) * (qn + max_xn + 2.0 * np.sqrt(qn * max_xn))
    margin = d_kk - 2.0 * err - dist_k
    return margin >= 0, margin, err


def _cpu_exact(base, q, k):
    N = len(base)
    dd = np.empty(N)
    for s in range(0, N, 500_000):
        dd[s:s + 500_000] = ((np.asarray(base[s:s + 500_000], dtype=np.float64) - q) ** 2).sum(1)
    kth = np.partition(dd, k - 1)[k - 1]
    top = np.flatnonzero(dd <= kth)  # every point tied at the k-th distance is a candidate
    order = top[np.lexsort((top, dd[top]))][:k]
    return order, dd[order]


def exact_knn(base: np.ndarray, queries: np.ndarray, k: int = 100, extra: int = 64,
              base_chunk: int = 200_000, q_chunk: int = 2048, device: str = "cuda") -> dict:
    """Exact k-NN of ``queries`` in ``base`` (squared L2). Returns ids, dist, and diagnostics."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    N, D = base.shape
    kk = min(k + extra, N)
    queries = np.asarray(queries, dtype=np.float32)

    cand, dpass, qn, max_xn = _candidates(base, queries, kk, "float32", base_chunk, q_chunk,
                                          device)
    ids, dist = _rescore(base, queries, cand, k)
    if kk >= N:
        ok = np.ones(len(queries), dtype=bool)
        margin = np.full(len(queries), math.inf)
        err = np.zeros(len(queries))
    else:
        ok, margin, err = _certify(dist[:, k - 1], dpass[:, kk - 1], qn, max_xn, D, "float32")
    n_fail32 = int((~ok).sum())

    n_fail64 = 0
    if n_fail32:  # float64 GPU candidate pass for the uncertified queries
        bad = np.flatnonzero(~ok)
        c2, d2, qn2, mx2 = _candidates(base, queries[bad], kk, "float64", 50_000, 256, device)
        i2, s2 = _rescore(base, queries[bad], c2, k)
        ok2, m2, _ = _certify(s2[:, k - 1], d2[:, kk - 1], qn2, mx2, D, "float64")
        ids[bad], dist[bad], margin[bad] = i2, s2, m2
        for j in np.flatnonzero(~ok2):  # exact ties across the whole candidate list
            ids[bad[j]], dist[bad[j]] = _cpu_exact(base, queries[bad[j]].astype(np.float64), k)
            margin[bad[j]] = 0.0
        n_fail64 = int((~ok2).sum())
    secs = time.time() - t0
    logger.info(f"exact_knn N={N} D={D} nq={len(queries)}: {secs:.0f}s, "
                f"fp32-uncertified={n_fail32}, fp64-uncertified={n_fail64}")
    return {"ids": ids, "dist": dist.astype(np.float32), "n_uncertified": n_fail32,
            "n_uncertified_fp64": n_fail64, "max_err_bound": float(np.max(err)),
            "seconds": secs, "min_cert_margin": float(np.min(margin))}


def agreement(ids_a: np.ndarray, ids_b: np.ndarray, dist_ref: np.ndarray, base: np.ndarray,
              queries: np.ndarray, k: int = 10, tol: float = 1e-5) -> dict:
    """Compare two GT id lists at k, tie-tolerantly: share of ``ids_b``'s top-k whose true
    distance is within the k-th reference distance x (1+tol), plus exact id-set agreement."""
    from nestbench.metrics import recall_at_k_batch

    rec = recall_at_k_batch(ids_b[:, :k], dist_ref, base, queries, k=k, tol=tol)
    same = np.mean([set(a[:k]) == set(b[:k]) for a, b in zip(ids_a, ids_b, strict=True)])
    return {"tie_tolerant_recall": float(rec.mean()), "min_query_recall": float(rec.min()),
            "exact_set_match": float(same)}
