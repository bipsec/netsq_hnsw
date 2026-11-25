"""Recall, latency percentiles, and bootstrap confidence intervals.

Copied from fewer_bits (annstudy.metrics, SIGIR study) unchanged, so recall and latency are
computed identically in both papers. Recall never trusts index distances: it recomputes true
squared L2 from the raw vectors and counts results within the k-th true distance x (1 + tol).
"""

import numpy as np


def recall_at_k(
    retrieved_ids: np.ndarray,
    gt_ids: np.ndarray,
    gt_distances: np.ndarray,
    passages: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    tol: float = 1e-5,
) -> np.ndarray:
    """Per-query recall by distance tolerance.

    For each query:
      1. Get the k-th ground-truth squared distance d_k.
      2. For each retrieved ID, recompute the true squared distance from raw vectors.
      3. Count how many retrieved items have true distance <= d_k * (1 + tol).
      4. Recall = count / k.

    Args:
        retrieved_ids: shape (n_queries, k_ret) — IDs returned by the index.
        gt_ids: shape (n_queries, k_gt) — ground truth IDs, k_gt >= k.
        gt_distances: shape (n_queries, k_gt) — ground truth squared L2 distances.
        passages: the raw passage vectors, shape (N, D).
        queries: the raw query vectors, shape (n_queries, D).
        k: number of neighbors for recall computation.
        tol: relative tolerance for distance comparison.

    Returns:
        Array of per-query recall values, shape (n_queries,).
    """
    n_queries = retrieved_ids.shape[0]
    k_ret = min(retrieved_ids.shape[1], k)
    recalls = np.zeros(n_queries, dtype=np.float64)

    gt_k_dist = gt_distances[:, k - 1]

    for qi in range(n_queries):
        query = queries[qi]
        threshold = gt_k_dist[qi] * (1.0 + tol)
        count = 0

        for j in range(k_ret):
            rid = retrieved_ids[qi, j]
            if rid < 0:
                continue
            true_dist = np.sum((passages[rid].astype(np.float64) - query.astype(np.float64)) ** 2)
            if true_dist <= threshold:
                count += 1

        recalls[qi] = count / k

    return recalls


def recall_at_k_batch(
    retrieved_ids: np.ndarray,
    gt_distances: np.ndarray,
    passages: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    tol: float = 1e-5,
) -> np.ndarray:
    """Vectorized recall computation for better performance on large query sets.

    Same logic as recall_at_k but processes in batches.
    """
    n_queries = retrieved_ids.shape[0]
    k_ret = min(retrieved_ids.shape[1], k)
    recalls = np.zeros(n_queries, dtype=np.float64)

    gt_k_dist = gt_distances[:, k - 1]

    batch_size = 100
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)

        for qi in range(start, end):
            query = queries[qi].astype(np.float64)
            threshold = gt_k_dist[qi] * (1.0 + tol)
            ret_ids = retrieved_ids[qi, :k_ret]
            valid = ret_ids >= 0

            if not np.any(valid):
                continue

            valid_ids = ret_ids[valid]
            ret_vecs = passages[valid_ids].astype(np.float64)
            dists = np.sum((ret_vecs - query) ** 2, axis=1)
            recalls[qi] = np.sum(dists <= threshold) / k

    return recalls


def latency_stats(per_query_ns: np.ndarray) -> dict:
    """Compute mean, p50, p95, p99 in microseconds from nanosecond timings."""
    us = per_query_ns / 1000.0
    return {
        "lat_mean_us": float(np.mean(us)),
        "lat_p50_us": float(np.percentile(us, 50)),
        "lat_p95_us": float(np.percentile(us, 95)),
        "lat_p99_us": float(np.percentile(us, 99)),
    }


def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns (point_estimate, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    point = float(stat_fn(values))

    boot_stats = np.zeros(n_resamples)
    n = len(values)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = (1.0 - ci) / 2.0
    ci_lo = float(np.percentile(boot_stats, 100 * alpha))
    ci_hi = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))

    return point, ci_lo, ci_hi


def bootstrap_recall(
    per_query_recall: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean recall."""
    return bootstrap_ci(per_query_recall, np.mean, n_resamples=n_resamples, seed=seed)


def bootstrap_latency_percentile(
    per_query_ns: np.ndarray,
    percentile: float = 95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for a latency percentile (returns microseconds)."""
    us = per_query_ns / 1000.0

    def pctl(arr):
        return np.percentile(arr, percentile)

    return bootstrap_ci(us, pctl, n_resamples=n_resamples, seed=seed)


def measure_latency(index, queries: np.ndarray, k: int = 10, n_passes: int = 3) -> np.ndarray:
    """Measure per-query latency across multiple passes.

    Each query is timed individually. Returns the per-query median
    across passes, in nanoseconds.
    """
    import time

    n_queries = queries.shape[0]
    timings = np.zeros((n_passes, n_queries), dtype=np.int64)

    for pass_idx in range(n_passes):
        for q_idx in range(n_queries):
            query = queries[q_idx : q_idx + 1]
            t0 = time.perf_counter_ns()
            index.search(query, k)
            t1 = time.perf_counter_ns()
            timings[pass_idx, q_idx] = t1 - t0

    return np.median(timings, axis=0).astype(np.int64)


def measure_qps(
    index, queries: np.ndarray, k: int = 10,
    batch_size: int = 1000, n_passes: int = 3,
) -> float:
    """Measure throughput (queries per second) with batched search.

    Returns the median QPS across passes.
    """
    import time

    n_queries = queries.shape[0]
    qps_per_pass = []

    for _ in range(n_passes):
        t0 = time.perf_counter()
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            batch = np.ascontiguousarray(queries[start:end])
            index.search(batch, k)
        elapsed = time.perf_counter() - t0
        qps_per_pass.append(n_queries / elapsed)

    return float(np.median(qps_per_pass))
