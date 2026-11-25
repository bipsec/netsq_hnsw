"""Tests for metrics module."""

import numpy as np
import pytest


def test_flat_recall_is_one(toy_vectors, toy_queries):
    """Flat index should achieve perfect recall."""
    import faiss

    from nestbench.metrics import recall_at_k

    dim = toy_vectors.shape[1]
    k = 10

    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(toy_vectors))

    gt_dists, gt_ids = index.search(toy_queries, k)
    ret_dists, ret_ids = index.search(toy_queries, k)

    recalls = recall_at_k(
        ret_ids, gt_ids, gt_dists,
        toy_vectors, toy_queries, k=k,
    )
    assert np.all(recalls == 1.0), f"Flat recall should be 1.0, got {recalls}"


def test_recall_handles_ties():
    """Recall should handle tied distances correctly."""
    from nestbench.metrics import recall_at_k

    passages = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float32)

    queries = np.array([[0.0, 0.0]], dtype=np.float32)

    gt_ids = np.array([[0, 1, 2, 3]], dtype=np.int64)
    gt_distances = np.array([[0.0, 1.0, 1.0, 1.0]], dtype=np.float32)

    retrieved_ids = np.array([[0, 4, 2, 3]], dtype=np.int64)

    recalls = recall_at_k(
        retrieved_ids, gt_ids, gt_distances,
        passages, queries, k=4,
    )
    assert recalls[0] == 1.0, (
        f"All 4 retrieved items are within the k-th GT distance, recall should be 1.0, "
        f"got {recalls[0]}"
    )


def test_bootstrap_ci_covers_mean():
    """Bootstrap CI should contain the sample mean."""
    from nestbench.metrics import bootstrap_ci

    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=1000)

    point, lo, hi = bootstrap_ci(values, np.mean, n_resamples=1000, seed=0)
    assert lo <= point <= hi, f"CI [{lo}, {hi}] doesn't contain point {point}"
    assert lo < hi, "CI should have nonzero width"


def test_latency_stats_plausible():
    """Latency stats should return reasonable values."""
    from nestbench.metrics import latency_stats

    ns_values = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
    stats = latency_stats(ns_values)

    assert stats["lat_mean_us"] == pytest.approx(3.0, abs=0.1)
    assert stats["lat_p50_us"] == pytest.approx(3.0, abs=0.1)
    assert stats["lat_p95_us"] > stats["lat_p50_us"]
    assert stats["lat_p99_us"] >= stats["lat_p95_us"]


def test_measure_latency_returns_plausible_values(toy_vectors, toy_queries):
    """Measured latency should be positive and reasonable."""
    import faiss

    from nestbench.metrics import measure_latency

    dim = toy_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(toy_vectors))

    per_query_ns = measure_latency(index, toy_queries, k=5, n_passes=2)
    assert per_query_ns.shape == (toy_queries.shape[0],)
    assert np.all(per_query_ns > 0), "All latencies should be positive"
    assert np.all(per_query_ns < 1e9), "No query should take more than 1 second"


