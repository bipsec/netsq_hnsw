"""Stage 1: GT exactness and certificate, format readers, labels, registry, premise bound."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _brute(base, q, k):
    d = ((q[:, None, :].astype(np.float64) - base[None].astype(np.float64)) ** 2).sum(2)
    ids = np.stack([np.lexsort((np.arange(len(base)), r))[:k] for r in d])
    return ids, np.take_along_axis(d, ids, 1)


@cuda
@pytest.mark.parametrize("scale", [1.0, 300.0])  # unit-like and SIFT-like magnitudes
def test_exact_knn_matches_float64_bruteforce(scale):
    from nestbench.gt import exact_knn

    rng = np.random.default_rng(0)
    base = (rng.standard_normal((5000, 48)) * scale).astype(np.float32)
    q = (rng.standard_normal((64, 48)) * scale).astype(np.float32)
    r = exact_knn(base, q, k=20, extra=16, base_chunk=1700, q_chunk=20)
    ids, d = _brute(base, q, 20)
    assert r["n_uncertified"] == 0
    np.testing.assert_array_equal(r["ids"], ids)
    np.testing.assert_allclose(r["dist"], d, rtol=1e-6)


@cuda
def test_uncertified_queries_fall_back_to_exact():
    from nestbench.gt import exact_knn

    # many exact duplicates make the (k+extra)-th distance equal the k-th: certificate fails
    base = np.repeat(np.eye(8, dtype=np.float32), 50, axis=0)
    q = np.eye(8, dtype=np.float32)[:2] * 0.9
    r = exact_knn(base, q, k=10, extra=4)
    assert r["n_uncertified"] == 2 and r["n_uncertified_fp64"] == 2
    ids, d = _brute(base, q, 10)
    np.testing.assert_array_equal(r["ids"], ids)


@cuda
def test_fp64_pass_certifies_what_fp32_cannot():
    from nestbench.gt import exact_knn

    # a large common offset inflates the norms (so the fp32 error bound) but not the distances
    rng = np.random.default_rng(3)
    base = (rng.standard_normal((4000, 32)) + 1e3).astype(np.float32)
    q = (rng.standard_normal((16, 32)) + 1e3).astype(np.float32)
    r = exact_knn(base, q, k=10, extra=1)
    assert r["n_uncertified"] > 0 and r["n_uncertified_fp64"] == 0
    ids, d = _brute(base, q, 10)
    np.testing.assert_array_equal(r["ids"], ids)


def test_vecs_readers_roundtrip_and_reject_bad(tmp_path):
    from nestbench.io_vecs import read_fbin, read_fvecs, read_ivecs

    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    rec = np.hstack([np.full((3, 1), 4, np.int32).view(np.float32), x])
    rec.tofile(tmp_path / "a.fvecs")
    np.testing.assert_array_equal(read_fvecs(tmp_path / "a.fvecs"), x)
    iv = np.hstack([np.full((2, 1), 3, np.int32), np.arange(6, dtype=np.int32).reshape(2, 3)])
    iv.tofile(tmp_path / "b.ivecs")
    assert read_ivecs(tmp_path / "b.ivecs").tolist() == [[0, 1, 2], [3, 4, 5]]
    (tmp_path / "c.fvecs").write_bytes((tmp_path / "a.fvecs").read_bytes()[:-4])
    with pytest.raises(ValueError):
        read_fvecs(tmp_path / "c.fvecs")
    # range-downloaded fbin: header says 1000 rows, body has 3
    with open(tmp_path / "d.fbin", "wb") as f:
        np.array([1000, 4], np.int32).tofile(f)
        x.tofile(f)
    np.testing.assert_array_equal(read_fbin(tmp_path / "d.fbin", n_max=3), x)
    with pytest.raises(ValueError):
        read_fbin(tmp_path / "d.fbin", n_max=4)


def test_tail_curve_c25_and_rotations():
    from nestbench.labels import axis_variance, c25, pca, random_rotation, tail_curve

    var = np.array([4.0, 2.0, 1.0, 1.0])
    t = tail_curve(var)
    assert t[0] == 1 and t[-1] == 0 and t[1] == pytest.approx(0.5)
    assert c25(var) == pytest.approx(0.5)
    R = random_rotation(16, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(16), atol=1e-12)
    rng = np.random.default_rng(1)
    A = rng.standard_normal((16, 16))
    cov = A @ A.T
    eig, comps = pca(cov)
    assert np.all(np.diff(eig) <= 1e-9)
    np.testing.assert_allclose(axis_variance(cov, comps), eig, rtol=1e-9)
    # an orthogonal map preserves total variance
    assert axis_variance(cov, R).sum() == pytest.approx(np.trace(cov))


def test_registry_expands_synthetic_grids_and_blocks():
    from nestbench.datasets import Dataset, registry
    from nestbench.stage1 import blocks

    reg = registry()
    assert sum(s["role"] == "S1" for s in reg.values()) == 12
    assert sum(s["role"] == "S2" for s in reg.values()) == 8
    assert len({s["seed"] for s in reg.values() if s["role"] in ("S1", "S2")}) == 20
    assert blocks(Dataset("msm-b-1024")) == [64, 64, 128, 256, 512]
    for n, want in (("gist-960", [120, 120, 240, 480]), ("glove-100", [12, 12, 25, 51]),
                    ("msm-a-384", [48, 48, 96, 192])):
        assert blocks(Dataset(n)) == want and sum(want) == Dataset(n).D


def test_prefix_bound_is_a_lower_bound():
    """LB_p = |q_P - x_P|^2 + (|q_T| - |x_T|)^2 never exceeds the true squared distance."""
    rng = np.random.default_rng(2)
    q, X = rng.standard_normal(64), rng.standard_normal((500, 64))
    true = ((X - q) ** 2).sum(1)
    for p in range(65):
        lb = ((X[:, :p] - q[:p]) ** 2).sum(1) + (np.linalg.norm(q[p:]) -
                                                  np.linalg.norm(X[:, p:], axis=1)) ** 2
        assert np.all(lb <= true + 1e-9)
