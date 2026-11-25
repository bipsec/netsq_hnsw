"""Stage 1 driver: materialize -> ground truth -> cross-checks -> labels -> premise check.

    python -m nestbench.stage1 --sets all        # or a list of names
    python -m nestbench.stage1 --manifest        # sha256 of every file

Every step is resumable: a step whose output exists (and is recorded in meta.json) is skipped.

**Premise check** (v5 Stage 1 gate). This runs before any NestSQ code exists and asks whether
early exit can work at all, using EXACT (unquantized) data and the dimension-only version of the
bound. For a query q, a candidate x, and a prefix P of p ordered axes (tail T = the rest):

    LB_p = |q_P - x_P|^2 + (|q_T| - |x_T|)^2  <=  |q - x|^2,

which follows from Cauchy-Schwarz on the tail. A candidate is rejected after p axes if
LB_p > tau, where tau is the ef-th smallest true distance (the HNSW level-0 threshold once the
result list is full). Two candidate pools bracket what HNSW actually sees:
- **hard**: true ranks ef+1 .. ef+500 (just outside the list; the hardest rejects);
- **random**: uniform base points (the easiest rejects).

Real HNSW distance computations fall between the two. Stage 2 measures them on actual graph
traces, with quantization.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from nestbench.datasets import ROOT, Dataset, registry

logger = logging.getLogger(__name__)

FEWER_BITS_GT = {
    "msm-a-384": "gt/gt_N500000_D384.npz", "msm-a-768": "gt/gt_N500000_D768.npz",
    "msm-a-1536": "gt/gt_N500000_D1536.npz", "msm-b-1024": "gt_m2/gt_N500000_D1024.npz",
}
PREMISE_SKIP = ("-10m", "-100k", "-1m", "-q10k")  # premise only on the main sets


def blocks(ds: Dataset) -> list[int]:
    """Block sizes (v4 A3; v5: Model A uses the D/8 rule since its MRL grains are undisclosed)."""
    D = ds.D
    if ds.spec.get("model") == "B" and D == 1024:
        return [64, 64, 128, 256, 512]
    return [D // 8, D // 8, D // 4, D - 2 * (D // 8) - D // 4]  # D/8 | D/8 | D/4 | rest (~D/2)


# ------------------------------------------------------------------------------ GT

def ground_truth(ds: Dataset) -> dict:
    from nestbench.gt import exact_knn

    if ds.path("gt.npz").exists() and "gt" in ds.meta():
        return ds.meta()["gt"]
    r = exact_knn(ds.base(), ds.queries(), k=ds.spec["k_gt"])
    np.savez(ds.path("gt.npz"), ids=r["ids"], dist=r["dist"])
    info = {k: r[k] for k in ("n_uncertified", "max_err_bound", "seconds", "min_cert_margin")}
    if ds.path("relq.npy").exists():
        rr = exact_knn(ds.base(), np.load(ds.path("relq.npy")), k=ds.spec["k_gt"])
        np.savez(ds.path("gt_relq.npz"), ids=rr["ids"], dist=rr["dist"])
        info["relq_n_uncertified"] = rr["n_uncertified"]
    ds.write_meta(gt=info)
    return info


def cross_check(ds: Dataset) -> dict | None:
    """Agreement of our GT with the provided GT (texmex, ann-benchmarks) or with the SIGIR
    study's CPU FAISS GT (MS MARCO)."""
    from nestbench.gt import agreement

    if "gt_check" in ds.meta():
        return ds.meta()["gt_check"]
    ids, dist = ds.gt()
    base, q = ds.base(), np.asarray(ds.queries())
    if ds.path("provided_gt.npy").exists():
        other, src = np.load(ds.path("provided_gt.npy")).astype(np.int64), "provided"
    elif ds.name in FEWER_BITS_GT:
        f = ROOT.parent / "fewer_bits" / "data" / FEWER_BITS_GT[ds.name]
        other, src = np.load(f)["ids"], f"fewer_bits {FEWER_BITS_GT[ds.name]}"
    else:
        return None
    out = {"source": src}
    for k in (10, 100):
        a = agreement(ids, other, dist, base, q, k=k)
        out.update({f"{m}@{k}": v for m, v in a.items()})
    ds.write_meta(gt_check=out)
    return out


# --------------------------------------------------------------------------- labels

def compute_labels(ds: Dataset) -> dict:
    from nestbench import labels as L

    m = ds.meta()
    if "labels" in m and ds.path("labels.npz").exists():
        return m["labels"]
    base = ds.base()
    mean, cov = L.covariance(base)
    eig, comps = L.pca(cov)
    rot = L.random_rotation(ds.D, ds.spec["rotation_seed"] + ds.D)
    var = {"native": L.axis_variance(cov, None), "pca": eig, "random": L.axis_variance(cov, rot)}
    np.savez(ds.path("pca.npz"), mean=mean, components=comps, eigvals=eig)
    np.savez(ds.path("rot.npz"), random=rot)
    np.savez(ds.path("spectrum.npz"), **{f"var_{k}": v for k, v in var.items()},
             **{f"tail_{k}": L.tail_curve(v) for k, v in var.items()})
    _, gdist = ds.gt()
    r10, quart = L.density(gdist, 10)
    nq = len(gdist)
    extra = {}
    if ds.spec["source"] in ("msmarco", "subsample") and nq == 10_000:
        sig = np.zeros(nq, dtype=np.int8)
        sig[:2000] = 1  # the SIGIR study's positional split: rows 0-1999 tuning
        extra["sigir_tuning"] = sig
    np.savez(ds.path("labels.npz"), density_r10=r10, density_quartile=quart,
             splits=L.splits(nq, ds.spec["n_splits"], ds.spec["split_seed"]), **extra)
    lid_mean, lid_std = L.lid_mle(base)
    bl = blocks(ds)
    cut = np.cumsum(bl)[:-1]
    info = {
        "C25": {k: L.c25(v) for k, v in var.items()},
        "C25_uncentred": {"native": L.second_moment_share(base, None, None),
                          "pca": L.second_moment_share(base, comps, mean),
                          "random": L.second_moment_share(base, rot, None)},
        "tail_at_block_ends": {k: [float(L.tail_curve(v)[c]) for c in cut]
                               for k, v in var.items()},
        "blocks": bl, "lid_mle_mean": lid_mean, "lid_mle_std": lid_std,
        "density_r10_quartiles": [float(x) for x in np.quantile(r10, [0.25, 0.5, 0.75])],
        "pca_top1_share": float(eig[0] / eig.sum()),
        "pca_dims_for_90pct": int(np.searchsorted(np.cumsum(eig) / eig.sum(), 0.9) + 1),
    }
    ds.write_meta(labels=info)
    return info


# -------------------------------------------------------------------------- premise

def premise(ds: Dataset, n_q: int = 200, pool: int = 500, efs=(10, 100), seed: int = 5) -> list:
    """Fraction of candidates rejected by the exact dimension-only bound after each prefix."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(seed)
    base = ds.base()
    qi = np.sort(rng.choice(len(ds.queries()), n_q, replace=False))
    q = torch.from_numpy(np.asarray(ds.queries()[qi], dtype=np.float32)).cuda()
    # full exact-enough distances for ranking (fp32; only ranks ef..ef+pool are used)
    dall = []
    for a in range(0, ds.N, 500_000):
        x = torch.from_numpy(np.ascontiguousarray(base[a:a + 500_000])).cuda()
        dall.append(((q * q).sum(1)[:, None] + (x * x).sum(1)[None] - 2 * q @ x.T).cpu())
    dall = torch.cat(dall, 1)
    srt = torch.sort(dall, 1)
    pca_ = np.load(ds.path("pca.npz"))
    rot = np.load(ds.path("rot.npz"))["random"]
    maps = {"native": (None, None), "pca": (pca_["components"], pca_["mean"]),
            "random": (rot, None)}
    D = ds.D
    grid = sorted(set(np.linspace(0, D, 17).astype(int).tolist()) |
                  set(np.cumsum(blocks(ds))[:-1].tolist()))
    rows = []
    for ef in efs:
        tau = srt.values[:, ef - 1].numpy().astype(np.float64)
        hard = srt.indices[:, ef:ef + pool].numpy()
        rnd = rng.integers(0, ds.N, size=(n_q, pool))
        for pool_name, ids in (("hard", hard), ("random", rnd)):
            for order, (R, mu) in maps.items():
                qq = np.asarray(ds.queries()[qi], dtype=np.float64)
                xx = np.asarray(base[np.sort(np.unique(ids))], dtype=np.float64)
                pos = np.searchsorted(np.sort(np.unique(ids)), ids)
                if mu is not None:
                    qq, xx = qq - mu, xx - mu
                if R is not None:
                    qq, xx = qq @ R.T, xx @ R.T
                X = torch.from_numpy(xx[pos]).cuda()          # n_q x pool x D
                Q = torch.from_numpy(qq).cuda()[:, None, :]
                diff2 = torch.cumsum((Q - X) ** 2, 2)
                qt = torch.flip(torch.cumsum(torch.flip(Q ** 2, [2]), 2), [2])
                xt = torch.flip(torch.cumsum(torch.flip(X ** 2, [2]), 2), [2])
                T = torch.from_numpy(tau).cuda()[:, None]
                for p in grid:
                    if p == 0:
                        lb = (qt[..., 0].sqrt() - xt[..., 0].sqrt()) ** 2
                    elif p == D:
                        lb = diff2[..., -1]
                    else:
                        lb = diff2[..., p - 1] + (qt[..., p].sqrt() - xt[..., p].sqrt()) ** 2
                    rows.append({"dataset": ds.name, "D": D, "ef": ef, "pool": pool_name,
                                 "order": order, "prefix": p, "prefix_frac": p / D,
                                 "rejected": float((lb > T).double().mean())})
                del X, Q, diff2, qt, xt
                torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------------------ main

def run(names: list[str], do_premise: bool = True) -> None:
    import pandas as pd

    from nestbench.prep import materialize

    out = ROOT / "results" / "raw" / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        logger.info(f"=== {n}")
        ds = materialize(n)
        ground_truth(ds)
        chk = cross_check(ds)
        if chk:
            logger.info(f"{n} GT check: {json.dumps(chk)}")
        info = compute_labels(ds)
        logger.info(f"{n} C25 {info['C25']}")
        f = out / f"premise_{n}.parquet"
        if do_premise and not n.endswith(PREMISE_SKIP) and not f.exists():
            pd.DataFrame(premise(ds)).to_parquet(f, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["all"])
    ap.add_argument("--no-premise", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.manifest:
        from nestbench.prep import manifest
        rec = manifest()
        print(f"manifest: {len(rec)} files")
        return
    names = list(registry()) if a.sets == ["all"] else a.sets
    run(names, not a.no_premise)


if __name__ == "__main__":
    main()
