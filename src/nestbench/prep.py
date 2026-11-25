"""Stage 1: materialize every benchmark set under data/sets/<name>/ (base.npy, queries.npy).

Sources:
- ``msmarco``: copied from fewer_bits (read-only), with the SIGIR study's exact files. Model A is
  OpenAI text-embedding-3-large: the 384/768/1536 files are reused as-is, and 1024 is the first
  1024 columns of the (unit-norm) 1536 file, L2-renormalized. That is the same operation
  fewer_bits used for its own prefixes (``truncate_and_renorm``). Model B is
  Qwen3-Embedding-0.6B at 1024. The first N rows are taken (the 500K corpus of the SIGIR study,
  or all 1M for Model A's 1M variant). MS MARCO side files (query split, relevance set, qrels,
  ids) go to data/msmarco/.
- ``sift``/``gist``: texmex fvecs; ``glove``: ann-benchmarks hdf5 (angular, so L2-normalized
  here); ``deep``: the first 1M / 10M vectors of Deep1B (range-downloaded base.1B.fbin) with
  the 10K public queries.
- ``gist_learnq``: 10K extra GIST queries drawn (seeded) from gist_learn, which is disjoint from
  the base set. The base is a symlink to gist-960's.
- ``subsample``: a seeded random subset of a parent set's base; queries are the parent's.
- ``s1``/``s2``: synthetic, generated on the GPU (see ``synth_s1``/``synth_s2``).
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from nestbench.datasets import ROOT, Dataset

logger = logging.getLogger(__name__)

FEWER_BITS = ROOT.parent / "fewer_bits" / "data"
RAW = ROOT / "data" / "raw"
MSMARCO_DIR = ROOT / "data" / "msmarco"
EMB_DIR = {"A": FEWER_BITS / "emb", "B": FEWER_BITS / "emb_m2"}
EMB_N = {"A": 1_000_000, "B": 500_000}
EMB_NQ, EMB_NRELQ = 10_000, 5_000


# ----------------------------------------------------------------------------- helpers

def write_npy(path: Path, src, rows: int | None = None, chunk: int = 100_000,
              renorm_to: int | None = None) -> None:
    """Stream ``src[:rows, :renorm_to]`` into a float32 .npy; renormalize rows if renorm_to."""
    rows = len(src) if rows is None else rows
    D = src.shape[1] if renorm_to is None else renorm_to
    tmp = path.with_suffix(".tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(rows, D))
    for a in range(0, rows, chunk):
        x = np.asarray(src[a:min(a + chunk, rows), :D], dtype=np.float32)
        if renorm_to is not None:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
        out[a:a + len(x)] = x
    out.flush()
    del out
    os.replace(tmp, path)


def normalize_rows(x: np.ndarray) -> tuple[np.ndarray, int]:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    zero = int((n[:, 0] == 0).sum())
    return (x / np.where(n == 0, 1, n)).astype(np.float32), zero


def sha256(path: Path, buf: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def _emb(model: str, stem: str, D: int, n: int) -> np.memmap:
    return np.memmap(EMB_DIR[model] / f"{stem}_D{D}.f32", dtype=np.float32, mode="r",
                     shape=(n, D))


# --------------------------------------------------------------------------- sources

def prep_msmarco(ds: Dataset) -> dict:
    m, D, N = ds.spec["model"], ds.D, ds.N
    have = {"A": (384, 768, 1536), "B": (1024,)}[m]
    src_D = D if D in have else 1536
    renorm = None if D in have else D
    for stem, n, out in (("passages", EMB_N[m], "base.npy"), ("queries", EMB_NQ, "queries.npy"),
                         ("relq", EMB_NRELQ, "relq.npy")):
        src = _emb(m, stem, src_D, n)
        write_npy(ds.path(out), src, rows=N if stem == "passages" else None, renorm_to=renorm)
    MSMARCO_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("query_split.json", "relevance_queries.json", "passage_ids.npy", "query_ids.npy"):
        if not (MSMARCO_DIR / f).exists():
            shutil.copy2(FEWER_BITS / f, MSMARCO_DIR / f)
    if not (MSMARCO_DIR / "qrels.train.tsv").exists():
        shutil.copy2(FEWER_BITS / "msmarco_raw" / "qrels.train.tsv", MSMARCO_DIR)
    return {"source_files": f"fewer_bits/data/{EMB_DIR[m].name}/*_D{src_D}.f32",
            "derivation": "copy" if renorm is None else f"first {D} of D{src_D}, L2-renorm",
            "model": {"A": "openai/text-embedding-3-large",
                      "B": "Qwen/Qwen3-Embedding-0.6B"}[m]}


def prep_texmex(ds: Dataset, which: str) -> dict:
    from nestbench.io_vecs import read_fvecs, read_ivecs

    d = RAW / which / which
    base = read_fvecs(d / f"{which}_base.fvecs")
    q = read_fvecs(d / f"{which}_query.fvecs")
    assert base.shape == (ds.N, ds.D), base.shape
    np.save(ds.path("base.npy"), base)
    np.save(ds.path("queries.npy"), q)
    np.save(ds.path("provided_gt.npy"), read_ivecs(d / f"{which}_groundtruth.ivecs"))
    return {"source_files": f"texmex {which}.tar.gz ({which}_base/query/groundtruth)"}


def prep_gist_learnq(ds: Dataset) -> dict:
    from nestbench.io_vecs import read_fvecs

    learn = read_fvecs(RAW / "gist" / "gist" / "gist_learn.fvecs")
    rng = np.random.default_rng(ds.spec["query_seed"])
    idx = np.sort(rng.choice(len(learn), ds.spec["n_queries"], replace=False))
    np.save(ds.path("queries.npy"), learn[idx])
    np.save(ds.path("query_learn_rows.npy"), idx)
    parent = Dataset(ds.spec["base_from"])
    link = ds.path("base.npy")
    if not link.exists():
        link.symlink_to(os.path.relpath(parent.path("base.npy"), ds.dir))
    return {"source_files": "gist_learn.fvecs (queries), gist-960 base (symlink)",
            "n_learn": int(len(learn))}


def prep_glove(ds: Dataset) -> dict:
    import h5py

    with h5py.File(RAW / "glove" / "glove-100-angular.hdf5", "r") as f:
        train, test = np.asarray(f["train"]), np.asarray(f["test"])
        neighbors = np.asarray(f["neighbors"])
    assert train.shape == (ds.N, ds.D), train.shape
    base, zb = normalize_rows(train)
    q, zq = normalize_rows(test)
    np.save(ds.path("base.npy"), base)
    np.save(ds.path("queries.npy"), q)
    np.save(ds.path("provided_gt.npy"), neighbors)
    return {"source_files": "ann-benchmarks glove-100-angular.hdf5 (train/test/neighbors)",
            "zero_norm_base": zb, "zero_norm_queries": zq,
            "derivation": "L2-normalized (angular -> L2); zero vectors kept as zero"}


def prep_deep(ds: Dataset) -> dict:
    from nestbench.io_vecs import read_fbin

    base = read_fbin(RAW / "deep" / "base.10M.partial.fbin", n_max=ds.N)
    write_npy(ds.path("base.npy"), base)
    np.save(ds.path("queries.npy"), np.asarray(read_fbin(RAW / "deep" / "query.public.10K.fbin")))
    return {"source_files": "Deep1B base.1B.fbin (first rows, HTTP range), query.public.10K.fbin",
            "derivation": f"first {ds.N} vectors of Deep1B"}


def prep_subsample(ds: Dataset) -> dict:
    parent = Dataset(ds.spec["parent"])
    rng = np.random.default_rng(ds.spec["subsample_seed"])
    idx = np.sort(rng.choice(parent.N, ds.N, replace=False))
    np.save(ds.path("base.npy"), np.asarray(parent.base()[idx]))
    shutil.copy2(parent.path("queries.npy"), ds.path("queries.npy"))
    if parent.path("relq.npy").exists():
        shutil.copy2(parent.path("relq.npy"), ds.path("relq.npy"))
    np.save(ds.path("subsample_rows.npy"), idx)
    return {"source_files": f"{parent.name} base rows (seeded subsample), {parent.name} queries"}


# ------------------------------------------------------------------------- synthetic

def _gen_chunks(n: int, D: int, seed: int, make, chunk: int = 100_000):
    import torch

    g = torch.Generator(device="cuda").manual_seed(seed)
    for a in range(0, n, chunk):
        yield a, make(min(chunk, n - a), g)


def synth_s1(ds: Dataset) -> dict:
    """x = R (sqrt(lambda) * z), z ~ N(0, I), lambda_i ∝ i^-beta normalized to sum D, R a
    seeded Haar rotation. beta = 0 is isotropic (no order can compact energy)."""
    import torch

    from nestbench.labels import random_rotation

    D, beta, seed = ds.D, ds.spec["beta"], ds.spec["seed"]
    lam = np.arange(1, D + 1, dtype=np.float64) ** (-beta)
    lam *= D / lam.sum()
    R = torch.from_numpy(random_rotation(D, seed).astype(np.float32)).cuda()
    s = torch.from_numpy(np.sqrt(lam).astype(np.float32)).cuda()

    def make(n, g):
        return ((torch.randn(n, D, generator=g, device="cuda") * s) @ R).cpu().numpy()

    for stem, n, sd in (("base.npy", ds.N, seed), ("queries.npy", ds.spec["n_queries"], seed + 1)):
        tmp = ds.path(stem).with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, D))
        for a, x in _gen_chunks(n, D, sd, make):
            out[a:a + len(x)] = x
        out.flush()
        del out
        os.replace(tmp, ds.path(stem))
    np.save(ds.path("generator_lambda.npy"), lam)
    return {"generator": "s1 Gaussian, lambda_i ∝ i^-beta (sum D), Haar rotation",
            "beta": beta, "seed": seed}


def synth_s2(ds: Dataset) -> dict:
    """Gaussian mixture. Centres c_j = R (sqrt(lambda) * g_j) with lambda_i ∝ i^-1 (sum D);
    cluster weights w_j ∝ j^-s (Zipf; s = 0 uniform); point = c_j + r_j * z, with r_j
    log-uniform in scale_range (relative to the unit mean per-axis centre std). Queries are
    drawn from the same mixture."""
    import torch

    from nestbench.labels import random_rotation

    D, C, s, seed = ds.D, ds.spec["C"], ds.spec["zipf_s"], ds.spec["seed"]
    lo, hi = ds.spec["scale_range"]
    rng = np.random.default_rng(seed)
    lam = 1.0 / np.arange(1, D + 1, dtype=np.float64)
    lam *= D / lam.sum()
    R = random_rotation(D, seed)
    centres = (rng.standard_normal((C, D)) * np.sqrt(lam)) @ R
    w = np.arange(1, C + 1, dtype=np.float64) ** (-s)
    w /= w.sum()
    r = np.exp(rng.uniform(np.log(lo), np.log(hi), C))
    Ct = torch.from_numpy(centres.astype(np.float32)).cuda()
    rt = torch.from_numpy(r.astype(np.float32)).cuda()
    wt = torch.from_numpy(w).cuda()

    def make_fn(labels_out):
        def make(n, g):
            lab = torch.multinomial(wt, n, replacement=True, generator=g)
            x = Ct[lab] + rt[lab, None] * torch.randn(n, D, generator=g, device="cuda")
            labels_out.append(lab.cpu().numpy().astype(np.int32))
            return x.cpu().numpy()
        return make

    for stem, n, sd in (("base", ds.N, seed), ("queries", ds.spec["n_queries"], seed + 1)):
        labs: list = []
        tmp = ds.path(f"{stem}.npy").with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, D))
        for a, x in _gen_chunks(n, D, sd, make_fn(labs)):
            out[a:a + len(x)] = x
        out.flush()
        del out
        os.replace(tmp, ds.path(f"{stem}.npy"))
        np.save(ds.path(f"{stem}_cluster.npy"), np.concatenate(labs))
    np.savez(ds.path("generator.npz"), centres=centres.astype(np.float32), weights=w, scales=r,
             lam=lam)
    return {"generator": "s2 Gaussian mixture (centres lambda ∝ 1/i, Zipf weights, "
                         "log-uniform isotropic scales)", "C": C, "zipf_s": s, "seed": seed}


SOURCES = {
    "msmarco": prep_msmarco, "sift": lambda d: prep_texmex(d, "sift"),
    "gist": lambda d: prep_texmex(d, "gist"), "gist_learnq": prep_gist_learnq,
    "glove": prep_glove, "deep": prep_deep, "subsample": prep_subsample,
    "s1": synth_s1, "s2": synth_s2,
}


def materialize(name: str, force: bool = False) -> Dataset:
    ds = Dataset(name)
    if not force and ds.meta().get("materialized"):
        return ds
    ds.dir.mkdir(parents=True, exist_ok=True)
    info = SOURCES[ds.spec["source"]](ds)
    base, q = ds.base(), ds.queries()
    assert base.shape == (ds.N, ds.D), (name, base.shape)
    assert q.shape[1] == ds.D
    nb = np.linalg.norm(np.asarray(base[: min(ds.N, 100_000)], dtype=np.float64), axis=1)
    ds.write_meta(materialized=True, spec=ds.spec, nq=int(len(q)), **info,
                  base_norm_mean=float(nb.mean()), base_norm_min=float(nb.min()),
                  base_norm_max=float(nb.max()),
                  has_nan=bool(np.isnan(np.asarray(base[:100_000])).any()))
    logger.info(f"materialized {name}: base {base.shape}, queries {q.shape}")
    return ds


def manifest(out: Path = ROOT / "data" / "MANIFEST.json") -> dict:
    """sha256 + size of every file under data/sets and data/msmarco (symlinks recorded as such)."""
    rec = {}
    for top in (ROOT / "data" / "sets", MSMARCO_DIR):
        for p in sorted(top.rglob("*")):
            if p.is_dir():
                continue
            rel = str(p.relative_to(ROOT / "data"))
            if p.is_symlink():
                rec[rel] = {"symlink": os.readlink(p)}
            else:
                rec[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    out.write_text(json.dumps(rec, indent=1))
    return rec
