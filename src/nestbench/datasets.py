"""Benchmark-suite registry and loaders (``configs/datasets.yaml``).

``registry()`` returns every set (real, extras, and the expanded synthetic grids) as a dict of
spec dicts; ``Dataset(name)`` gives paths and memory-mapped loaders for a materialized set
under ``data/sets/<name>/``.
"""

import json
from functools import cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "datasets.yaml"
SETS_DIR = ROOT / "data" / "sets"


def _fmt(x) -> str:
    return f"{x:g}"


@cache
def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def registry() -> dict[str, dict]:
    cfg = _config()
    out = {}
    for name, spec in cfg["sets"].items():
        out[name] = {"name": name, **cfg["defaults"], **spec}
    s1 = cfg["synthetic"]["s1"]
    for D in s1["D"]:
        for b in s1["beta"]:
            name = f"s1-d{D}-b{_fmt(b)}"
            out[name] = {"name": name, **cfg["defaults"], "role": "S1", "family": "synthetic-gauss",
                         "source": "s1", "D": D, "N": s1["N"], "n_queries": s1["n_queries"],
                         "beta": float(b), "seed": s1["seed"] + 10 * D + int(10 * b),
                         "native_order": "none", "normalize": False}
    s2 = cfg["synthetic"]["s2"]
    for D in s2["D"]:
        for C in s2["C"]:
            for z in s2["zipf_s"]:
                name = f"s2-d{D}-c{C}-z{_fmt(z)}"
                out[name] = {"name": name, **cfg["defaults"], "role": "S2",
                             "family": "synthetic-mixture", "source": "s2", "D": D,
                             "N": s2["N"], "n_queries": s2["n_queries"], "C": C,
                             "zipf_s": float(z), "scale_range": s2["scale_range"],
                             "seed": s2["seed"] + 10 * D + C + int(z),
                             "native_order": "none", "normalize": False}
    return out


def names(role_prefix: str | None = None) -> list[str]:
    reg = registry()
    return [n for n, s in reg.items() if role_prefix is None or s["role"].startswith(role_prefix)]


class Dataset:
    """Paths and loaders for ``data/sets/<name>/``."""

    def __init__(self, name: str, root: Path | None = None):
        reg = registry()
        if name not in reg:
            raise KeyError(f"unknown dataset {name}")
        self.name, self.spec = name, reg[name]
        self.dir = Path(root or SETS_DIR) / name
        self.D, self.N = int(self.spec["D"]), int(self.spec["N"])

    def path(self, stem: str) -> Path:
        return self.dir / stem

    # -- arrays (memory-mapped; callers copy what they need)
    def base(self) -> np.ndarray:
        return np.load(self.path("base.npy"), mmap_mode="r")

    def queries(self) -> np.ndarray:
        return np.load(self.path("queries.npy"), mmap_mode="r")

    def gt(self) -> tuple[np.ndarray, np.ndarray]:
        g = np.load(self.path("gt.npz"))
        return g["ids"], g["dist"]

    def labels(self) -> dict:
        return dict(np.load(self.path("labels.npz")))

    def meta(self) -> dict:
        p = self.path("meta.json")
        return json.loads(p.read_text()) if p.exists() else {}

    def write_meta(self, **upd) -> dict:
        m = self.meta()
        m.update(upd)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path("meta.json").write_text(json.dumps(m, indent=1, default=float))
        return m

    def split(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Calibration / test query indices of resample ``i`` (0..n_splits-1)."""
        s = self.labels()["splits"][i].astype(bool)
        return np.flatnonzero(s), np.flatnonzero(~s)
