"""Resident-memory measurement of a serialized index, in a fresh process.

Method from fewer_bits (annstudy.runner.run_memory_spec): RSS recorded after an in-process build
is inflated by the N x D input array, so memory is measured in a new interpreter: touch the
warm-up queries, record baseline RSS, ``faiss.read_index`` (no mmap, so the index is resident),
run the warm-up queries, record RSS again. The DELTA is the index's resident footprint. It is
reported next to the formula bytes/vector (v4 B6) in E3.
"""

import json
import subprocess
import sys

_CHILD = r"""
import json, sys
import numpy as np, psutil
spec = json.loads(sys.argv[1])
import faiss
faiss.omp_set_num_threads(1)
q = np.load(spec["queries_npy"], mmap_mode="r")[: spec.get("n_warmup", 200)]
q = np.ascontiguousarray(np.array(q, dtype=np.float32))
proc = psutil.Process()
base = proc.memory_info().rss
idx = faiss.read_index(spec["index_path"])
for p, v in spec.get("params", {}).items():
    obj = idx
    *path, last = p.split(".")
    for a in path:
        obj = getattr(obj, a)
    setattr(obj, last, v)
for i in range(len(q)):
    idx.search(q[i:i + 1], spec.get("k", 10))
after = proc.memory_info().rss
print(json.dumps({"rss_baseline_bytes": base, "rss_loaded_bytes": after,
                  "rss_bytes": max(0, after - base), "ntotal": int(idx.ntotal)}))
"""


def measure_index_rss(index_path: str, queries_npy: str, k: int = 10, n_warmup: int = 200,
                      params: dict | None = None, python: str | None = None) -> dict:
    """RSS delta of loading + warming ``index_path`` in a fresh interpreter.

    ``queries_npy`` is a .npy of float32 queries (memory-mapped, first ``n_warmup`` used).
    ``params`` sets attributes before the warm-up, e.g. ``{"hnsw.efSearch": 64}``.
    Returns rss_* fields plus ``bytes_per_vec_rss``.
    """
    spec = {"index_path": str(index_path), "queries_npy": str(queries_npy), "k": k,
            "n_warmup": n_warmup, "params": params or {}}
    out = subprocess.run([python or sys.executable, "-c", _CHILD, json.dumps(spec)],
                         capture_output=True, text=True, check=True)
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    rec["bytes_per_vec_rss"] = rec["rss_bytes"] / max(1, rec["ntotal"])
    return rec
