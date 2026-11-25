"""Results schema: one row per run, saved as Parquet under ``results/raw/<experiment>/``.

The v4 plan (B4) fixes the measurement columns; v5 adds provenance columns so that no number can
be reported without knowing which faiss build, which host, and which host state produced it.
``base_row()`` fills the provenance part; experiment code fills the rest. Columns an experiment
does not use stay null (e.g. ``nprobe`` for HNSW). ``validate()`` rejects rows that miss a
required column, so a schema drift fails loudly instead of producing a half-empty table.
"""

import uuid
from datetime import UTC, datetime

# v4 B4 measurement columns
MEASUREMENT_COLUMNS = [
    "dataset", "D", "N", "order", "index", "M", "efC", "efSearch", "nprobe", "nlist", "m",
    "k_factor", "profile", "alpha", "threads", "core_type", "rep",
    "recall_at_10", "recall_at_100", "mrr_at_10", "qps", "lat_p50_us", "lat_p99_us",
    "bytes_per_vec", "rss_bytes", "build_s", "dco", "bytes_read", "seed",
]

# v5 provenance columns (always filled by base_row)
PROVENANCE_COLUMNS = [
    "run_id", "experiment", "timestamp", "git_commit", "hardware_id", "faiss_version",
    "faiss_compile_options", "faiss_path", "timed",
]

# host state columns (filled for timed rows from nestbench.host_guard.host_state)
HOST_COLUMNS = [
    "host_checked_at", "host_timing_ok", "host_problems", "host_vllm_running",
    "host_mem_available_gb", "host_swap_used_gb", "host_load1", "host_perf_freq_khz",
]

REQUIRED = PROVENANCE_COLUMNS + ["dataset", "D", "N", "index"]

_SNAPSHOT_CACHE: dict | None = None


def _snap() -> dict:
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is None:
        from nestbench.hardware import snapshot
        _SNAPSHOT_CACHE = snapshot()
    return _SNAPSHOT_CACHE


def base_row(experiment: str, timed: bool, host: dict | None = None, **fields) -> dict:
    """Provenance-filled row. A timed row must carry a host state that passed the guard."""
    s = _snap()
    row = {c: None for c in MEASUREMENT_COLUMNS + HOST_COLUMNS}
    row.update({
        "run_id": uuid.uuid4().hex, "experiment": experiment,
        "timestamp": datetime.now(UTC).isoformat(), "git_commit": s["git_commit"],
        "hardware_id": s["hardware_id"], "faiss_version": s["faiss_version"],
        "faiss_compile_options": s["faiss_compile_options"], "faiss_path": s["faiss_path"],
        "timed": bool(timed),
    })
    if timed and (host is None or not host.get("host_timing_ok")):
        raise ValueError("a timed row needs a host state that passed nestbench.host_guard")
    if host:
        row.update({k: host.get(k) for k in HOST_COLUMNS})
    row.update(fields)
    return row


def validate(rows: list[dict]) -> None:
    for r in rows:
        missing = [c for c in REQUIRED if r.get(c) is None]
        if missing:
            raise ValueError(f"row {r.get('run_id')} misses required columns {missing}")
        if r["timed"] and not r.get("host_timing_ok"):
            raise ValueError(f"timed row {r['run_id']} was not measured on a quiet host")
