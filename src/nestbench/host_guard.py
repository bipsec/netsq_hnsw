"""Host guard: is this machine quiet enough for a timed (latency/QPS) run?

The DGX Spark's CPU and GPU share one LPDDR5x memory system (about 273 GB/s), and the machine
normally hosts a resident vLLM server. The SIGIR study measured on it while vLLM ran and had to
disclaim latency (between-session drift ~1.36x, canary CV 0.15). This paper's main claim is a QPS
claim, so plan v5 requires: **timed runs only with vLLM stopped**. Untimed work (ground truth,
builds, PCA) may run at any time.

``host_state()`` returns a dict that every timed results row carries (``host_*`` columns), and
``check_timing()`` raises if the host is not fit for timing. The checks:

1. No GPU compute process other than whitelisted desktop helpers (``nvidia-smi``), and no
   process whose command line mentions vLLM.
2. Available RAM at least ``min_free_gb``.
3. Every performance core uses the ``performance`` governor and reports its max frequency.
4. 1-minute load average below ``max_load`` (other CPU work steals bandwidth and cores).
"""

import json
import os
import subprocess
from datetime import UTC, datetime

# GPU processes that hold a small allocation but do no compute (remote desktop on GB10).
GPU_WHITELIST = ("gnome-remote-desktop",)


def gpu_compute_apps() -> list[dict]:
    """GPU compute processes from nvidia-smi ([] if nvidia-smi is unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    apps = []
    for ln in out.stdout.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 3:
            mem = int(parts[2]) if parts[2].isdigit() else -1
            apps.append({"pid": int(parts[0]), "name": parts[1], "used_mib": mem})
    return apps


def vllm_pids() -> list[int]:
    """PIDs whose command line mentions vllm (case-insensitive), excluding this process."""
    import psutil

    me = os.getpid()
    pids = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            text = " ".join([p.info["name"] or ""] + (p.info["cmdline"] or [])).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "vllm" in text and p.info["pid"] != me and "host_guard" not in text:
            pids.append(p.info["pid"])
    return pids


def host_state(min_free_gb: float = 40.0, max_load: float = 2.0) -> dict:
    """Snapshot of everything that can invalidate a timing; ``timing_ok`` summarises it."""
    import psutil

    from nestbench.hardware import core_map, performance_cores

    apps = gpu_compute_apps()
    busy_gpu = [a for a in apps if not any(w in a["name"] for w in GPU_WHITELIST)]
    vpids = vllm_pids()
    mem = psutil.virtual_memory()
    load1 = os.getloadavg()[0]
    cores = core_map()
    perf = set(performance_cores(cores))
    perf_cores = [c for c in cores if c["cpu"] in perf]
    freq_ok = all(c["governor"] == "performance" and c["cur_khz"] == c["max_khz"]
                  for c in perf_cores)

    problems = []
    if busy_gpu:
        listed = ", ".join(f"{a['name']}({a['pid']},{a['used_mib']}MiB)" for a in busy_gpu)
        problems.append(f"GPU compute processes: {listed}")
    if vpids:
        problems.append(f"vLLM processes running: {vpids}")
    if mem.available / 2**30 < min_free_gb:
        problems.append(f"available RAM {mem.available / 2**30:.1f} GB < {min_free_gb} GB")
    if not freq_ok:
        problems.append("performance cores not at max frequency / performance governor")
    if load1 > max_load:
        problems.append(f"load average {load1:.2f} > {max_load}")

    return {
        "host_checked_at": datetime.now(UTC).isoformat(),
        "host_timing_ok": not problems,
        "host_problems": "; ".join(problems),
        "host_vllm_running": bool(vpids) or any("vllm" in a["name"].lower() for a in apps),
        "host_gpu_apps": json.dumps(apps),
        "host_mem_available_gb": round(mem.available / 2**30, 1),
        "host_swap_used_gb": round(psutil.swap_memory().used / 2**30, 2),
        "host_load1": round(load1, 2),
        "host_perf_cores": json.dumps(sorted(perf)),
        "host_perf_freq_khz": json.dumps([c["cur_khz"] for c in perf_cores]),
    }


class HostNotQuiet(RuntimeError):
    pass


def check_timing(**kw) -> dict:
    """Return the host state, or raise ``HostNotQuiet`` if a timed run must not start."""
    st = host_state(**kw)
    if not st["host_timing_ok"]:
        raise HostNotQuiet(st["host_problems"])
    return st


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Check whether the host is fit for timed runs.")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    ap.add_argument("--max-load", type=float, default=2.0)
    ap.add_argument("--require", action="store_true",
                    help="exit 1 if the host is not fit for timing (use before a timed campaign)")
    a = ap.parse_args()
    st = host_state(a.min_free_gb, a.max_load)
    print(json.dumps(st, indent=2))
    if a.require and not st["host_timing_ok"]:
        print(f"\nNOT FIT FOR TIMING: {st['host_problems']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
