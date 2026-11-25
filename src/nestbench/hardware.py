"""Hardware and environment snapshot for reproducibility.

Adapted from fewer_bits (annstudy.hardware). Additions for this paper:

- the faiss build in use (``faiss.get_compile_options()`` and the module path), because timed
  results must come from our fork build from Stage 4 on and must never be mixed with the wheel;
- a per-core map (max frequency, current frequency, governor), because the DGX Spark mixes
  Cortex-X925 performance cores with Cortex-A725 efficiency cores and all timed runs are pinned to
  the performance cores (plan v5, B5).

``hardware_id`` hashes (CPU model names, physical cores, total RAM). The CPU model here joins every
``lscpu`` model name ("Cortex-X925 + Cortex-A725"), so ids are NOT comparable with fewer_bits'
(which read /proc/cpuinfo, empty on Arm). That is intended: v4 B5 forbids mixing numbers across
the two studies. Paths inside the repository are stored relative to it (the repo is public).
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CPU_SYSFS = Path("/sys/devices/system/cpu")


def _get_git_commit(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5, cwd=cwd)
        if result.returncode == 0:
            sha = result.stdout.strip()
            dirty = subprocess.run(["git", "diff", "--quiet"], capture_output=True, timeout=5,
                                   cwd=cwd)
            return sha if dirty.returncode == 0 else f"{sha}-dirty"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "no-repo"


def _get_package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not-installed"


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def core_map() -> list[dict]:
    """One dict per logical CPU: max/current kHz and governor (empty list if sysfs is absent)."""
    cores = []
    for d in sorted(CPU_SYSFS.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        f = d / "cpufreq"
        mx, cur = _read(f / "cpuinfo_max_freq"), _read(f / "scaling_cur_freq")
        cores.append({"cpu": int(d.name[3:]),
                      "max_khz": int(mx) if mx else None,
                      "cur_khz": int(cur) if cur else None,
                      "governor": _read(f / "scaling_governor")})
    return cores


def performance_cores(cores: list[dict] | None = None) -> list[int]:
    """CPUs whose max frequency equals the highest on the machine (the X925 cores on GB10).

    On a homogeneous machine this is every CPU.
    """
    cores = core_map() if cores is None else cores
    mx = max((c["max_khz"] or 0) for c in cores) if cores else 0
    return [c["cpu"] for c in cores if (c["max_khz"] or 0) == mx]


REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def faiss_build() -> dict:
    """Which faiss is importable here: version, compile options, and the module path."""
    try:
        import faiss
        return {"faiss_version": faiss.__version__,
                "faiss_compile_options": str(faiss.get_compile_options()).strip(),
                "faiss_path": _repo_relative(Path(faiss.__file__).parent)}
    except Exception as e:  # pragma: no cover - only when faiss is broken
        return {"faiss_version": "unavailable", "faiss_compile_options": "",
                "faiss_path": f"import failed: {e}"}


def snapshot() -> dict:
    """Capture a reproducible snapshot of the execution environment."""
    import psutil

    cpu_models = []
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        cpu_models = [ln.split(":", 1)[1].strip() for ln in out.splitlines()
                      if ln.strip().startswith("Model name")]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    cpu_model = " + ".join(cpu_models) or platform.processor() or "unknown"

    mem = psutil.virtual_memory()
    physical_cores = psutil.cpu_count(logical=False) or 0
    logical_cores = psutil.cpu_count(logical=True) or 0
    cores = core_map()

    packages = ["faiss-cpu", "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
                "matplotlib", "psutil", "torch", "h5py", "threadpoolctl"]

    snap = {
        "timestamp": datetime.now(UTC).isoformat(),
        "cpu_model": cpu_model,
        "machine": platform.machine(),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "performance_cores": performance_cores(cores),
        "core_map": cores,
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version,
        "git_commit": _get_git_commit(),
        "hostname": platform.node(),
        "library_versions": {pkg: _get_package_version(pkg) for pkg in packages},
        **faiss_build(),
    }
    hw_key = f"{cpu_model}|{physical_cores}|{int(mem.total)}"
    snap["hardware_id"] = hashlib.sha256(hw_key.encode()).hexdigest()[:16]
    return snap


def save_snapshot(path: str = "results/hardware.json") -> dict:
    snap = snapshot()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(snap, indent=2))
    return snap


def done_on_this_host(path) -> bool:
    """True only if ``path`` exists AND every row was written on this machine.

    Latency is a host property, so a resumable runner must redo jobs measured elsewhere
    (results travel between this host and the x86 VM).
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return False
    try:
        hw = pd.read_parquet(p, columns=["hardware_id"])["hardware_id"]
    except Exception:
        return False
    return bool(len(hw)) and bool((hw == snapshot()["hardware_id"]).all())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Snapshot hardware and library versions.")
    ap.add_argument("--output", default="results/hardware.json")
    a = ap.parse_args()
    snap = save_snapshot(a.output)
    print(json.dumps({k: v for k, v in snap.items() if k != "core_map"}, indent=2))
    print(f"\nWrote {a.output} (hardware_id={snap['hardware_id']})")


if __name__ == "__main__":
    main()
