"""Configure FAISS's instruction-set selection before faiss is imported.

The faiss loader probes for SIMD variants (``swigfaiss_avx512`` / ``swigfaiss_avx2``) based on
the CPU's capabilities. Some wheels — notably the PyPI ``faiss-cpu`` *Windows* wheels — ship only
a generic ``_swigfaiss`` with no AVX2/AVX512 binary, so on an AVX2-capable CPU the loader tries
and fails to import ``faiss.swigfaiss_avx2`` and logs a ``ModuleNotFoundError`` on every import
before falling back to generic. Pinning ``FAISS_OPT_LEVEL=generic`` skips that doomed probe.

Deciding that from *filenames alone* is wrong: a modern unified abi3 wheel (e.g. faiss-cpu 1.15.1
on Linux) also ships a single ``_swigfaiss.abi3.so``, yet that binary is compiled WITH AVX2
(``get_compile_options() == "OPTIMIZE DD AVX2"``). Pinning generic there would claim to disable
the SIMD path that the study's latency numbers depend on.

So the decision is made from what the binary reports, not from what files exist: probe
``faiss.get_compile_options()`` once in a throwaway subprocess (faiss must not be imported into
this process before the env var is set), and pin ``generic`` only when that report shows no SIMD.
The probe result is cached in the environment so spawned runners inherit it instead of re-probing.

Must be called BEFORE ``import faiss`` to take effect.
"""

import importlib.util
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

_SIMD_MARKERS = ("swigfaiss_avx512", "swigfaiss_avx2", "swigfaiss_sve")
_SIMD_COMPILE_FLAGS = ("AVX512", "AVX2", "NEON", "SVE")

#: Cache for the probe result, propagated to child processes via the environment.
COMPILE_OPTIONS_ENV = "NESTBENCH_FAISS_COMPILE_OPTIONS"


def faiss_has_simd_variant() -> bool:
    """True if the installed faiss package ships a separate SIMD (AVX2/AVX512/SVE) module."""
    spec = importlib.util.find_spec("faiss")  # locates without importing faiss itself
    if not spec or not spec.submodule_search_locations:
        return False
    for loc in spec.submodule_search_locations:
        try:
            for name in os.listdir(loc):
                if any(marker in name for marker in _SIMD_MARKERS):
                    return True
        except OSError:
            continue
    return False


def faiss_compile_options(timeout: int = 120) -> str:
    """Return ``faiss.get_compile_options()`` without importing faiss into this process.

    Cached in ``COMPILE_OPTIONS_ENV`` so repeated calls and child processes are free. Returns
    an empty string if the probe fails (callers then fall back to the filename heuristic).
    """
    cached = os.environ.get(COMPILE_OPTIONS_ENV)
    if cached is not None:
        return cached

    if "faiss" in sys.modules:  # already imported here; no need for a subprocess
        opts = str(sys.modules["faiss"].get_compile_options())
    else:
        env = {k: v for k, v in os.environ.items() if k != "FAISS_OPT_LEVEL"}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "import faiss; print(faiss.get_compile_options())"],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            opts = proc.stdout.strip() if proc.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug(f"faiss compile-option probe failed: {e}")
            opts = ""

    os.environ[COMPILE_OPTIONS_ENV] = opts
    return opts


def faiss_build_has_simd() -> bool:
    """True if the installed faiss *binary* reports a SIMD instruction set.

    Prefers the authoritative compile-option probe; falls back to the filename heuristic if the
    probe could not run.
    """
    opts = faiss_compile_options()
    if opts:
        return any(flag in opts.upper() for flag in _SIMD_COMPILE_FLAGS)
    return faiss_has_simd_variant()


def ensure_opt_level() -> str | None:
    """Pin FAISS_OPT_LEVEL=generic iff the installed faiss binary has no SIMD build.

    Respects an explicit FAISS_OPT_LEVEL already set by the user. Idempotent.
    """
    if os.environ.get("FAISS_OPT_LEVEL"):
        return os.environ["FAISS_OPT_LEVEL"]
    if not faiss_build_has_simd():
        os.environ["FAISS_OPT_LEVEL"] = "generic"
        logger.info("faiss has no SIMD build; pinned FAISS_OPT_LEVEL=generic")
    return os.environ.get("FAISS_OPT_LEVEL")
