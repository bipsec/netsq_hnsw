"""Stage 0 infrastructure: host guard, results schema, memory measurement, stats, Pareto."""

import numpy as np
import pandas as pd
import pytest

# ------------------------------------------------------------------------- host guard

def _fake_state(monkeypatch, apps, vllm, avail_gb=100.0, load=0.1, freq_ok=True):
    import nestbench.hardware as hw
    import nestbench.host_guard as hg

    monkeypatch.setattr(hg, "gpu_compute_apps", lambda: apps)
    monkeypatch.setattr(hg, "vllm_pids", lambda: vllm)
    cores = [{"cpu": 0, "max_khz": 3_900_000, "cur_khz": 3_900_000 if freq_ok else 2_000_000,
              "governor": "performance"},
             {"cpu": 1, "max_khz": 2_800_000, "cur_khz": 1_000_000, "governor": "powersave"}]
    monkeypatch.setattr(hw, "core_map", lambda: cores)

    class VM:
        available = avail_gb * 2**30

    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: VM())
    monkeypatch.setattr("os.getloadavg", lambda: (load, load, load))
    return hg


def test_guard_refuses_with_vllm(monkeypatch):
    hg = _fake_state(monkeypatch, [{"pid": 7, "name": "VLLM::EngineCore", "used_mib": 44000}], [7])
    st = hg.host_state()
    assert not st["host_timing_ok"] and st["host_vllm_running"]
    with pytest.raises(hg.HostNotQuiet, match="vLLM"):
        hg.check_timing()


def test_guard_whitelists_remote_desktop_and_checks_only_perf_cores(monkeypatch):
    # the efficiency core (cpu 1) is slow and on powersave; only performance cores matter
    hg = _fake_state(monkeypatch, [{"pid": 3, "name": "/usr/libexec/gnome-remote-desktop-daemon",
                                    "used_mib": 279}], [])
    st = hg.check_timing()
    assert st["host_timing_ok"] and not st["host_vllm_running"]
    assert st["host_perf_cores"] == "[0]"


@pytest.mark.parametrize("kw,needle", [({"avail_gb": 10}, "RAM"), ({"load": 9.0}, "load"),
                                       ({"freq_ok": False}, "frequency")])
def test_guard_other_failures(monkeypatch, kw, needle):
    hg = _fake_state(monkeypatch, [], [], **kw)
    st = hg.host_state()
    assert not st["host_timing_ok"] and needle in st["host_problems"]


# ---------------------------------------------------------------------------- schema

def test_timed_row_requires_quiet_host():
    from nestbench.schema import base_row, validate

    with pytest.raises(ValueError, match="host_guard"):
        base_row("E0", timed=True, host={"host_timing_ok": False}, dataset="x", D=8, N=1,
                 index="flat")
    r = base_row("E0", timed=False, dataset="x", D=8, N=1, index="flat")
    validate([r])
    assert r["faiss_compile_options"] and r["hardware_id"]
    bad = dict(r, dataset=None)
    with pytest.raises(ValueError, match="dataset"):
        validate([bad])


# ---------------------------------------------------------------------------- memory

def test_rss_delta_tracks_index_size(tmp_path, toy_queries):
    import faiss

    from nestbench.memory import measure_index_rss

    rng = np.random.default_rng(0)
    x = rng.standard_normal((200_000, 32)).astype(np.float32)
    idx = faiss.IndexFlatL2(32)
    idx.add(x)
    p = tmp_path / "flat.faiss"
    faiss.write_index(idx, str(p))
    qn = tmp_path / "q.npy"
    np.save(qn, toy_queries)
    rec = measure_index_rss(str(p), str(qn), n_warmup=5)
    # 200K x 32 x 4 B = 25.6 MB; RSS delta should be within a few MB of that
    assert 0.8 * 25.6e6 < rec["rss_bytes"] < 1.5 * 25.6e6
    assert rec["ntotal"] == 200_000


# ------------------------------------------------------------------- stats / pareto

def test_holm_and_randomization():
    from nestbench.stats import boot_mean_ci, holm, randomization_p

    assert list(holm(np.array([0.01, 0.04, 0.03]))) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    assert randomization_p(rng.normal(0.5, 1, 400), b=2000) < 0.01
    assert randomization_p(rng.normal(0.0, 1, 400), b=2000) > 0.01
    m, lo, hi = boot_mean_ci(rng.normal(1, 1, 500), b=2000)
    assert lo < m < hi


def test_pareto_keeps_nondominated():
    from nestbench.analysis.pareto import pareto

    d = pd.DataFrame({"bytes": [1, 2, 3, 4], "lat": [10, 5, 7, 1]})
    assert pareto(d, "bytes", "lat")["bytes"].tolist() == [1, 2, 4]


# --------------------------------------------------------------------------- hardware

def test_snapshot_records_faiss_build_and_perf_cores():
    from nestbench.hardware import snapshot

    s = snapshot()
    assert s["faiss_version"] != "unavailable"
    assert s["performance_cores"], "at least one performance core"
    assert len(s["hardware_id"]) == 16
