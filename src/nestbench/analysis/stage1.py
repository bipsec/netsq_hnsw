"""Stage 1 analysis: benchmark-suite table, energy-spectrum and premise figures, gate checks.

    python -m nestbench.analysis.stage1

Writes:
- results/summary/stage1_datasets.csv and .md (the dataset table);
- results/summary/stage1_premise.csv;
- figures/fig_spectrum.{png,pdf,md} and figures/fig_premise.{png,pdf,md};
- results/summary/stage1_gates.json.

"Paper order" is the order the method uses by default (v4 A1): native MRL order for MS MARCO,
PCA order for everything else.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nestbench.analysis.style import INK_2, NEUTRAL, SERIES, save, style, write_data
from nestbench.datasets import ROOT, Dataset, registry

SUMMARY = ROOT / "results" / "summary"
MAIN = ["msm-a-384", "msm-a-768", "msm-a-1024", "msm-a-1536", "msm-b-1024",
        "sift-128", "gist-960", "glove-100", "deep-96"]
SHORT = {"msm-a-384": "MSM-A-384", "msm-a-768": "MSM-A-768", "msm-a-1024": "MSM-A-1024",
         "msm-a-1536": "MSM-A-1536", "msm-b-1024": "MSM-B-1024", "sift-128": "SIFT",
         "gist-960": "GIST", "glove-100": "GloVe", "deep-96": "DEEP"}


def paper_order(name: str) -> str:
    return "native" if registry()[name]["native_order"] == "mrl" else "pca"


def dataset_table() -> pd.DataFrame:
    rows = []
    for n, spec in registry().items():
        ds = Dataset(n)
        m = ds.meta()
        if "labels" not in m:
            continue
        lab, gt, chk = m["labels"], m.get("gt", {}), m.get("gt_check") or {}
        po = paper_order(n)
        tails = lab["tail_at_block_ends"][po]
        rows.append({
            "dataset": n, "role": spec["role"], "family": spec["family"], "D": ds.D, "N": ds.N,
            "nq": m["nq"], "normalized": spec["normalize"], "paper_order": po,
            "blocks": "|".join(map(str, lab["blocks"])),
            "C25_native": lab["C25"]["native"], "C25_pca": lab["C25"]["pca"],
            "C25_random": lab["C25"]["random"], "C25_paper": lab["C25"][po],
            "tail_after_b1": tails[0], "tail_after_b2": tails[1],
            "pca_dims_90pct": lab["pca_dims_for_90pct"], "lid_mle": lab["lid_mle_mean"],
            "r10_median": lab["density_r10_quartiles"][1],
            "norm_mean": m["base_norm_mean"],
            "gt_uncertified": gt.get("n_uncertified"),
            "gt_check_source": chk.get("source"),
            "gt_check_recall10": chk.get("tie_tolerant_recall@10"),
            "gt_check_recall100": chk.get("tie_tolerant_recall@100"),
            "gt_check_exact_set10": chk.get("exact_set_match@10"),
        })
    return pd.DataFrame(rows)


def premise_table() -> pd.DataFrame:
    files = sorted((ROOT / "results" / "raw" / "stage1").glob("premise_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else \
        pd.DataFrame()


def fig_spectrum(tab: pd.DataFrame) -> Path:
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    curves = []
    # (a) real sets in paper order, (b) MS MARCO: native vs PCA vs random, (c) S1 grid
    for i, n in enumerate(m for m in MAIN if m in set(tab.dataset)):
        po = paper_order(n)
        t = np.load(Dataset(n).path("spectrum.npz"))[f"tail_{po}"]
        x = np.arange(len(t)) / (len(t) - 1)
        axes[0].plot(x, np.maximum(t, 1e-4), color=SERIES[i % 3], lw=0.9,
                     marker=markers[i], markevery=max(1, len(t) // 6), ms=3,
                     label=f"{SHORT[n]} ({po})")
        curves.append(pd.DataFrame({"panel": "a", "dataset": n, "order": po,
                                    "axis_frac": x[:: max(1, len(t) // 32)],
                                    "tail": t[:: max(1, len(t) // 32)]}))
    for j, order in enumerate(("native", "pca", "random")):
        for n, ls in (("msm-a-1536", "-"), ("msm-b-1024", "--")):
            if n not in set(tab.dataset):
                continue
            t = np.load(Dataset(n).path("spectrum.npz"))[f"tail_{order}"]
            x = np.arange(len(t)) / (len(t) - 1)
            axes[1].plot(x, np.maximum(t, 1e-4), color=SERIES[j], ls=ls, lw=0.9,
                         label=f"{SHORT[n]} {order}")
            curves.append(pd.DataFrame({"panel": "b", "dataset": n, "order": order,
                                        "axis_frac": x[:: max(1, len(t) // 32)],
                                        "tail": t[:: max(1, len(t) // 32)]}))
    for j, beta in enumerate((0, 0.5, 1, 2)):
        n = f"s1-d1024-b{beta:g}"
        if n not in set(tab.dataset):
            continue
        t = np.load(Dataset(n).path("spectrum.npz"))["tail_pca"]
        x = np.arange(len(t)) / (len(t) - 1)
        axes[2].plot(x, np.maximum(t, 1e-4), color=(SERIES + [NEUTRAL])[j], lw=0.9,
                     marker=markers[j], markevery=170, ms=3, label=f"β = {beta:g}")
        curves.append(pd.DataFrame({"panel": "c", "dataset": n, "order": "pca",
                                    "axis_frac": x[::32], "tail": t[::32]}))
    titles = ["(a) Real sets, paper order", "(b) MS MARCO: three orders",
              "(c) Synthetic S1, D = 1024, PCA"]
    for ax, ti in zip(axes, titles, strict=True):
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.05)
        ax.set_xlim(0, 1)
        ax.axvline(0.25, color=NEUTRAL, lw=0.6, ls=":")
        ax.set_title(ti, loc="left")
        ax.set_xlabel("share of axes read")
        ax.legend(fontsize=5.2, loc="lower left", handlelength=1.6)
    axes[0].set_ylabel("tail variance share (unread)")
    axes[0].text(0.26, 0.6, "C25 cut", fontsize=5.5, color=INK_2)
    out = save(fig, "fig_spectrum")
    write_data("fig_spectrum", "Energy spectra",
               "Share of total per-axis variance left in the unread tail after reading the "
               "first share of axes (tail[p] = 1 - cumulative variance share). C25 = 1 - tail at "
               "0.25. Sampled every ~1/32 of the axes.",
               [("Curves", None, pd.concat(curves, ignore_index=True))],
               ["data/sets/<name>/spectrum.npz"])
    return out


def fig_premise(pr: pd.DataFrame) -> Path:
    plt = style()
    d = pr[(pr.ef == 100) & pr.dataset.isin(MAIN)].copy()
    d = d[d.apply(lambda r: r.order == paper_order(r.dataset), axis=1)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    for ax, pool in zip(axes, ("hard", "random"), strict=True):
        for i, n in enumerate(m for m in MAIN if m in set(d.dataset)):
            e = d[(d.dataset == n) & (d.pool == pool)].sort_values("prefix_frac")
            ax.plot(e.prefix_frac, e.rejected, color=SERIES[i % 3], lw=0.9, marker=markers[i],
                    ms=2.8, label=SHORT[n])
        ax.set_title(f"({'ab'[pool == 'random']}) {pool} candidates, ef = 100", loc="left")
        ax.set_xlabel("share of axes read (paper order)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("rejected by exact prefix bound")
    axes[1].legend(fontsize=5.5, loc="lower right", ncol=2)
    out = save(fig, "fig_premise")
    write_data("fig_premise", "Premise check: dimension-only exact bound",
               "Share of candidates whose lower bound |q_P - x_P|^2 + (|q_T| - |x_T|)^2 exceeds "
               "tau = the 100th true distance, after reading a prefix P of the axes, on exact "
               "fp32 data (no quantization). 200 queries per set; hard = true ranks 101-600, "
               "random = 500 uniform base points.",
               [("Points", None, d[["dataset", "pool", "order", "prefix", "prefix_frac",
                                    "rejected"]].reset_index(drop=True))],
               ["results/raw/stage1/premise_*.parquet"])
    return out


def gates(tab: pd.DataFrame, pr: pd.DataFrame) -> dict:
    chk = tab[tab.gt_check_source.notna()]
    provided = chk[chk.gt_check_source == "provided"]
    g = {
        "gt_all_certified": bool((tab.gt_uncertified == 0).all()),
        "gt_matches_provided": bool((provided.gt_check_recall10 == 1.0).all()
                                    and (provided.gt_check_recall100 >= 0.9999).all()),
        "gt_provided_sets": provided.dataset.tolist(),
        "gt_matches_fewer_bits": bool((chk[chk.gt_check_source != "provided"]
                                       .gt_check_recall10 >= 0.9999).all()),
    }
    hi = tab[(tab.D >= 384) & tab.dataset.isin(MAIN)]
    ok = hi[(hi.tail_after_b1 <= 0.15) | (hi.tail_after_b2 <= 0.15)]
    g["premise_tail_le_15pct_high_D"] = ok.dataset.tolist()
    g["premise_pass"] = bool(len(ok) >= 1)
    g["pass"] = g["gt_all_certified"] and g["gt_matches_provided"] and g["premise_pass"]
    return g


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    tab = dataset_table()
    tab.to_csv(SUMMARY / "stage1_datasets.csv", index=False)
    (SUMMARY / "stage1_datasets.md").write_text(tab.to_markdown(index=False, floatfmt=".4g"))
    pr = premise_table()
    pr.to_csv(SUMMARY / "stage1_premise.csv", index=False)
    fig_spectrum(tab)
    if len(pr):
        fig_premise(pr)
    g = gates(tab, pr)
    (SUMMARY / "stage1_gates.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))


if __name__ == "__main__":
    main()
