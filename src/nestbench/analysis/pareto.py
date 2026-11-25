"""Pareto-frontier helpers — copied from fewer_bits (annstudy.analysis.tradeoff)."""

import numpy as np
import pandas as pd


def qualifying(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["reachable"].fillna(False).astype(bool)]
    if "meets_R_on_test" in d.columns:
        d = d[d["meets_R_on_test"].fillna(False).astype(bool)]
    return d


def pareto(d: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Rows not dominated in (x, y), both minimised; sorted by x."""
    d = d.sort_values([x, y])
    keep, best = [], np.inf
    for i, r in d.iterrows():
        if r[y] < best:
            keep.append(i)
            best = r[y]
    return d.loc[keep]


