"""MS MARCO relevance metrics (binary qrels) — copied from fewer_bits (annstudy.relevance).

Rankings are arrays of passage ROW indices into the corpus prefix (what FAISS returns); relevant
sets are row indices too. ``-1`` entries (FAISS padding) and repeated ids are ignored, so a
malformed ranking can never score above a correct one. The relevance query set itself
(``data/relevance_queries.json``: 5,000 MS MARCO train queries with at least one relevant passage
inside the 500K prefix; positions 0-999 tuning, 1000-4999 evaluation) is copied from fewer_bits
unchanged, so MRR@10 here is comparable with the SIGIR paper.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/msmarco_raw")
REL_SET = Path("data/relevance_queries.json")
N_TUNE = 1000

# ----------------------------------------------------------------------------- metrics

def _clean(ranking, k):
    """First k distinct, valid ids of a ranking, in order."""
    seen, out = set(), []
    for x in ranking:
        x = int(x)
        if x < 0 or x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) == k:
            break
    return out


def rr_at_k(ranking, relevant: set, k: int = 10) -> float:
    """Reciprocal rank of the first relevant id within the top k, else 0."""
    for i, x in enumerate(_clean(ranking, k)):
        if x in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranking, relevant: set, k: int = 10) -> float:
    """Binary-gain nDCG@k with log2(rank+1) discount; ideal DCG from |relevant|."""
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(_clean(ranking, k)) if x in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / idcg)


def recall_at_k(ranking, relevant: set, k: int = 100) -> float:
    """Share of relevant ids found in the top k."""
    if not relevant:
        return float("nan")
    return len(set(_clean(ranking, k)) & relevant) / len(relevant)


def load_qrels(path=RAW_DIR / "qrels.train.tsv") -> dict[int, set[int]]:
    q = pd.read_csv(path, sep="\t", header=None, names=["qid", "it", "pid", "rel"])
    q = q[q["rel"] > 0]
    return {int(k): set(int(p) for p in g) for k, g in q.groupby("qid")["pid"]}


def load_train_queries(path=RAW_DIR / "queries.train.tsv") -> dict[int, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, text = line.rstrip("\n").partition("\t")
            out[int(qid)] = text
    return out


def split_positions(rel_set: dict) -> tuple[np.ndarray, np.ndarray]:
    """(tuning, evaluation) positions; asserts the stored split is the documented one."""
    n = rel_set["meta"]["n"]
    assert rel_set["meta"]["split"] == {"tuning": [0, N_TUNE - 1], "evaluation": [N_TUNE, n - 1]}
    assert [q["pos"] for q in rel_set["queries"]] == list(range(n))
    return np.arange(N_TUNE), np.arange(N_TUNE, n)


def load_relevance_set(path=REL_SET) -> dict:
    return json.loads(Path(path).read_text())
