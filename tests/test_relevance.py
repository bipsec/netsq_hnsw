"""Tests for the S2 relevance metrics and the relevance query-set builder."""

import math

from nestbench.relevance import (
    ndcg_at_k,
    recall_at_k,
    rr_at_k,
)


def test_rr_first_relevant_rank_and_cutoff():
    assert rr_at_k([5, 3, 9], {3}) == 0.5
    assert rr_at_k([5, 3, 9], {7}) == 0.0
    assert rr_at_k(list(range(20)), {15}, k=10) == 0.0  # outside the top 10


def test_padding_and_repeated_ids_never_help():
    # -1 padding is skipped; a repeated irrelevant id does not push a relevant one down twice
    assert rr_at_k([-1, 4, 4, 2], {2}) == 0.5
    assert recall_at_k([2, 2, 2], {2, 3}, k=100) == 0.5  # duplicates counted once


def test_ndcg_two_relevant_passages():
    # relevant at ranks 1 and 3; ideal has both at ranks 1-2
    got = ndcg_at_k([1, 9, 2], {1, 2})
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert abs(got - dcg / idcg) < 1e-12
    assert ndcg_at_k([1, 2], {1, 2}) == 1.0
    assert ndcg_at_k([7, 8], {1, 2}) == 0.0


def test_no_relevant_is_nan_not_zero():
    assert math.isnan(ndcg_at_k([1], set()))
    assert math.isnan(recall_at_k([1], set()))


