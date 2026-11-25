"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def toy_vectors():
    """100 random unit-norm vectors of dimension 32."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((100, 32)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def toy_queries():
    """10 random unit-norm query vectors of dimension 32."""
    rng = np.random.default_rng(99)
    vecs = rng.standard_normal((10, 32)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
