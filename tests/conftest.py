import numpy as np
import pytest

from copulabatch.sim import pairs

# tests draw from their own seeds; the committed results use the held-out split in bench.py
MIXED = [("gaussian", 0.6), ("t", (0.5, 4.0)), ("clayton", 2.0), ("clayton_180", 1.5),
         ("gumbel", 1.8), ("gumbel_180", 2.2), ("frank", 5.0), ("clayton_90", 1.2),
         ("gumbel_270", 1.6), ("frank", -4.0), ("gaussian", -0.4)]


@pytest.fixture(scope="session")
def mixed_specs():
    return MIXED


@pytest.fixture(scope="session")
def mixed_pairs():
    return pairs(MIXED, 250, np.random.default_rng(12345))


def needs_jax():
    return pytest.importorskip("jax")
