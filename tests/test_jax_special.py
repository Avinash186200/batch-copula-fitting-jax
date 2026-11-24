import numpy as np
import pytest
from scipy import special

jax = pytest.importorskip("jax")

from copulabatch import jax_ops  # noqa: F401  (float64)
from copulabatch import jax_special as js

NUS = (2.0, 2.5, 3.0, 4.0, 7.0, 15.0, 30.0, 50.0)


def test_t_ppf_matches_scipy():
    p = np.concatenate([np.geomspace(1e-6, 0.5, 300), 1 - np.geomspace(1e-6, 0.5, 300), [0.5]])
    for nu in NUS:
        got = np.asarray(js.t_ppf(p, nu))
        want = special.stdtrit(nu, p)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def test_t_cdf_matches_scipy():
    x = np.concatenate([-np.geomspace(1e-6, 300, 300), np.geomspace(1e-6, 300, 300), [0.0]])
    for nu in NUS:
        np.testing.assert_allclose(np.asarray(js.t_cdf(x, nu)), special.stdtr(nu, x),
                                   rtol=1e-11, atol=1e-14)


def test_log_betainc_matches_scipy():
    x = np.linspace(0.001, 0.999, 200)
    for a in (0.5, 1.0, 3.0, 12.5, 25.0):
        got = np.asarray(js.log_betainc(a, 0.5, x, 1 - x))
        np.testing.assert_allclose(got, np.log(special.betainc(a, 0.5, x)), rtol=1e-10, atol=1e-12)


def test_t_ppf_broadcasts_over_nu():
    p = np.array([[0.01, 0.3], [0.7, 0.99]])
    nu = np.array([[3.0], [20.0]])
    np.testing.assert_allclose(np.asarray(js.t_ppf(p, nu)), special.stdtrit(nu, p), rtol=1e-9)
