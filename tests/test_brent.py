import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from copulabatch import brent

CENTRES = np.array([-3.5, -1.0, 0.2, 2.0, 4.9])


def f(x):
    return (x - CENTRES) ** 2 * (1 + 0.1 * np.sin(3 * x)) + np.cosh(x - CENTRES)


def test_matches_scipy_bounded_brent():
    x, fx, nfev, ok = brent.minimize_numpy(f, np.full(5, -4.0), np.full(5, 6.0), xtol=1e-9)
    assert ok.all()
    for i in range(5):
        r = minimize_scalar(lambda t: f(np.full(5, t))[i], bounds=(-4, 6), method="bounded",
                            options={"xatol": 1e-9})
        assert x[i] == pytest.approx(r.x, abs=1e-8)
        assert nfev[i] == r.nfev            # same algorithm, same number of evaluations


def test_minimum_on_the_boundary():
    x, _, _, ok = brent.minimize_numpy(lambda t: t, np.array([1.0, -2.0]), np.array([3.0, 5.0]),
                                       xtol=1e-8)
    assert ok.all()
    np.testing.assert_allclose(x, [1.0, -2.0], atol=1e-6)


def test_lanes_are_independent():
    x_all, *_ = brent.minimize_numpy(f, np.full(5, -4.0), np.full(5, 6.0), xtol=1e-9)
    for i, c in enumerate(CENTRES):
        def g(t, c=c):
            return (t - c) ** 2 * (1 + 0.1 * np.sin(3 * t)) + np.cosh(t - c)
        xi, *_ = brent.minimize_numpy(g, np.array([-4.0]), np.array([6.0]), xtol=1e-9)
        assert xi[0] == pytest.approx(x_all[i], abs=1e-12)


def test_maxiter_reports_not_converged():
    _, _, nfev, ok = brent.minimize_numpy(f, np.full(5, -4.0), np.full(5, 6.0), xtol=1e-12,
                                          maxiter=3)
    assert not ok.any() and (nfev == 4).all()
