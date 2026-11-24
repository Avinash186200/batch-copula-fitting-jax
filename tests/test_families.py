import numpy as np
import pytest
from scipy import stats

from copulabatch import families as F

ALL_PARAMS = {"gaussian": 0.6, "t": (0.6, 5.0), "clayton": 2.0, "clayton_180": 2.0,
              "clayton_90": 2.0, "clayton_270": 2.0, "gumbel": 2.0, "gumbel_180": 2.0,
              "gumbel_90": 2.0, "gumbel_270": 2.0, "frank": 5.0}


def test_registry_covers_all_families():
    assert set(ALL_PARAMS) == set(F.FAMILY_NAMES)
    assert F.get("t").k == 2 and F.get("frank").k == 1


@pytest.mark.parametrize("name", F.FAMILY_NAMES)
def test_density_matches_mixed_derivative_of_cdf(name):
    p = ALL_PARAMS[name]
    h = 1e-4
    for u, v in [(0.3, 0.4), (0.7, 0.2), (0.15, 0.85), (0.6, 0.6)]:
        num = (F.cdf(name, p, u + h, v + h) - F.cdf(name, p, u + h, v - h)
               - F.cdf(name, p, u - h, v + h) + F.cdf(name, p, u - h, v - h)) / (4 * h * h)
        dens = np.exp(F.logpdf(name, p, np.array([u]), np.array([v])))[0]
        assert dens == pytest.approx(num, rel=1e-4)


@pytest.mark.parametrize("name", F.FAMILY_NAMES)
def test_density_integrates_to_one(name):
    # midpoint rule on a 400 x 400 grid; tails are integrable for every family
    g = (np.arange(400) + 0.5) / 400
    u, v = np.meshgrid(g, g)
    total = np.exp(F.logpdf(name, ALL_PARAMS[name], u.ravel(), v.ravel())).mean()
    assert total == pytest.approx(1.0, abs=0.02)


def test_gaussian_and_t_match_scipy_joint_densities():
    u = np.array([0.1, 0.5, 0.9, 0.33])
    v = np.array([0.2, 0.7, 0.95, 0.05])
    rho, nu = 0.55, 6.0
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    ref = (stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).logpdf(np.c_[x, y])
           - stats.norm.logpdf(x) - stats.norm.logpdf(y))
    np.testing.assert_allclose(F.logpdf("gaussian", rho, u, v), ref, rtol=1e-10)
    x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    ref = (stats.multivariate_t([0, 0], [[1, rho], [rho, 1]], df=nu).logpdf(np.c_[x, y])
           - stats.t.logpdf(x, nu) - stats.t.logpdf(y, nu))
    np.testing.assert_allclose(F.logpdf("t", (rho, nu), u, v), ref, rtol=1e-9)


@pytest.mark.parametrize("name", F.FAMILY_NAMES)
def test_sampler_matches_kendall_tau(name):
    p = ALL_PARAMS[name]
    u, v = F.sample(name, p, 4000, np.random.default_rng(7))
    assert ((u > 0) & (u < 1) & (v > 0) & (v < 1)).all()
    assert stats.kendalltau(u, v)[0] == pytest.approx(F.tau(name, p), abs=0.03)


@pytest.mark.parametrize("name", F.FAMILY_NAMES)
def test_param_from_tau_inverts_tau(name):
    for t in (0.1, 0.4, 0.7):
        t = -t if F.get(name).rotation in (90, 270) else t
        assert F.tau(name, F.param_from_tau(name, t, nu=5.0)) == pytest.approx(t, abs=1e-9)


def test_tail_closed_forms():
    assert F.tail_dependence("gaussian", 0.9) == (0.0, 0.0)
    assert F.tail_dependence("frank", 10.0) == (0.0, 0.0)
    assert F.tail_dependence("clayton", 2.0) == pytest.approx((2 ** -0.5, 0.0))
    assert F.tail_dependence("clayton_180", 2.0) == pytest.approx((0.0, 2 ** -0.5))
    assert F.tail_dependence("gumbel", 2.0) == pytest.approx((0.0, 2 - 2 ** 0.5))
    assert F.tail_dependence("gumbel_180", 2.0) == pytest.approx((2 - 2 ** 0.5, 0.0))
    lo, up = F.tail_dependence("t", (0.5, 4.0))
    assert lo == up and 0.2 < lo < 0.4
    # t tail dependence falls as nu grows and is positive even at rho = 0
    assert F.tail_dependence("t", (0.5, 30.0))[0] < lo
    assert F.tail_dependence("t", (0.0, 4.0))[0] > 0


def test_tail_closed_form_is_the_limit_of_c_qq_over_q():
    for name in ("clayton", "gumbel_180", "t"):
        p = ALL_PARAMS[name]
        lam = F.tail_dependence(name, p)[0]
        assert F.cdf(name, p, 1e-6, 1e-6) / 1e-6 == pytest.approx(lam, abs=2e-3)


def test_tail_corners():
    assert F.tail_corners("t") == {"LL", "UU"}
    assert F.tail_corners("gaussian") == frozenset()
    assert F.tail_corners("clayton") == {"LL"}
    assert F.tail_corners("gumbel_180") == {"LL"}
    assert F.tail_corners("clayton_90") == {"HL"}
    assert F.tail_corners("gumbel_90") == {"LH"}


def test_rotated_cdf_matches_samples():
    rng = np.random.default_rng(3)
    for name in ("clayton_90", "gumbel_270", "clayton_180"):
        u, v = F.sample(name, 2.0, 200_000, rng)
        for a, b in [(0.2, 0.3), (0.8, 0.1)]:
            assert np.mean((u <= a) & (v <= b)) == pytest.approx(F.cdf(name, 2.0, a, b), abs=4e-3)


def test_pseudo_obs_are_ranks_over_n_plus_1():
    x = np.array([[3.0, 1.0, 2.0, 10.0]])
    np.testing.assert_allclose(F.pseudo_obs(x), [[0.6, 0.2, 0.4, 0.8]])
    # ties get average ranks
    np.testing.assert_allclose(F.pseudo_obs(np.array([1.0, 1.0, 2.0])), [0.375, 0.375, 0.75])


def test_empirical_tail_counts_joint_exceedances():
    u = np.array([0.01, 0.02, 0.5, 0.99, 0.98])
    v = np.array([0.03, 0.9, 0.5, 0.97, 0.01])
    lo, up = F.empirical_tail(u, v, 0.05)
    assert lo == pytest.approx(1 / (5 * 0.05)) and up == pytest.approx(1 / (5 * 0.05))


def test_gaussian_diagonal_owens_t_matches_quadrature():
    for rho in (-0.6, 0.0, 0.3, 0.9):
        for q in (0.01, 0.05, 0.3, 0.8):
            fast = F.cdf("gaussian", rho, q, q)
            slow = F.cdf("gaussian", rho, q, q + 1e-13)      # off-diagonal: quadrature path
            assert fast == pytest.approx(slow, rel=1e-8, abs=1e-12)
