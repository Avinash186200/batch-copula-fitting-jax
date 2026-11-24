import numpy as np
import pytest
from scipy import stats

from copulabatch import families as F
from copulabatch.bench import window_pseudo_obs
from copulabatch.sim import RegimeConfig, pairs, regime_returns

CFG = RegimeConfig(n_assets=4, days=900, stress_start=300, stress_end=600)


def test_pairs_are_pseudo_observations():
    U, V = pairs([("gaussian", 0.5), ("clayton", 1.0)], 50, np.random.default_rng(0))
    assert U.shape == V.shape == (2, 50)
    np.testing.assert_allclose(np.sort(U, axis=1), np.tile(np.arange(1, 51) / 51, (2, 1)))


def test_regimes_have_the_intended_dependence():
    R = regime_returns(CFG, np.random.default_rng(1))
    assert R.shape == (900, 4)
    s = CFG.stress_mask()
    calm, stress = R[~s], R[s]
    assert stress.std() > 2 * calm.std()
    tau_calm = stats.kendalltau(calm[:, 0], calm[:, 1])[0]
    tau_stress = stats.kendalltau(stress[:, 0], stress[:, 1])[0]
    assert tau_calm == pytest.approx(F.tau("gaussian", CFG.calm_rho), abs=0.08)
    assert tau_stress == pytest.approx(F.tau("clayton", CFG.stress_theta), abs=0.08)


def test_window_pseudo_obs_are_point_in_time():
    R = regime_returns(CFG, np.random.default_rng(2))
    ends, W, prs = [200, 400], 150, [(0, 1), (2, 3)]
    U, V = window_pseudo_obs(R, ends, W, prs)
    R2 = R.copy()
    R2[200:] = np.random.default_rng(3).standard_normal(R2[200:].shape)   # rewrite day 200 on
    U2, V2 = window_pseudo_obs(R2, ends, W, prs)
    # rows 0-1: the window ending at day 200 (exclusive) never sees the rewrite
    np.testing.assert_array_equal(U[:2], U2[:2])
    np.testing.assert_array_equal(V[:2], V2[:2])
    # rows 2-3: the window ending at day 400 does, so the test can detect a leak
    assert not np.array_equal(U[2:], U2[2:])
    assert not np.array_equal(V[2:], V2[2:])
    # and the window really is [end - W, end)
    np.testing.assert_allclose(U[0], F.pseudo_obs(R[50:200, 0]))
