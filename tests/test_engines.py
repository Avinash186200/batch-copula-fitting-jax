import numpy as np
import pytest

from copulabatch import FAMILY_NAMES
from copulabatch import families as F
from copulabatch.bench import TOL_LOGLIK, TOL_PARAM, TOL_T_LOGLIK, TOL_T_RHO
from copulabatch.engines import fit, grid_ranks
from copulabatch.sim import pairs


@pytest.fixture(scope="module")
def reference(mixed_pairs):
    return fit(*mixed_pairs, engine="reference")


def _assert_agrees(r, ref):
    one = [j for j, f in enumerate(ref.families) if f != "t"]
    jt = ref.families.index("t")
    assert np.abs(r.theta[:, one] - ref.theta[:, one]).max() <= TOL_PARAM
    assert np.abs(r.loglik[:, one] - ref.loglik[:, one]).max() <= TOL_LOGLIK
    assert np.abs(r.theta[:, jt] - ref.theta[:, jt]).max() <= TOL_T_RHO
    assert (r.loglik[:, jt] - ref.loglik[:, jt]).min() >= -TOL_T_LOGLIK
    np.testing.assert_array_equal(r.best, ref.best)


def test_numpy_engine_agrees_with_reference(mixed_pairs, reference):
    _assert_agrees(fit(*mixed_pairs, engine="numpy"), reference)


def test_jax_engine_agrees_with_reference(mixed_pairs, reference):
    pytest.importorskip("jax")
    _assert_agrees(fit(*mixed_pairs, engine="jax"), reference)


def test_jax_chunking_and_padding_do_not_change_results(mixed_pairs):
    pytest.importorskip("jax")
    U, V = mixed_pairs
    fams = ("gaussian", "clayton", "frank")
    whole = fit(U, V, engine="jax", families=fams)
    chunked = fit(U, V, engine="jax", families=fams, chunk=8)    # 11 pairs -> 8 + 3 padded
    np.testing.assert_allclose(chunked.theta, whole.theta, rtol=0, atol=1e-12)


def test_reference_picks_the_right_tail_class_on_clear_cases(mixed_specs, reference):
    # n = 250 with strong dependence: most picks should be right (not a statistical claim)
    hits = sum(F.tail_corners(t) == F.tail_corners(c)
               for (t, _), c in zip(mixed_specs, reference.best_name))
    assert hits >= 8


def test_rank_grid_fast_path_matches_general_path():
    U, V = pairs([("t", (0.6, 4.0))] * 3, 201, np.random.default_rng(5))
    assert grid_ranks(U, V) is not None
    fast = fit(U, V, engine="numpy", families=("t",))
    # nudge one value off the grid: forces the general (no-ranks) path
    U2 = U.copy()
    U2[:, 0] += 1e-9
    assert grid_ranks(U2, V) is None
    slow = fit(U2, V, engine="numpy", families=("t",))
    np.testing.assert_allclose(fast.theta, slow.theta, atol=1e-6)
    np.testing.assert_allclose(fast.loglik, slow.loglik, atol=1e-5)


def test_ties_use_the_general_path_and_still_fit():
    x = np.round(np.random.default_rng(1).standard_normal((2, 300)), 1)   # many ties
    y = x + np.round(np.random.default_rng(2).standard_normal((2, 300)), 1)
    U, V = F.pseudo_obs(x), F.pseudo_obs(y)
    assert grid_ranks(U, V) is None
    r = fit(U, V, engine="numpy", families=("gaussian", "t"))
    assert np.isfinite(r.loglik).all() and (r.theta[:, 0] > 0.5).all()


def test_fit_result_shapes_and_aic():
    U, V = pairs([("clayton", 3.0)] * 4, 300, np.random.default_rng(9))
    r = fit(U, V, engine="numpy")
    F_ = len(FAMILY_NAMES)
    assert r.theta.shape == r.nu.shape == r.loglik.shape == (4, F_)
    jt = r.families.index("t")
    assert np.isnan(np.delete(r.nu, jt, axis=1)).all() and np.isfinite(r.nu[:, jt]).all()
    np.testing.assert_allclose(r.aic, 2 * r.k - 2 * r.loglik)
    assert r.k[jt] == 2 and r.k.sum() == F_ + 1
    assert isinstance(r.params(0, "t"), tuple) and isinstance(r.params(0, "frank"), float)


def test_single_pair_1d_input_is_accepted():
    U, V = pairs([("gumbel", 2.0)], 200, np.random.default_rng(4))
    r = fit(U[0], V[0], engine="numpy", families=("gumbel",))
    assert r.theta.shape == (1, 1)


def test_rejects_values_outside_the_unit_interval():
    with pytest.raises(ValueError):
        fit(np.array([[0.0, 0.5]]), np.array([[0.2, 0.5]]), engine="numpy")
    with pytest.raises(ValueError):
        fit(np.ones((2, 3)) * 0.5, np.ones((3, 3)) * 0.5, engine="numpy")
    with pytest.raises(ValueError):
        fit(np.full((1, 3), 0.5), np.full((1, 3), 0.5), engine="cuda")


def test_independence_gives_parameters_near_their_independence_values():
    U, V = pairs([("frank", 1e-12)] * 2, 2000, np.random.default_rng(11))
    r = fit(U, V, engine="numpy", families=("gaussian", "frank", "clayton"))
    assert np.abs(r.theta[:, 0]).max() < 0.1
    assert np.abs(r.theta[:, 1]).max() < 0.6
    assert r.theta[:, 2].max() < 0.2


def test_converged_flags_are_reported_by_every_engine(mixed_pairs, reference):
    assert reference.converged.shape == reference.theta.shape
    assert fit(*mixed_pairs, engine="numpy").converged.all()
    if pytest.importorskip("jax"):
        assert fit(*mixed_pairs, engine="jax").converged.all()


def test_iteration_cap_is_flagged(monkeypatch, mixed_pairs):
    import copulabatch.engines as E
    monkeypatch.setattr(E, "MAXITER", 2)
    r = fit(*mixed_pairs, engine="numpy", families=("gaussian", "t"))
    assert not r.converged.any()
