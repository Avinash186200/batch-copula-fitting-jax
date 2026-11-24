"""Brent's bounded 1-D minimiser written branch-free, so one code path serves

* NumPy: all N problems advance together as arrays (``minimize_numpy``);
* JAX: a scalar problem inside ``lax.while_loop``, batched by ``vmap`` (see ``batch_jax``).

It is the classic golden-section + successive-parabolic-interpolation method (Brent 1973,
"localmin"), the same algorithm as ``scipy.optimize.minimize_scalar(method="bounded")``.
Every ``if`` of the textbook version becomes an ``xp.where`` on the state, so each lane
follows its own path. Converged lanes are frozen.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

GOLDEN = 0.5 * (3.0 - np.sqrt(5.0))
SQRT_EPS = float(np.sqrt(np.finfo(float).eps))


class BrentState(NamedTuple):
    a: object
    b: object
    v: object
    w: object
    x: object
    fv: object
    fw: object
    fx: object
    d: object
    e: object
    done: object
    nfev: object


def init(f, a, b, xp) -> BrentState:
    x = a + GOLDEN * (b - a)
    fx = f(x)
    zero = xp.zeros_like(x)
    return BrentState(a, b, x, x, x, fx, fx, fx, zero, zero, xp.zeros_like(x, dtype=bool),
                      xp.ones_like(x, dtype=np.int32))


def converged(s: BrentState, xtol, xp):
    tol1 = SQRT_EPS * xp.abs(s.x) + xtol / 3.0
    m = 0.5 * (s.a + s.b)
    return xp.abs(s.x - m) <= 2.0 * tol1 - 0.5 * (s.b - s.a)


def step(s: BrentState, f, xtol, xp) -> BrentState:
    """One Brent iteration on every lane that is not done."""
    a, b, v, w, x, fv, fw, fx, d, e = s[:10]
    tol1 = SQRT_EPS * xp.abs(x) + xtol / 3.0
    tol2 = 2.0 * tol1
    m = 0.5 * (a + b)
    done = s.done | (xp.abs(x - m) <= tol2 - 0.5 * (b - a))

    # trial parabola through (v, fv), (w, fw), (x, fx)
    r = (x - w) * (fx - fv)
    q = (x - v) * (fx - fw)
    p = (x - v) * q - (x - w) * r
    q = 2.0 * (q - r)
    p = xp.where(q > 0.0, -p, p)
    q = xp.abs(q)
    q_safe = xp.where(q == 0.0, 1.0, q)
    use_para = ((xp.abs(e) > tol1) & (xp.abs(p) < xp.abs(0.5 * q * e))
                & (p > q * (a - x)) & (p < q * (b - x)))
    d_para = p / q_safe
    u_para = x + d_para
    near_edge = ((u_para - a) < tol2) | ((b - u_para) < tol2)
    d_para = xp.where(near_edge, xp.where(x <= m, tol1, -tol1), d_para)

    e_gold = xp.where(x >= m, a - x, b - x)
    d_new = xp.where(use_para, d_para, GOLDEN * e_gold)
    e_new = xp.where(use_para, d, e_gold)

    u = x + xp.where(xp.abs(d_new) >= tol1, d_new, xp.where(d_new >= 0.0, tol1, -tol1))
    fu = f(u)

    better = fu <= fx
    a_n = xp.where(better, xp.where(u >= x, x, a), xp.where(u < x, u, a))
    b_n = xp.where(better, xp.where(u >= x, b, x), xp.where(u < x, b, u))
    # not better: maybe u replaces w, or v
    to_w = (~better) & ((fu <= fw) | (w == x))
    to_v = (~better) & (~to_w) & ((fu <= fv) | (v == x) | (v == w))
    v_n = xp.where(better | to_w, w, xp.where(to_v, u, v))
    fv_n = xp.where(better | to_w, fw, xp.where(to_v, fu, fv))
    w_n = xp.where(better, x, xp.where(to_w, u, w))
    fw_n = xp.where(better, fx, xp.where(to_w, fu, fw))
    x_n = xp.where(better, u, x)
    fx_n = xp.where(better, fu, fx)

    keep = lambda old, new: xp.where(done, old, new)
    return BrentState(keep(a, a_n), keep(b, b_n), keep(v, v_n), keep(w, w_n), keep(x, x_n),
                      keep(fv, fv_n), keep(fw, fw_n), keep(fx, fx_n), keep(d, d_new),
                      keep(e, e_new), done, s.nfev + xp.where(done, 0, 1).astype(s.nfev.dtype))


def minimize_numpy(f, a, b, xtol=1e-8, maxiter=500):
    """Minimise f elementwise over [a, b] for arrays a, b. f maps an (N,) array to (N,).

    Returns (x, fx, nfev, converged). Every lane is evaluated on every iteration (simple and
    vectorised); frozen lanes just ignore the result.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = init(f, a, b, np)
    for _ in range(maxiter):
        s = step(s, f, xtol, np)
        if s.done.all():
            break
    return s.x, s.fx, s.nfev, converged(s, xtol, np)
