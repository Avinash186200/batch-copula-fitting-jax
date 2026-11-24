"""Special functions JAX does not ship (or ships slowly): the Student-t cdf and quantile.

``jax.scipy.stats.t`` has a pdf but no cdf or ppf. The cdf is

    F_nu(-y) = 1/2 * I_z(nu/2, 1/2),   z = nu / (nu + y^2),

with I the regularised incomplete beta function. ``jax.scipy.special.betainc`` exists but
evaluates its continued fraction in a per-element loop; on CPU it was far slower in
development timing than the fixed-length, fully fused continued fraction below, which XLA compiles to
straight-line vector code. The quantile is then found by Newton's method on

    phi(s) = log F(-e^s) - log q,     q = min(p, 1 - p) <= 1/2,

i.e. in log-magnitude and log-probability. In the tails F(-y) ~ c * y^-nu, so phi is
nearly linear in s and Newton converges in a few steps even for nu = 2, where a Newton
step in y itself crawls. The start is the normal quantile. Fixed iteration counts keep
everything jit- and vmap-friendly; ``tests/test_jax_special.py`` checks both functions
against ``scipy.special.stdtr`` / ``stdtrit``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy import special as jsp

CF_TERMS = 24        # continued-fraction terms (accuracy plateaus from about 20, nu <= 50)
NEWTON_STEPS = 8     # quantile Newton steps (error plateaus from about 6)
_TINY = 1e-300


def log_betainc(a, b, x, x1):
    """log I_x(a, b), with x1 = 1 - x passed in separately to avoid cancellation.

    Modified Lentz continued fraction (Numerical Recipes ``betacf``) with a fixed number
    of terms, using I_x(a, b) = 1 - I_{1-x}(b, a) when x > (a + 1) / (a + b + 2).
    """
    flip = x > (a + 1.0) / (a + b + 2.0)
    aa = jnp.where(flip, b, a)
    bb = jnp.where(flip, a, b)
    xx = jnp.where(flip, x1, x)
    xx1 = jnp.where(flip, x, x1)
    guard = lambda t: jnp.where(jnp.abs(t) < _TINY, _TINY, t)
    qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
    c = jnp.ones_like(xx)
    d = 1.0 / guard(1.0 - qab * xx / qap)
    h = d
    for m in range(1, CF_TERMS + 1):
        m2 = 2.0 * m
        for num in (m * (bb - m) * xx / ((qam + m2) * (aa + m2)),
                    -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))):
            d = 1.0 / guard(1.0 + num * d)
            c = guard(1.0 + num / c)
            h = h * d * c
    log_i = (aa * jnp.log(xx) + bb * jnp.log(xx1) - jnp.log(aa) + jnp.log(h)
             - (jsp.gammaln(aa) + jsp.gammaln(bb) - jsp.gammaln(aa + bb)))
    return jnp.where(flip, jnp.log1p(-jnp.exp(log_i)), log_i)


def t_logcdf_neg(y, nu):
    """log F_nu(-y) for y >= 0."""
    den = nu + y * y
    return jnp.log(0.5) + log_betainc(0.5 * nu, 0.5, nu / den, y * y / den)


def t_cdf(x, nu):
    """Student-t cdf, elementwise."""
    lo = jnp.exp(t_logcdf_neg(jnp.abs(x), nu))
    return jnp.where(x < 0, lo, 1.0 - lo)


def t_logpdf(x, nu):
    return (jsp.gammaln(0.5 * (nu + 1.0)) - jsp.gammaln(0.5 * nu) - 0.5 * jnp.log(nu * jnp.pi)
            - 0.5 * (nu + 1.0) * jnp.log1p(x * x / nu))


def t_ppf(p, nu):
    """Student-t quantile, elementwise; p in (0, 1), nu > 0 (broadcasts).

    Relative accuracy about 1e-9 except within 1e-7 of p = 0.5, where the quantile is near
    0 and only the absolute error (below 4e-10) is controlled; p = 0.5 returns exactly 0.
    """
    p = jnp.asarray(p)
    nu = jnp.asarray(nu, dtype=p.dtype)
    q = jnp.minimum(p, 1.0 - p)
    centre = q >= 0.5 - 1e-15
    q_s = jnp.where(centre, 0.25, q)
    log_q = jnp.log(q_s)
    y0 = jnp.maximum(-jsp.ndtri(q_s), 1e-6)
    s0 = jnp.log(y0) * jnp.ones_like(nu)

    def body(_, s):
        y = jnp.exp(s)
        lf = t_logcdf_neg(y, nu)
        phi = lf - log_q
        dphi = -y * jnp.exp(t_logpdf(y, nu) - lf)
        return s - jnp.clip(phi / dphi, -2.0, 2.0)

    s = jax.lax.fori_loop(0, NEWTON_STEPS, body, s0)
    y = jnp.where(centre, 0.0, jnp.exp(s))
    return jnp.where(p < 0.5, -y, y)
