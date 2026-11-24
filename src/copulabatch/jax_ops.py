"""The ``Ops`` namespace for JAX. Importing this module switches JAX to float64.

Copula likelihoods need double precision: Clayton and Gumbel terms such as u^-theta reach
1e100 near the corners, and the fits are compared with the scipy reference to 1e-4
(parameters) and 1e-6 (log-likelihood). float64 is kept; its GPU cost was not measured
(consumer GPUs run float64 far slower than float32, and no GPU run is reported here).
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy import special as jsp

from .families import Ops
from .jax_special import t_ppf

JAX_OPS = Ops(jnp, jsp.ndtri, jsp.gammaln, t_ppf)
