# Walkthrough

## 1. Fit a batch of pairs

```python
import numpy as np
from copulabatch import fit, pseudo_obs, families as F
from copulabatch.sim import pairs

rng = np.random.default_rng(0)
# 1,000 simulated pairs of 500 days from a t copula (rho 0.6, nu 4)
U, V = pairs([("t", (0.6, 4.0))] * 1000, 500, rng)

res = fit(U, V, engine="jax")        # or "numpy", or "reference" (slow, one pair at a time)
res.best_name[:5]                    # AIC choice per pair
j = res.families.index("t")
res.theta[:, j], res.nu[:, j]        # fitted rho and nu for every pair
res.aic.shape                        # (1000, 11)
```

For real returns, build the pseudo-observations per pair from a window of returns:

```python
X = ...                               # (days, assets) returns
U = pseudo_obs(X[-250:, 0])           # ranks / (n + 1)
V = pseudo_obs(X[-250:, 1])
```

or use the command line on a CSV (first column a date):

```bash
python -m copulabatch.cli fit-csv returns.csv --index-col --window 250 > pairs.csv
```

It prints one row per pair: the chosen family, its parameters, Kendall's tau and the lower and upper tail coefficients.

## 2. Read a fit

```python
name = res.best_name[0]
p = res.params(0, name)               # theta, or (rho, nu) for t
F.tau(name, p)                        # implied Kendall's tau
F.tail_dependence(name, p)            # (lambda_L, lambda_U), closed form
F.joint_crash_prob(name, p, 0.01)     # P(both below their own 1% quantile)
```

## 3. Choose an engine

| engine | when |
|---|---|
| `reference` | checking results; a handful of pairs |
| `numpy` | no JAX available; families without t are fast, t is slow (scipy's `stdtrit`) |
| `jax` | the default for batches; compiles once per (families, chunk shape, n), would run on a GPU if jaxlib has one (untested) |

JAX notes:

- Importing `copulabatch.jax_ops` turns on float64 for the process.
- The first call for a new shape compiles (several seconds for all 11 families). Use stable `n` and let the chunking pad N.
- On CPU, XLA runs these element-wise kernels on one core. To use more, set `XLA_FLAGS=--xla_force_host_platform_device_count=16` before importing JAX; `fit_jax` then spreads chunks with `pmap` (`make speed-multicore`).
- For a GPU, install a CUDA build of jaxlib on Linux (for example `pip install "jax[cuda13]"`). Consumer GPUs run float64 far slower than float32.

## 4. Rebuild the results

```bash
make bench           # results/{recovery,agreement,selection,tail,demo}.json and RESULTS.md
make speed           # results/speed_local.json (this machine)
pytest -q            # includes a full rerun compared with the committed JSON
```

`python -m copulabatch.cli bench --split dev --out out` runs the same experiments on the development seeds.
