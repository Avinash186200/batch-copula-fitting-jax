# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-24

### Added

- Eleven bivariate copula families: Gaussian, Student-t, Clayton, Gumbel, Frank, and the 180 / 90 / 270 degree rotations of Clayton and Gumbel. Log-densities written once against a small `Ops` namespace so the scipy, NumPy and JAX engines evaluate the same expressions. Kendall's tau and its inverse, closed-form tail dependence, CDFs (closed form or one-dimensional quadrature), samplers, rank pseudo-observations.
- A branch-free Brent minimiser (golden section plus parabolic steps, the algorithm behind `scipy.optimize.minimize_scalar(method="bounded")`) that runs as NumPy arrays or inside `jax.lax.while_loop` under `vmap`.
- Three engines with one result type: a scipy reference loop (one pair at a time; 2-D L-BFGS-B for the t copula), a vectorised NumPy engine and a JAX `jit(vmap)` engine with chunking, optional `pmap` over several devices, float64 throughout. The t copula is fitted by profile likelihood (outer Brent over log nu, inner Brent over rho).
- A Student-t quantile for JAX: a fixed-length continued fraction for the incomplete beta function and Newton steps in log space; a rank-grid fast path that inverts n/2 grid points per trial nu instead of 2n values.
- Experiments on held-out seeds: parameter recovery against sample size, engine agreement with stated tolerances, AIC family selection with confusion matrices, closed-form vs empirical tail dependence, Gaussian vs t joint-crash probabilities, and a rolling-window demo through a stress regime with a pair VaR / ES scenario.
- A throughput harness (pairs per second against N, per engine, with machine details) and a `fit-csv` command for a real returns file.
- Convergence flags from every solver (`FitResult.converged`, `pairs_hit_iteration_cap` in every results file, a `converged` column in `fit-csv`).
- Tests including a full rerun against the committed results, `constraints.txt` with exact versions, CI on ubuntu (Python 3.11, CPU jaxlib) with a secret scan.
