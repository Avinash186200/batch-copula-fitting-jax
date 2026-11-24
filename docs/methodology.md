# Methodology

The rules behind every number in the README, and what was decided where.

## From returns to pseudo-observations

Sklar's theorem splits a joint distribution into its marginals and a copula. Here the marginals are never modelled: each series is replaced by its ranks divided by n + 1 (`families.pseudo_obs`), the empirical probability integral transform. Fitting a parametric copula to these pseudo-observations by maximum likelihood is the semi-parametric "pseudo-MLE" (Genest, Ghoudi and Rivest, 1995). Dividing by n + 1 keeps every value strictly inside (0, 1), where all the log-densities are finite. Ties get average ranks.

## Families

| family | parameter (bounds) | Kendall's tau | lambda_L | lambda_U |
|---|---|---|---|---|
| Gaussian | rho in [-0.995, 0.995] | 2/pi arcsin(rho) | 0 | 0 |
| Student-t | rho in [-0.995, 0.995], nu in [2, 50] | 2/pi arcsin(rho) | 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho))) | same |
| Clayton | theta in [1e-4, 30] | theta / (theta + 2) | 2^(-1/theta) | 0 |
| Gumbel | theta in [1, 20] | 1 - 1/theta | 0 | 2 - 2^(1/theta) |
| Frank | theta in [-40, 40] | 1 - 4/theta (1 - D_1(theta)) | 0 | 0 |

Rotations of Clayton and Gumbel: with (U', V') from the base copula, rotation 180 is (1 - U', 1 - V') (the survival copula: tails swap), rotation 90 is (1 - U', V') and rotation 270 is (U', 1 - V') (negative dependence, tail in an off-diagonal corner). A rotated density is the base density at the flipped point. Eleven families in all; AIC uses k = 2 for t and k = 1 otherwise.

The log-densities are in `families.py`, written once against a small `Ops` namespace (`numpy` + scipy special functions, or `jax.numpy` + JAX ones), so every engine evaluates the same formulas. They are numerically careful where it matters: Clayton uses `expm1` so theta near 0 does not cancel, Gumbel uses `logaddexp`, Frank uses `expm1` and floors |theta| at 1e-10.

Checks (`tests/test_families.py`): each density matches the mixed finite difference of its CDF, integrates to 1 on a grid, Gaussian and t match scipy's joint densities divided by the marginals, each sampler reproduces its Kendall's tau, the inverse tau map inverts, tail coefficients match their closed forms and the limit of C(q, q)/q.

## Fitting

**One-parameter families.** Brent's bounded minimiser (golden section plus parabolic interpolation, Brent 1973) on the mean negative log-likelihood, absolute tolerance 1e-7 on the parameter. The likelihoods are smooth and, over these brackets, unimodal in practice, which is the case Brent is designed for.

**Student-t.** Profile likelihood: an outer Brent over log nu in [log 2, log 50]; for each trial nu the t scores are computed once and an inner Brent finds the best rho. This reaches the joint optimum up to solver tolerance. The first version alternated nu | rho and rho | nu (coordinate ascent); on development pairs it needed several rounds to come close of the 2-D optimum, because rho and nu are correlated in the likelihood, while the profile needs one t-quantile evaluation per outer step.

**Branch-free Brent** (`brent.py`). Each `if` of the textbook algorithm became an `xp.where` on the solver state, so the same code runs as NumPy arrays (all N problems step together; converged lanes are frozen) or as a scalar problem inside `jax.lax.while_loop`, which `jax.vmap` batches. It follows scipy's `minimize_scalar(method="bounded")` step for step: `tests/test_brent.py` checks the same minimiser and the same number of function evaluations.

**Engines** (`engines.py`):

- `reference`: a Python loop over pairs. scipy `minimize_scalar(bounded)` per one-parameter family; a joint 2-D L-BFGS-B over (rho, log nu) for t, started from the normal-scores correlation and nu = 8. Deliberately a different optimiser for t, so agreement is evidence, not a tautology.
- `numpy`: the shared recipe on (N, n) arrays with the vectorised Brent.
- `jax`: the recipe for one pair, `vmap`-ed and `jit`-compiled, float64. Pairs go in chunks (padded to a power of two, so few shapes compile). With several local devices it uses `pmap`.

**Student-t quantile in JAX** (`jax_special.py`). JAX has no t cdf or ppf, and its `betainc` evaluates a continued fraction per element; on CPU it was far slower in development timing than a fixed-length (24-term) modified-Lentz continued fraction that XLA compiles to straight-line vector code. The quantile uses 8 Newton steps on log F(-e^s) - log q, which is close to linear in the tails. Checked against `scipy.special.stdtrit` to 1e-9 relative over nu in [2, 50] and p in [1e-6, 1 - 1e-6], with one exception: within 1e-7 of p = 0.5 the quantile is close to 0 and the relative error grows (absolute error stays below 4e-10; p = 0.5 itself returns exactly 0). Pseudo-observations i / (n + 1) come no closer to 0.5 than 1 / (2(n + 1)) unless they equal it, so this does not arise for any n below a few million.

**Rank-grid fast path.** Without ties, every row of pseudo-observations is a permutation of i / (n + 1). For each trial nu the t quantiles are then computed on the lower half of that grid (the t is symmetric) and gathered by rank: n/2 quantiles instead of 2n. Ties switch this off automatically (`engines.grid_ranks`); a test checks both paths give the same fit.

## Experiments and seeds

All randomness comes from `numpy.random.default_rng([split, experiment, ...])`. Split 1 (`--split dev`) was used while building; the committed results use split 2 (held-out).

Decided on development runs, before the held-out run:

- the t fitting method (profile likelihood, see above);
- the agreement tolerances (1e-4 on parameters, 1e-6 on log-likelihoods), set above the largest gaps seen on development pairs (6.1e-6 and 6.6e-11 for one-parameter families in the development run; the t reference's flat direction in nu makes nu differ by up to a few tenths of a percent while the log-likelihood agrees to 1e-6);
- the demo's stress block length (first 250 days, then 300, so that more than one rolling window lies wholly inside it);
- the agreement sample (99 pairs, then 54, to keep CI fast) and the per-experiment engine choice (NumPy for one-family recovery fits, JAX elsewhere), both for run time.

The held-out results were generated twice: the second run followed two speed-only changes (an Owen's T formula for the Gaussian diagonal CDF in place of quadrature, which agrees to 1e-8, and JAX instead of NumPy for the t recovery fits, which agree within the tolerances above). No method or setting changed between them in response to held-out numbers.

A third held-out run followed an external review, with these changes, none chosen by looking at which way the numbers moved: the demo's joint-crash threshold went from q = 0.05 to q = 0.04, because with 250-day windows and pseudo-observations rank / 251, "u <= 0.05" selects 12 days (a 4.8% tail) while "u <= 0.04" selects exactly 10; the empirical tail table now averages 100 samples instead of 20; each scenario draw has its own generator; and every fit now records whether its solves converged. Recovery, agreement, selection and the joint-crash experiment did not change; the demo's crash probabilities and the tail table did (the old values are in the git history).

### Recovery

Nine true families at Kendall's tau 0.5 (-0.5 for 90-degree rotations), t with nu = 4; 100 pairs per (family, n), n in {250, 1000, 4000}; each fitted with the true family only. Bias and RMSE of the parameter and of the implied tau; sqrt(n) x RMSE should be roughly flat if the estimator converges at the usual rate.

### Agreement

54 pairs (6 per true family, tau in [0.2, 0.7], t nu in [3, 8]), n = 500, all 11 families by every engine. The test asserts the tolerances above and that every engine makes the same AIC choice.

### Selection

100 pairs per true family at n = 250 and n = 1000, tau ~ U(0.2, 0.7), t nu ~ U(3, 8), all 11 candidates, AIC. Accuracy, tail-class accuracy (the selected family has tail dependence in the same corners as the true one), and the full confusion matrix.

### Tail dependence

(a) Six families at tau 0.5: closed-form lambda vs the empirical estimator at q = 0.01 and 0.05 (mean and standard deviation over 100 samples of n = 20,000), and the exact C(q, q)/q that the estimator targets at finite q. (b) Truth = t copula (rho = sin(pi/4), nu = 4), 100 samples of n = 1,000, each fitted with Gaussian and t; model-implied P(U <= q, V <= q) against the truth (CDFs: Owen's T for the Gaussian diagonal, one-dimensional quadrature for t).

### Demo and the point-in-time rule

A 10-asset, 1,500-day panel: calm days follow a Gaussian copula (equicorrelation 0.3), days 750-1049 a multivariate Clayton copula (theta 2: lambda_L 0.71), with t(5) marginals and higher stress volatility. A window ending at day e uses days [e - 250, e) only, and ranks are computed inside the window (`bench.window_pseudo_obs`; `tests/test_sim.py` rewrites the future and checks earlier windows do not change). Windows are labelled calm-before, mixed, stress (wholly inside the block) or calm-after.

The stress scenario takes the last window wholly inside the stress block. For each of the 45 pairs it draws 100,000 points from four copulas (the true Clayton, the AIC-selected fit, a Gaussian fitted to the same window, and the Gaussian fitted to the last calm window), maps them through the window's empirical marginal quantiles, and computes the 99% VaR and ES of a 50/50 pair portfolio. Same marginals in every row, so the differences come from the copula alone. Each (pair, copula) draw has its own generator, `default_rng([split, 5, 1, pair, model])`, so a change in one fit cannot shift the random numbers of another. The demo's joint-crash probabilities use q = 0.04, so that q times the window length is a whole number of days.

### Convergence

Both Brent paths return a converged flag (in JAX, the flag after the `while_loop`); for t it is the AND of the outer nu solve and the final inner rho solve. `FitResult.converged` holds it per pair and family, `fit-csv` prints it, and every results file records `pairs_hit_iteration_cap` (pairs where any family stopped at the 500-iteration cap). For the scipy reference, the flag is scipy's `success`.

## Reproducibility

`make bench` is deterministic for a given platform and library set. `tests/test_bench.py` reruns it and compares every value with the committed JSON at 1e-4 relative or 1e-6 absolute. The tolerance allows for solver-path differences: different CPUs round the last bits differently, and a last-bit difference can send Brent down a different, equally valid path that stops elsewhere within its tolerance, which in flat likelihoods (nu, Frank theta) moves a parameter by more than xtol = 1e-7. Keys starting with `diff_` (engine-vs-reference gaps near rounding level) are not compared value by value; `within_tolerance` covers them. The committed JSON was produced on Windows with the versions in `constraints.txt`; the Linux check is the CI run on ubuntu (I could not run a local Linux container for this repo, see the README's Limitations).

Throughput (`make speed`) is machine-specific and never checked.
