# Results

Synthetic data only. Rebuilt by `make bench` (sections 1-5, checked against a fresh run in CI) and `make speed` (section 6, machine-specific). Seeds: holdout split.

## 1. Parameter recovery

True Kendall's tau = 0.5 (negative for the 90-degree rotations), t copula nu = 4.0. Each row: 100 independent simulated pairs of length n, fitted with the true family (engine: numpy, and jax for t). theta is rho for Gaussian and t.

| family | n | pairs | theta true | theta bias | theta RMSE | sqrt(n) x RMSE | tau bias | tau RMSE | nu (t only) |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | 250 | 100 | 0.7071 | +0.0046 | 0.0343 | 0.54 | +0.0049 | 0.0312 |  |
| gaussian | 1000 | 100 | 0.7071 | -0.0016 | 0.0152 | 0.48 | -0.0013 | 0.0136 |  |
| gaussian | 4000 | 100 | 0.7071 | +0.0007 | 0.0081 | 0.51 | +0.0007 | 0.0073 |  |
| t | 250 | 100 | 0.7071 | +0.0032 | 0.0379 | 0.60 | +0.0038 | 0.0340 | median 4.07 (IQR 3.23-5.42), 0 at bound |
| t | 1000 | 100 | 0.7071 | +0.0001 | 0.0199 | 0.63 | +0.0004 | 0.0179 | median 3.88 (IQR 3.38-4.40), 0 at bound |
| t | 4000 | 100 | 0.7071 | -0.0021 | 0.0108 | 0.68 | -0.0018 | 0.0097 | median 3.90 (IQR 3.74-4.25), 0 at bound |
| clayton | 250 | 100 | 2.0000 | +0.0704 | 0.2415 | 3.82 | +0.0071 | 0.0287 |  |
| clayton | 1000 | 100 | 2.0000 | +0.0167 | 0.1068 | 3.38 | +0.0017 | 0.0132 |  |
| clayton | 4000 | 100 | 2.0000 | -0.0088 | 0.0670 | 4.23 | -0.0012 | 0.0085 |  |
| clayton_180 | 250 | 100 | 2.0000 | +0.0366 | 0.2336 | 3.69 | +0.0029 | 0.0289 |  |
| clayton_180 | 1000 | 100 | 2.0000 | +0.0184 | 0.1164 | 3.68 | +0.0019 | 0.0144 |  |
| clayton_180 | 4000 | 100 | 2.0000 | +0.0040 | 0.0570 | 3.61 | +0.0004 | 0.0071 |  |
| gumbel | 250 | 100 | 2.0000 | +0.0310 | 0.1373 | 2.17 | +0.0055 | 0.0329 |  |
| gumbel | 1000 | 100 | 2.0000 | +0.0022 | 0.0640 | 2.02 | +0.0000 | 0.0158 |  |
| gumbel | 4000 | 100 | 2.0000 | +0.0062 | 0.0338 | 2.14 | +0.0014 | 0.0084 |  |
| gumbel_180 | 250 | 100 | 2.0000 | +0.0211 | 0.1352 | 2.14 | +0.0031 | 0.0328 |  |
| gumbel_180 | 1000 | 100 | 2.0000 | +0.0115 | 0.0654 | 2.07 | +0.0024 | 0.0158 |  |
| gumbel_180 | 4000 | 100 | 2.0000 | -0.0001 | 0.0299 | 1.89 | -0.0001 | 0.0075 |  |
| frank | 250 | 100 | 5.7363 | -0.0291 | 0.5530 | 8.74 | -0.0031 | 0.0307 |  |
| frank | 1000 | 100 | 5.7363 | -0.0228 | 0.2552 | 8.07 | -0.0016 | 0.0140 |  |
| frank | 4000 | 100 | 5.7363 | +0.0142 | 0.1329 | 8.40 | +0.0007 | 0.0073 |  |
| clayton_90 | 250 | 100 | 2.0000 | +0.0433 | 0.2341 | 3.70 | -0.0038 | 0.0280 |  |
| clayton_90 | 1000 | 100 | 2.0000 | -0.0062 | 0.1356 | 4.29 | +0.0014 | 0.0170 |  |
| clayton_90 | 4000 | 100 | 2.0000 | -0.0039 | 0.0606 | 3.84 | +0.0006 | 0.0076 |  |
| gumbel_90 | 250 | 100 | 2.0000 | +0.0141 | 0.1229 | 1.94 | -0.0017 | 0.0301 |  |
| gumbel_90 | 1000 | 100 | 2.0000 | +0.0070 | 0.0665 | 2.10 | -0.0012 | 0.0166 |  |
| gumbel_90 | 4000 | 100 | 2.0000 | -0.0007 | 0.0294 | 1.86 | +0.0003 | 0.0073 |  |

Pairs whose solver stopped at the iteration cap: 0 of 2700.

## 2. Batch engines vs the scipy reference

54 pairs (mixed true families), n = 500, all 11 families fitted by each engine. Solver tolerance xtol = 1e-07. Tolerances asserted in tests: |d theta| <= 0.0001 and |d loglik| <= 1e-06 for one-parameter families; for t, |d rho| <= 0.0001 and loglik_batch >= loglik_ref - 1e-06.

Pairs at the iteration cap: numpy 0, jax 0; scipy reference pairs without a success flag: 1 (of 54).

| engine | max abs d theta (1-param) | max abs d loglik (1-param) | max abs d rho (t) | max rel d nu (t) | t loglik gain vs ref (min / max) | same AIC choice | within tolerance |
|---|---|---|---|---|---|---|---|
| numpy | 1.4e-06 | 6.6e-11 | 1.3e-05 | 4.0e-03 | -1.1e-07 / +2.6e-06 | 54 / 54 | True |
| jax | 7.5e-06 | 2.1e-10 | 1.3e-05 | 4.0e-03 | -1.1e-07 / +2.6e-06 | 54 / 54 | True |

## 3. Family selection by AIC

100 pairs per true family, tau drawn from U(0.2, 0.7) (negated for rotations), t nu from U(3.0, 8.0); candidates: all 11 families (engine: jax). Tail class = the set of corners with tail dependence.

| n | pairs | at iteration cap | accuracy | tail-class accuracy | Gauss | t | Clay | Clay180 | Gumb | Gumb180 | Frank | Clay90 | Gumb90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 900 | 0 | 0.810 | 0.902 | 0.69 | 0.71 | 0.86 | 0.92 | 0.77 | 0.76 | 0.90 | 0.89 | 0.79 |
| 1000 | 900 | 0 | 0.976 | 0.988 | 0.94 | 0.97 | 1.00 | 0.98 | 0.99 | 0.98 | 0.99 | 0.99 | 0.94 |

Confusion matrix, n = 250 (rows: true, columns: selected)

| true \ selected | Gauss | t | Clay | Clay180 | Clay90 | Clay270 | Gumb | Gumb180 | Gumb90 | Gumb270 | Frank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gauss | 69 | 9 | 1 | . | . | . | 8 | 6 | . | . | 7 |
| t | 14 | 71 | . | . | . | . | 4 | 9 | . | . | 2 |
| Clay | . | . | 86 | . | . | . | . | 13 | . | . | 1 |
| Clay180 | . | . | . | 92 | . | . | 8 | . | . | . | . |
| Gumb | 3 | 4 | . | 16 | . | . | 77 | . | . | . | . |
| Gumb180 | 3 | 7 | 14 | . | . | . | . | 76 | . | . | . |
| Frank | 5 | 2 | . | 1 | . | . | . | 2 | . | . | 90 |
| Clay90 | 1 | . | . | . | 89 | . | . | . | . | 10 | . |
| Gumb90 | 3 | 6 | . | . | . | 10 | . | . | 79 | . | 2 |

Confusion matrix, n = 1000 (rows: true, columns: selected)

| true \ selected | Gauss | t | Clay | Clay180 | Clay90 | Clay270 | Gumb | Gumb180 | Gumb90 | Gumb270 | Frank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gauss | 94 | 6 | . | . | . | . | . | . | . | . | . |
| t | . | 97 | . | . | . | . | 1 | 2 | . | . | . |
| Clay | . | . | 100 | . | . | . | . | . | . | . | . |
| Clay180 | . | . | . | 98 | . | . | 2 | . | . | . | . |
| Gumb | . | . | . | 1 | . | . | 99 | . | . | . | . |
| Gumb180 | . | . | 2 | . | . | . | . | 98 | . | . | . |
| Frank | 1 | . | . | . | . | . | . | . | . | . | 99 |
| Clay90 | . | . | . | . | 99 | . | . | . | . | 1 | . |
| Gumb90 | 1 | 1 | . | . | . | 4 | . | . | 94 | . | . |

## 4. Tail dependence

Closed-form coefficients vs the empirical estimator lambda_L(q) = #{u <= q, v <= q} / (n q) (and the upper analogue). 'exact at q' is C(q, q) / q from the copula itself: what the estimator converges to at that q as n grows, which is not the limit lambda_L.

| family (tau 0.5) | n | samples | lambda_L | lambda_U | emp L q=0.01 mean (sd) | exact at 0.01 | emp L q=0.05 mean (sd) | exact at 0.05 | emp U q=0.01 | emp U q=0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| gaussian | 20000 | 100 | 0.000 | 0.000 | 0.273 (0.037) | 0.273 | 0.399 (0.020) | 0.398 | 0.280 | 0.398 |
| t | 20000 | 100 | 0.397 | 0.397 | 0.437 (0.044) | 0.432 | 0.480 (0.022) | 0.482 | 0.434 | 0.485 |
| clayton | 20000 | 100 | 0.707 | 0.000 | 0.690 (0.064) | 0.707 | 0.712 (0.029) | 0.708 | 0.030 | 0.136 |
| gumbel | 20000 | 100 | 0.000 | 0.586 | 0.152 (0.025) | 0.148 | 0.291 (0.019) | 0.289 | 0.589 | 0.598 |
| clayton_180 | 20000 | 100 | 0.000 | 0.707 | 0.029 (0.014) | 0.029 | 0.138 (0.011) | 0.136 | 0.706 | 0.712 |
| frank | 20000 | 100 | 0.000 | 0.000 | 0.054 (0.017) | 0.054 | 0.226 (0.015) | 0.225 | 0.055 | 0.223 |

Joint crashes. Truth: t copula, rho = 0.7071, nu = 4.0. 100 samples of n = 1000; each fitted with Gaussian and t. AIC prefers t in 100 / 100. True lambda = 0.397; fitted t lambda (mean) = 0.395; Gaussian lambda = 0. Pairs at the iteration cap: 0. Mean fitted rho: Gaussian 0.6987, t 0.7052; median fitted nu 3.94.

| q | true P(U<=q, V<=q) | if independent | empirical (mean) | Gaussian fit (mean) | t fit (mean) | Gaussian / true | t / true |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.00432 | 0.00010 | 0.00419 | 0.00266 | 0.00433 | 0.616 | 1.001 |
| 0.05 | 0.02409 | 0.00250 | 0.02364 | 0.01956 | 0.02409 | 0.812 | 1.000 |

## 5. Demo: rolling fits through a stress regime

10 assets, 1500 days. Days 750-1049 are a stress regime (Clayton copula, theta 2.0, lambda_L 0.707; vol 0.025); other days are calm (Gaussian copula, rho 0.3; vol 0.01); t(5) marginals. Rolling window 250 days, step 25: 51 windows x 45 pairs = 2295 fits (engine: jax; 0 at the iteration cap). Joint-crash probability at q = 0.04 (q x window is a whole number of days, so the empirical count uses exactly that share of each asset's days): true value 0.0050 (calm) and 0.0283 (stress).

| window phase | windows | share of pairs with a lower-tail family | mean lambda_L (selected) | crash prob: empirical | crash prob: selected family | crash prob: Gaussian fit |
|---|---|---|---|---|---|---|
| calm_before | 21 | 0.186 | 0.036 | 0.0048 | 0.0053 | 0.0050 |
| mixed | 18 | 0.965 | 0.481 | 0.0221 | 0.0201 | 0.0109 |
| stress | 3 | 1.000 | 0.677 | 0.0279 | 0.0271 | 0.0136 |
| calm_after | 9 | 0.109 | 0.020 | 0.0047 | 0.0051 | 0.0053 |

Stress scenario: 50/50 pair portfolio, empirical marginals of the stress window ending day 1050, 100,000 copula draws per pair, averaged over 45 pairs. 99% one-day VaR and ES as a fraction of the position.

| copula | VaR 99% | ES 99% |
|---|---|---|
| true stress copula | 0.0468 | 0.0523 |
| selected on stress window | 0.0464 | 0.0520 |
| gaussian on stress window | 0.0420 | 0.0478 |
| gaussian on last calm window | 0.0360 | 0.0414 |

## 6. Throughput (machine-specific, never checked by tests)

### laptop-cpu-16dev

Windows-10-10.0.26200-SP0; Intel64 Family 6 Model 170 Stepping 4, GenuineIntel; 22 logical CPUs; Python 3.11.9; JAX 0.10.2 backend cpu (cpu, 16 device(s); XLA_FLAGS --xla_force_host_platform_device_count=16). All 11 families + AIC per pair, n = 500. Batch engines: fastest of the timed calls after one untimed warm-up call (the first call includes JAX compilation). Reference loop: one timed call, no warm-up.

| engine | N pairs | pairs / s | seconds | timed calls | first call (s) |
|---|---|---|---|---|---|
| jax | 10 | 32.5 | 0.308 | 2 | 5.28 |
| jax | 100 | 335.2 | 0.298 | 2 | 0.29 |
| jax | 1000 | 400.0 | 2.500 | 2 | 7.58 |
| jax | 4000 | 374.0 | 10.694 | 2 | 16.24 |

### laptop-cpu

Windows-10-10.0.26200-SP0; Intel64 Family 6 Model 170 Stepping 4, GenuineIntel; 22 logical CPUs; Python 3.11.9; JAX 0.10.2 backend cpu (cpu, 1 device(s)). All 11 families + AIC per pair, n = 500. Batch engines: fastest of the timed calls after one untimed warm-up call (the first call includes JAX compilation). Reference loop: one timed call, no warm-up.

| engine | N pairs | pairs / s | seconds | timed calls | first call (s) |
|---|---|---|---|---|---|
| reference | 10 | 5.5 | 1.803 | 1 | 1.80 |
| reference | 30 | 5.4 | 5.574 | 1 | 5.57 |
| reference | 100 | 5.3 | 18.876 | 1 | 18.88 |
| numpy | 10 | 19.6 | 0.511 | 2 | 0.53 |
| numpy | 100 | 22.2 | 4.510 | 2 | 4.59 |
| numpy | 1000 | 20.5 | 48.799 | 2 | 73.13 |
| jax | 10 | 64.4 | 0.155 | 2 | 4.21 |
| jax | 100 | 124.1 | 0.805 | 2 | 4.08 |
| jax | 1000 | 257.0 | 3.891 | 2 | 9.58 |
| jax | 4000 | 246.4 | 16.232 | 2 | 20.39 |
