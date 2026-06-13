# Ledoit-Wolf Shrinkage for Portfolio Optimisation

Implementation of Ledoit-Wolf covariance shrinkage applied to a
Maximum-Sharpe portfolio on 12 NSE stocks. The notebook derives the optimal
shrinkage intensity by hand, checks it against `scikit-learn`, and compares
the resulting portfolio against the standard sample-covariance MPT baseline
on conditioning, weight stability around the March-2023 banking shock, and
out-of-sample returns.

**Stack:** Python, NumPy, pandas, SciPy (`trust-constr`), scikit-learn
(`LedoitWolf`), yfinance, Matplotlib, seaborn.

This was done as a course project for DA6701 - Data Science and AI in Finance at IIT Madras

## Problem

Mean-variance optimisation needs the inverse of a $p \times p$ covariance
matrix $\hat{\Sigma}$. With $T \approx 48$ monthly observations and $p = 12$
assets, the sample covariance is noisy. Its condition number
$\kappa = \lambda_{\max}/\lambda_{\min}$ is both large and unstable across
bootstrap resamples, so small changes in the input data swing the optimised
weights by a lot. The resulting tangency portfolio tends to be concentrated
and does not generalise.

Ledoit-Wolf replaces $\hat{\Sigma}$ with a convex combination of itself and a
scaled identity:

$$
\hat{\Sigma}_{\mathrm{LW}} = (1-\alpha^*)\,\hat{\Sigma} + \alpha^*\,\bar{\lambda}\,\mathbf{I},
\qquad \bar{\lambda} = \mathrm{tr}(\hat{\Sigma})/p
$$

The shrinkage intensity $\alpha^* \in [0,1]$ minimises the expected Frobenius
loss $\mathbb{E}\|\hat{\Sigma}_{\mathrm{LW}} - \Sigma\|_F^2$ in closed form.

## Results

| Metric | Sample MPT | Ledoit-Wolf |
|---|---:|---:|
| Condition number $\kappa$ | 38.0 | 15.2 (2.5x reduction) |
| In-sample Sharpe | 1.147 | 1.172 |
| OOS annualised return | 6.44% | 7.00% |
| OOS annualised volatility | 10.29% | 9.96% |
| OOS Sharpe ratio | -0.006 | 0.050 |
| Maximum drawdown | -6.31% | -5.50% |
| Crisis weight shift $\sum\|\Delta w\|$ | 14.45 pp | 8.03 pp (1.8x more stable) |
| Manual vs sklearn $\alpha^*$ | | match to 8.3e-17 |

Recovered shrinkage intensity: $\alpha^* = 0.1736$. Bootstrap $\kappa$ over
100 resamples of the sample covariance lands in $[37, 328]$ with std 40, so
the unregularised estimator is fragile in addition to being large.

## Universe and Data

12 NSE stocks across four sectors, 5 years of monthly adjusted closes from
Yahoo Finance:

| Sector | Tickers |
|---|---|
| Information Technology | TCS, INFY, HCLTECH |
| Metals and Mining | TATASTEEL, HINDALCO, JINDALSTEEL |
| Banking and Finance | HDFCBANK, ICICIBANK, KOTAKBANK |
| FMCG | HINDUNILVR, ITC, NESTLEIND |

Simple monthly returns $r_t = P_t/P_{t-1} - 1$. The last 12 months are held
out as the test set; the earlier 47 months are used for estimation.

## What the notebook does

The pipeline in `finance_2.ipynb` is split into 12 numbered sections.

1. Imports and matplotlib style (LaTeX if available, mathtext fallback
   otherwise).
2. Data download and train/test split.
3. Sample covariance diagnostics: eigenvalues, $\kappa$, and a 100-sample
   bootstrap distribution of $\kappa$ to show the estimator is unstable.
4. Ledoit-Wolf shrinkage. Fitted with `sklearn.covariance.LedoitWolf`, then
   re-derived from scratch in section 4.1:
   $\hat\delta = \|\hat{\Sigma} - \bar{\lambda}\mathbf{I}\|_F^2$,
   $\hat\beta = T^{-2}\sum_t \|x_t x_t^\top - \hat{\Sigma}\|_F^2$,
   $\alpha^* = \min(1, \hat\beta/\hat\delta)$ from the centred returns. The
   manual value matches scikit-learn to about $10^{-17}$.
5. Max-Sharpe optimisation. Long-only constrained problem
   $\max_w (w^\top\hat\mu - r_f)/\sqrt{w^\top\hat\Sigma w}$ solved with
   `scipy.optimize.minimize` (`trust-constr`), $\sum w_i = 1$, $w_i \geq 0$,
   $r_f = 6.5\%$ p.a. Solved separately on $\hat\Sigma$ and
   $\hat\Sigma_{\mathrm{LW}}$.
6. Side-by-side covariance heatmaps.
7. Eigenvalue spectrum on log scale (with the shrinkage target overlaid) and
   the bootstrap $\kappa$ histogram.
8. Crisis stress test. Two equal-length 24-month windows, one ending Feb
   2023 (pre-crisis) and one ending Jun 2023 (includes the SVB / Signature /
   Credit Suisse months). Both runs use a common $\hat\mu$ taken from the
   full training set so only $\hat\Sigma$ varies, which isolates the
   covariance effect on the weights.
9. Out-of-sample performance on the 12-month hold-out: annualised return and
   vol, Sharpe, max drawdown, cumulative return.
10. Rolling 36-month estimation window with monthly rebalancing, tracking
    $\sum_i |w_{i,t} - w_{i,t-1}|$ over time.
11. Out-of-sample equity curves with a monthly spread panel.
12. Reflection: bias-variance interpretation and what this means in practice.

## Figures

| File | What it shows |
|---|---|
| `viz1_covariance_heatmaps.png` | $\hat\Sigma$ vs $\hat\Sigma_{\mathrm{LW}}$ heatmaps, shared colour scale. |
| `viz_supplementary_eigenvalue_spectrum.png` | Eigenvalue spectrum (log) and bootstrap $\kappa$ histogram. |
| `viz3_weight_sensitivity_crisis.png` | Weights, weight delta, and pre vs post-crisis sensitivity bars. |
| `viz_turnover_analysis.png` | Monthly turnover time series, 36-month rolling window. |
| `viz4_oos_performance.png` | OOS cumulative wealth curves with a monthly spread panel. |

## Findings

**Conditioning:** Shrinkage cuts $\kappa$ from 38.0 to 15.2 by pulling the
eigenvalue spectrum toward $\bar{\lambda}$. The bootstrap distribution of
$\kappa$ confirms the sample estimator is also statistically unstable, not
just large.

**Manual derivation:** The closed form $\alpha^* = \hat\beta/\hat\delta$
coded from the 2004 paper matches scikit-learn to machine precision (about
$10^{-17}$), so the value `LedoitWolf().shrinkage_` returns is the same one
the formula gives by hand on this data.

**Stability around the crisis:** Adding the four banking-shock months
(March to June 2023) to the estimation window shifts the weights by 14.45
pp under the sample estimator and 8.03 pp under LW, so LW is roughly 1.8x
more stable. Banking-sector names take most of the hit under both
estimators, but the per-stock shifts are smaller under LW.

**Out-of-sample:** Over the 12-month hold-out, LW gives a higher annualised
return (7.00% vs 6.44%), lower volatility (9.96% vs 10.29%), a shallower max
drawdown (-5.50% vs -6.31%), and a better Sharpe (0.050 vs -0.006). The
absolute gap is small but the sign is consistent on all four metrics.

**Turnover:** Average monthly turnover is close on this universe (0.186 vs
0.190). The stability advantage of LW shows up mainly when the regime
shifts (the crisis test), not in routine month-to-month rebalancing on a
small, sector-diversified universe.

## Running it

Install:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn yfinance
```

LaTeX rendering is optional. If `latex` and `dvipng` are not on `PATH` the
notebook drops back to matplotlib's Computer Modern mathtext.

Run:

```bash
jupyter notebook finance_2.ipynb
# or
jupyter nbconvert --to notebook --execute finance_2.ipynb
```

Data is pulled live from Yahoo Finance at execution time, so the exact
numbers may drift slightly from the table above as the rolling 5-year window
moves.

## Files

```
ass3/
├── finance_2.ipynb                          # implementation, 26 cells, 12 sections
├── viz1_covariance_heatmaps.png             # section 6
├── viz_supplementary_eigenvalue_spectrum.png# section 7
├── viz3_weight_sensitivity_crisis.png       # section 8
├── viz_turnover_analysis.png                # section 9.1
├── viz4_oos_performance.png                 # section 10
└── README.md
```

## References

- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices. Journal of Multivariate Analysis,
  88(2), 365-411.
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
