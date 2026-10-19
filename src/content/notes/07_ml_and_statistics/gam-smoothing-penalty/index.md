---
title: "GAMs and the smoothing penalty: what lambda and effective degrees of freedom mean"
slug: gam-smoothing-penalty
date: 2019-12-09
topic: ML & Statistics
tags: [university-lab, gam, splines, smoothing, regularization, mgcv]
summary: "How a generalized additive model fits smooth curves with penalized splines, what the smoothing parameter and the effective degrees of freedom tell you, and why the default mgcv fit is not an unpenalized one."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Generalized Additive Models: An Introduction with R (2nd ed.)"
    author: "Simon N. Wood"
    url: "https://doi.org/10.1201/9781315370279"
    type: book
  - title: "The Elements of Statistical Learning, Sections 5.4–5.5 and 9.1"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book


status: draft
---

A generalized additive model (GAM) is a regression where each predictor gets its own smooth curve instead of a straight line. You don't have to guess the shape in advance (a seasonal bump, a threshold, a plateau); the data decide, within limits set by a smoothing penalty. In a lab I modelled weekly mortality with a smooth function of the week of the year, and the part worth remembering is what that penalty does and how it's chosen.

## The model

For a response $y_i$ and predictors $x_{i1}, \dots, x_{ip}$,

$$
g\big(E[y_i]\big) = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \dots + f_p(x_{ip}),
$$

where $g$ is a link function (the identity for a Gaussian response) and each $f_j$ is a smooth function. Some terms can stay linear. My first model was $\text{mortality}_i = \beta_0 + \beta_1\,\text{year}_i + f(\text{week}_i) + \varepsilon_i$ with $\varepsilon_i \sim N(0, \sigma^2)$.

"Additive" is the key restriction. Each predictor's effect can be curved, but effects don't interact unless you add an explicit interaction term. That keeps every $f_j$ easy to plot and interpret.

## Splines plus a penalty

Each $f_j$ is written as a spline, a weighted sum of $k$ basis functions: $f(x) = \sum_{m=1}^{k} b_m(x)\,\gamma_m$. With a generous $k$ the spline could follow every wiggle of the noise. So instead of plain least squares, `mgcv` minimizes a penalized fit:

$$
\sum_i \big(y_i - \hat y_i\big)^2 + \lambda \int f''(x)^2 \, dx .
$$

- The first term rewards fitting the data.
- $\int f''(x)^2\,dx$ measures wiggliness: the total squared curvature.
- $\lambda \ge 0$ is the **smoothing parameter**. $\lambda \to 0$ gives an almost unpenalized spline that interpolates noise. $\lambda \to \infty$ forces $f'' = 0$, so $f$ becomes a straight line.

So $k$ sets the *maximum* flexibility and $\lambda$ decides how much of it is used. A common piece of advice is: make $k$ large enough and let $\lambda$ do the work.

## Effective degrees of freedom

A penalized spline with $k$ basis functions doesn't really use $k$ parameters. The **effective degrees of freedom** (edf) measure how many it actually uses. For a linear smoother with $\hat y = S_\lambda y$, the edf is the trace of the smoother matrix, $\operatorname{tr}(S_\lambda)$. An edf near 1 means "basically a straight line"; a large edf means a wiggly curve. In the lab's summary tables, raising the penalty made the edf of `s(Week)` drop, which is exactly what should happen.

## Choosing lambda

By default (with `method = "GCV.Cp"`, which I used), `mgcv` picks $\lambda$ by generalized cross-validation,

$$
\text{GCV}(\lambda) = \frac{n \sum_i (y_i - \hat y_i)^2}{\big(n - \operatorname{tr}(S_\lambda)\big)^2},
$$

which approximates leave-one-out error without refitting $n$ times. (Today `method = "REML"` is often recommended because it's more stable.) You can override it by passing `sp` directly.

### The mistake in my plots

I put three fitted curves side by side and labelled the default one "Without penalty", next to "Low penalty" (`sp = 0.5`) and "High penalty" (`sp = 10`). The default fit isn't unpenalized. It uses whatever $\lambda$ GCV selected, which is usually a moderate, non-zero value. A truly unpenalized fit would be `sp = 0` and would wiggle far more. And whether `sp = 0.5` counts as "low" depends on the scale of the data and the basis. Compare edf values, not raw `sp` numbers.

```r
library(mgcv)
set.seed(5)
d <- data.frame(week = rep(1:52, 6), year = rep(1995:2000, each = 52))
d$y <- 1700 + 400 * cos(2 * pi * d$week / 52) + rnorm(nrow(d), sd = 80)

fit_gcv   <- gam(y ~ year + s(week, k = 30), data = d, method = "GCV.Cp")  # lambda picked by GCV
fit_small <- gam(y ~ year + s(week, k = 30), data = d, sp = 1e-4)          # almost no penalty
fit_big   <- gam(y ~ year + s(week, k = 30), data = d, sp = 1e5)           # heavy penalty
fit_gcv$sp                                     # about 0.028: GCV's lambda is not zero
info <- function(f) c(edf_week = round(sum(f$edf) - 2, 1),   # minus intercept and year
                      deviance = round(deviance(f)))
rbind(gcv = info(fit_gcv), small = info(fit_small), big = info(fit_big))
# edf about 8.3 (GCV), 25.8 (almost unpenalized), 1.0 (heavy penalty: a straight line)
```

Note the deviance column: the almost unpenalized fit always has the lowest training deviance. That doesn't make it the best model. It's the same story as polynomial degree or tree depth.

## Reading the result

Two more lessons from the lab:

- **Look at the residuals against other variables.** After fitting year and week, the residuals still went up and down with the influenza counts. That was a clear hint to add `s(Influenza)`, and doing so improved the fit.
- **A small p-value for a smooth term** means the curve is clearly not flat. It doesn't tell you the shape, so always plot the fitted $f_j$ with its confidence band.

## Key takeaways

- A GAM replaces each linear term with a smooth function and keeps the effects additive.
- The fit balances squared error against $\lambda \int f''^2$; $k$ caps flexibility and $\lambda$ chooses how much is used.
- Effective degrees of freedom ($\operatorname{tr} S_\lambda$) are the honest measure of complexity; edf $\approx 1$ means a line.
- `mgcv`'s default fit is penalized with a $\lambda$ chosen by GCV or REML, not unpenalized.

## Resources

- 732A99 Machine Learning, Linköping University (course lab, autumn 2019)
- Simon N. Wood, *Generalized Additive Models: An Introduction with R*, 2nd ed., CRC Press (2017): https://doi.org/10.1201/9781315370279
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 5.4 (smoothing splines and effective degrees of freedom), Section 5.5 (choosing $\lambda$) and Section 9.1 (generalized additive models): https://hastie.su.domains/ElemStatLearn/
- R package `mgcv` documentation (`gam`, `s`, `gam.selection`): https://cran.r-project.org/package=mgcv
