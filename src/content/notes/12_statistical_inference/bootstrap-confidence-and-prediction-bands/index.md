---
title: "Bootstrap bands: confidence vs prediction, parametric vs non-parametric"
slug: bootstrap-confidence-and-prediction-bands
date: 2019-12-08
topic: Statistical Inference
tags: [university-lab, bootstrap, uncertainty, confidence-intervals, prediction-intervals, regression]
summary: "How to put uncertainty bands around any fitted curve with the bootstrap, and the difference between a confidence band (where the curve is) and a prediction band (where new points land)."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "An Introduction to the Bootstrap"
    author: "Bradley Efron, Robert J. Tibshirani"
    type: book
  - title: "The Elements of Statistical Learning, Section 8.2 (The Bootstrap and Maximum Likelihood)"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book
  - title: "boot: Bootstrap Functions (R package)"
    author: "Angelo Canty, Brian Ripley"
    url: "https://cran.r-project.org/package=boot"
    type: docs


status: draft
---

For a straight-line regression there are textbook formulas for confidence and prediction intervals. For a regression tree, a spline or almost anything else, there aren't. The bootstrap fills that gap: refit the model many times on data that "could have happened", and look at how much the fitted curve moves. (For bootstrap intervals of a single number, like a mean, see [bootstrap and jackknife confidence intervals](/notes/bootstrap-and-jackknife-confidence-intervals/).) I used it in a lab to put bands around a regression tree, and the two choices that matter are which kind of band you want and how you generate the fake datasets.

## Two different questions

- **Confidence band**: where is the *true regression curve* $f(x) = E[Y \mid X = x]$? It only reflects uncertainty in the fitted model.
- **Prediction band**: where will a *new observation* $Y_{new}$ at $x$ land? It has the model uncertainty plus the noise around the curve, so it's always wider.

If the noise is $\varepsilon \sim N(0, \sigma^2)$, then roughly

$$
\operatorname{Var}(Y_{new} - \hat f(x)) = \operatorname{Var}(\hat f(x)) + \sigma^2 .
$$

The first term shrinks with more data. The second doesn't. So a prediction band never collapses to the curve, however much data you have. A sanity check I used: about 95% of the observed points should fall inside a 95% prediction band (in my lab 2 of 48 fell outside, which is fine). Nothing like that holds for a confidence band.

## Two ways to make bootstrap datasets

**Non-parametric bootstrap.** Resample the rows $(x_i, y_i)$ with replacement, refit, and predict on a grid of $x$ values. Repeat $B$ times (say 1,000). At each grid point, the 2.5% and 97.5% quantiles of the $B$ predictions give a pointwise 95% confidence band. No distribution assumed. The cost is that some resamples miss whole regions of $x$, so bands can be bumpy, especially for a step-shaped model like a tree.

**Parametric bootstrap.** Assume a model for the noise, for example $y_i \sim N(\hat f(x_i), \hat\sigma^2)$, where $\hat f$ is the fit to the real data and $\hat\sigma$ the residual standard deviation. Keep the $x_i$ fixed, simulate new $y_i$ from that model, refit, predict. This gives smoother bands, but only if the noise model is roughly right. With skewed residuals (which I saw in the lab's histogram), the normal assumption is doubtful, and the smooth band can look more trustworthy than it really is.

For a **prediction band** with the parametric bootstrap, add fresh noise to each bootstrap prediction: $\hat f^{*b}(x) + \varepsilon^{*}$ with $\varepsilon^{*} \sim N(0, \hat\sigma^2)$. Then take quantiles as before.

```r
library(boot)
set.seed(1)
d <- data.frame(x = sort(runif(60, 0, 10)))
d$y <- sin(d$x) + rnorm(60, sd = 0.4)
grid <- data.frame(x = seq(0, 10, length.out = 50))
fit_curve <- function(data) predict(lm(y ~ poly(x, 5), data = data), newdata = grid)

# Non-parametric: resample rows
np <- boot(d, function(data, i) fit_curve(data[i, ]), R = 1000)

# Parametric: keep x, simulate y from the fitted model
fit0 <- lm(y ~ poly(x, 5), data = d)
s <- sigma(fit0)
gen <- function(data, mle) { data$y <- rnorm(nrow(data), fitted(fit0), s); data }
pb_conf <- boot(d, fit_curve, R = 1000, sim = "parametric", ran.gen = gen)
pb_pred <- boot(d, function(data) fit_curve(data) + rnorm(nrow(grid), 0, s),
                R = 1000, sim = "parametric", ran.gen = gen)

band  <- function(b) apply(b$t, 2, quantile, probs = c(0.025, 0.975))
width <- function(b) mean(diff(band(b)))
c(nonparam_conf = width(np), param_conf = width(pb_conf), param_pred = width(pb_pred))
# about 0.69, 0.51 and 1.64: the prediction band is roughly 3 times wider
```

## Things that tripped me up

- **Pointwise vs simultaneous.** Taking quantiles at each $x$ separately gives a *pointwise* band: at any single $x$ the true curve is inside with about 95% probability. The whole curve being inside everywhere at once is less likely. `boot::envelope()` returns both: `$point` is pointwise and `$overall` is a wider simultaneous band.
- **`boot.ci()` is for one statistic.** When the statistic returns a whole vector of predictions, `boot.ci()` only looks at one element (the first, by default). For bands you want quantiles across all columns of `b$t`.
- **Wide bands aren't a bug.** Bumpy, wide bands for a tree on 48 points mostly say "this model is unstable on this little data", which is useful to know.

## Key takeaways

- A confidence band is about the curve; a prediction band is about new points and adds the noise variance.
- Non-parametric bootstrap resamples rows and assumes little; parametric bootstrap simulates from a fitted noise model and is smoother but only as good as that model.
- For prediction bands, add simulated noise to each bootstrap prediction.
- Quantiles per $x$ give pointwise bands; simultaneous bands are wider.

## Resources

- 732A99 Machine Learning, Linköping University (course lab, autumn 2019)
- Bradley Efron and Robert J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall (1993), chapters 6 and 9 (parametric bootstrap and regression)
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 8.2: https://hastie.su.domains/ElemStatLearn/
- R package `boot` documentation (`boot`, `envelope`, `boot.ci`): https://cran.r-project.org/package=boot
