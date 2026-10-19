---
title: "Gaussian process regression: the posterior via Cholesky, length scales, and periodic kernels"
slug: gaussian-process-regression
date: 2020-10-10
topic: Bayesian Methods
tags: [university-lab, gaussian-processes, kernels, bayesian-inference, nonparametric-regression]
summary: "How a Gaussian process turns a kernel into a prior over functions, how to compute the posterior mean and variance stably with a Cholesky factorisation, what the hyperparameters do, and how to build a kernel for seasonal data."
sources:
  - title: "732A96 Advanced Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Gaussian Processes for Machine Learning, Chapters 2, 4 and 5"
    author: "Carl Edward Rasmussen, Christopher K. I. Williams"
    url: "https://gaussianprocess.org/gpml/"
    type: book
  - title: "kernlab: Kernel-based machine learning lab (gausspr)"
    author: "Alexandros Karatzoglou, Alex Smola, Kurt Hornik"
    url: "https://cran.r-project.org/package=kernlab"
    type: docs


status: draft
---

A Gaussian process (GP) is a way to put a prior directly on functions, without choosing a parametric form like "quadratic". You only say how similar the function's values at two inputs should be, through a kernel, and Bayes does the rest. In the last Advanced Machine Learning lab I implemented the GP posterior by hand, then used `kernlab` on six years of daily temperatures. The math is short; the traps were about scales.

## A prior over functions

$f \sim \mathcal{GP}(0, k)$ means: for any finite set of inputs $x_1, \dots, x_n$, the values $f(x_1), \dots, f(x_n)$ are jointly normal with mean 0 and covariance matrix $K_{ij} = k(x_i, x_j)$. The kernel is everything. The most common one is the squared exponential:

$$
k(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right),
$$

where $\sigma_f^2$ is the prior variance of $f$ (how far it swings from 0) and $\ell$ is the length scale (how far apart two inputs can be and still have strongly correlated function values). Small $\ell$ gives wiggly functions; large $\ell$ gives smooth ones.

## The posterior

With observations $y_i = f(x_i) + \varepsilon_i$, $\varepsilon_i \sim N(0, \sigma_n^2)$, the posterior of $f$ at new inputs $X_*$ is normal with

$$
\bar f_* = K_*^\top (K + \sigma_n^2 I)^{-1} y, \qquad \text{cov}(f_*) = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*,
$$

where $K = k(X, X)$ is the training covariance, $K_* = k(X, X_*)$ the train-test covariance and $K_{**} = k(X_*, X_*)$. Two things I find helpful to notice: the mean is a weighted sum of kernel functions centred at the data, and the variance **doesn't depend on $y$** at all, only on where the data are.

Inverting the matrix directly is slow and numerically fragile. Algorithm 2.1 in Rasmussen and Williams uses the Cholesky factor $L$ of $K + \sigma_n^2 I$ instead, and gives the log marginal likelihood almost for free:

```r
se_kernel <- function(x1, x2, sigma_f = 1, ell = 0.3)
  sigma_f^2 * exp(-0.5 * outer(x1, x2, "-")^2 / ell^2)

gp_posterior <- function(x, y, x_star, sigma_n, ...) {
  K  <- se_kernel(x, x, ...); Ks <- se_kernel(x, x_star, ...)
  L  <- t(chol(K + diag(sigma_n^2, length(x))))     # lower-triangular factor
  alpha <- backsolve(t(L), forwardsolve(L, y))       # (K + sigma_n^2 I)^-1 y
  v <- forwardsolve(L, Ks)
  list(mean = drop(t(Ks) %*% alpha),
       var  = diag(se_kernel(x_star, x_star, ...)) - colSums(v^2),
       log_marg_lik = -0.5 * sum(y * alpha) - sum(log(diag(L))) - length(x) / 2 * log(2 * pi))
}

x <- c(-0.9, -0.5, 0, 0.3, 0.7); y <- c(0.5, -0.2, -0.8, 0.4, -0.5)
grid <- seq(-1, 1, length.out = 101)
for (ell in c(0.3, 1)) {
  post <- gp_posterior(x, y, grid, sigma_n = 0.1, sigma_f = 1, ell = ell)
  band <- 1.96 * sqrt(post$var)                      # pointwise 95% band for f
  cat("ell =", ell, " max half-width:", round(max(band), 2),
      " log marginal likelihood:", round(post$log_marg_lik, 2), "\n")
}
# ell = 0.3  max half-width: 1.5   log marginal likelihood: -5.76
# ell = 1    max half-width: 0.4   log marginal likelihood: -38.2
```

## What the hyperparameters do

When I updated the prior with one point, then two, then five, the band pinched in near each observation (down to about $\pm 2\sigma_n$) and stayed wide far from the data, back to the prior's $\pm 1.96\sigma_f$. Increasing $\ell$ from 0.3 to 1 made the mean much smoother.

The snippet shows the catch. With $\ell = 1$ the band is *narrower*, but the model can't bend enough to follow data that change sign within a short range, so it's confidently wrong, and the log marginal likelihood is far lower ($-38$ vs $-6$). Narrow bands aren't a sign of a good fit. The marginal likelihood $p(y \mid \sigma_f, \ell, \sigma_n)$ is the standard way to compare or optimise hyperparameters.

## Scales: the trap I fell into

Hyperparameters only mean something relative to the units of $x$ and $y$. "$\ell = 0.2$" is tiny if $x$ is a day number from 1 to 2190, and reasonable if $x$ has been standardised. `kernlab::gausspr` standardises both inputs and outputs by default (`scaled = TRUE`), so its $\ell$ lives on the standardised scale.

In my lab report I computed the posterior variance myself (because `gausspr` didn't return it reliably) but mixed the scales: standardised temperatures, raw day numbers for time, $\ell = 0.2$ and a noise level taken from residuals in raw degrees. The resulting band wasn't the band of the model `gausspr` had fitted. To be consistent, standardise $x$ and $y$ the same way the package does, compute mean and variance on that scale, then transform back (multiply standard deviations by $\text{sd}(y)$).

I also drew the band as mean $\pm 1$ standard deviation while calling it a 95% band. A pointwise 95% band is mean $\pm 1.96$ standard deviations.

## Kernels for seasonal data

With temperature as a function of time, an SE kernel on the day number captures slow changes across years, but has no idea that 1 January 2011 is similar to 1 January 2010. Using day-of-year as the input does the opposite: it forces every year to have the same curve. A kernel can express both at once:

$$
k(x, x') = \sigma_f^2 \exp\!\left(-\frac{2\sin^2(\pi |x - x'| / d)}{\ell_1^2}\right) \exp\!\left(-\frac{(x - x')^2}{2\ell_2^2}\right),
$$

where $d$ is the period (one year), $\ell_1$ controls how smooth the function is within a period, and $\ell_2$ controls how quickly the correlation between the same day in different years fades. The first factor alone says "every year is identical"; multiplying by the second (a "locally periodic" kernel) lets the yearly pattern drift slowly. In the lab this model sat between the other two: it followed the seasonal shape and still allowed year-to-year differences. Building kernels by multiplying and adding simpler ones is one of the nicest things about GPs.

`gausspr` also does GP *classification* (with a logistic-type link and an approximate posterior). In the lab, going from two to all four features of a banknote dataset raised test accuracy from 0.925 to 0.995 with default settings, which mostly showed how easy it is to use rather than anything deep.

## Key takeaways

- A GP is a prior over functions defined by a kernel; the SE kernel has a signal variance $\sigma_f^2$ and a length scale $\ell$.
- Posterior mean and variance come from one Cholesky factorisation; the variance doesn't depend on $y$.
- Narrow bands can mean an over-smooth, confidently wrong model; use the marginal likelihood to compare hyperparameters.
- Hyperparameters depend on the scale of $x$ and $y$; `gausspr` standardises both by default.
- A pointwise 95% band is $\pm 1.96$ sd. Periodic × SE kernels model seasonality that drifts.

## Resources

- Carl Edward Rasmussen, Christopher K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006, Section 2.2 and Algorithm 2.1, Chapter 4 (covariance functions), Chapter 5 (model selection): https://gaussianprocess.org/gpml/
- David Duvenaud, "The Kernel Cookbook: Advice on covariance functions": https://www.cs.toronto.edu/~duvenaud/cookbook/
- kernlab reference manual, `gausspr`: https://cran.r-project.org/package=kernlab
- 732A96 Advanced Machine Learning, Linköping University (course lab, autumn 2020).
