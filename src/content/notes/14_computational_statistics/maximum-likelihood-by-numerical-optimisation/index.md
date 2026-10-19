---
title: "Maximum likelihood with optim(): log-likelihoods, gradients and gradient checks"
slug: maximum-likelihood-by-numerical-optimisation
date: 2020-02-02
topic: Computational Statistics
tags: [university-lab, maximum-likelihood, optimization, gradient-check, bfgs, conjugate-gradient]
summary: "Why we minimise the negative log-likelihood instead of maximising the likelihood, how BFGS and conjugate gradient compare, and why an analytic gradient should always be checked against finite differences."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "Computational Statistics (2nd ed.), Chapter 2: Optimization and solving nonlinear equations"
    author: "Geof H. Givens, Jennifer A. Hoeting"
    type: book
  - title: "Numerical Optimization (2nd ed.), Chapters 5, 6 and 8"
    author: "Jorge Nocedal, Stephen J. Wright"
    type: book
  - title: "R documentation: optim()"
    author: "R Core Team"
    url: "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html"
    type: docs


status: draft
---

For a normal distribution the maximum likelihood estimates have a closed form, so it's a perfect test case for numerical optimisation: you know the right answer, so you can see when the optimiser gets it wrong. In our computational statistics lab it did get it wrong at first, and the reason (a buggy gradient that nobody checked) is the most useful lesson from that lab.

## The likelihood and why we take its log

For independent observations $y_1, \dots, y_n$ from $N(\mu, \sigma^2)$, the likelihood is

$$
L(\mu, \sigma) = \prod_{i=1}^n \frac{1}{\sigma\sqrt{2\pi}} \exp\!\Big(-\frac{(y_i - \mu)^2}{2\sigma^2}\Big),
$$

where $\mu$ is the mean, $\sigma$ the standard deviation, and each factor is the density of one observation. Each factor is often well below 1, so with a few hundred or thousand observations the product drops below the smallest double ($\approx 10^{-308}$) and becomes exactly 0 (underflow). Even before that, a likelihood like $10^{-150}$ is awkward for an optimiser to work with. Taking logs turns the product into a sum, which is numerically harmless:

$$
\ell(\mu, \sigma) = -\frac{n}{2}\log(2\pi) - n\log\sigma - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mu)^2.
$$

The log is increasing, so the maximiser doesn't change. Optimisers in R minimise by default, so we hand them $-\ell$.

Setting the partial derivatives to zero gives

$$
\hat\mu = \frac{1}{n}\sum_{i=1}^n y_i, \qquad \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^n (y_i - \hat\mu)^2.
$$

Note the $1/n$, not $1/(n-1)$: the MLE of the variance is slightly biased. For the lab data this gave $\hat\mu \approx 1.2755$ and $\hat\sigma \approx 2.0060$.

## The gradient, and the bug

The gradient of $-\ell$ is

$$
\frac{\partial(-\ell)}{\partial\mu} = -\frac{1}{\sigma^2}\sum_{i=1}^n (y_i - \mu), \qquad \frac{\partial(-\ell)}{\partial\sigma} = \frac{n}{\sigma} - \frac{1}{\sigma^3}\sum_{i=1}^n (y_i - \mu)^2.
$$

In the first version of our lab report the gradient function had two mistakes:

1. The $\mu$-part was written as `(1/2*sigma^2) * sum(...)`. In R, `1/2*sigma^2` means $\sigma^2/2$, not $1/(2\sigma^2)$. Operator precedence bit us.
2. The $\sigma$-part was the derivative of $+\ell$, not of $-\ell$. So it pointed uphill for the function we were minimising.

With that gradient, both CG and BFGS stopped at $\sigma \approx -2.73$ (a negative standard deviation!), with a worse objective value than the closed-form answer, and `optim()` still reported `convergence = 0`. Our text said "the algorithms converged to the true value in all cases", which wasn't true for that run. The later version of the report had the gradient fixed (the code still read the data from a global variable instead of its `y` argument, which only worked by luck).

Two lessons:

- **Convergence code 0 only means the algorithm's stopping rule fired.** It says nothing about whether you found the right answer.
- **Always check an analytic gradient against finite differences** at a couple of random points before using it. It takes five lines and catches sign errors, missing factors and precedence bugs.

## Keep $\sigma$ positive by reparametrising

The negative $\sigma$ also shows another weakness: nothing stopped the optimiser from leaving the valid region. The log-likelihood only depends on $\sigma^2$, so a negative $\sigma$ looks just as valid to it. A common trick is to optimise over $s = \log\sigma$, which can be any real number, and use $\sigma = e^s$. By the chain rule, $\partial(-\ell)/\partial s = \sigma \cdot \partial(-\ell)/\partial\sigma = n - \sigma^{-2}\sum(y_i - \mu)^2$.

```r
set.seed(2)
y <- rnorm(100, mean = 1.3, sd = 2)
nll <- function(p, y) {                       # p = (mu, log sigma)
  mu <- p[1]; sigma <- exp(p[2])
  length(y) * log(sigma) + sum((y - mu)^2) / (2 * sigma^2)   # constant dropped
}
nll_grad <- function(p, y) {
  mu <- p[1]; sigma <- exp(p[2])
  c(-sum(y - mu) / sigma^2, length(y) - sum((y - mu)^2) / sigma^2)
}
# Gradient check: analytic vs central differences at an arbitrary point
p0 <- c(0.5, 0.2); h <- 1e-6
num <- sapply(1:2, function(k) {
  e <- replace(numeric(2), k, h); (nll(p0 + e, y) - nll(p0 - e, y)) / (2 * h)
})
rbind(analytic = nll_grad(p0, y), numeric = num)       # should agree to ~6 digits

fit_bfgs <- optim(c(0, 0), nll, nll_grad, y = y, method = "BFGS")
fit_cg   <- optim(c(0, 0), nll, nll_grad, y = y, method = "CG")
rbind(BFGS = fit_bfgs$counts, CG = fit_cg$counts)      # BFGS: 30/10, CG: 168/49
c(exp(fit_bfgs$par[2]), sqrt(mean((y - mean(y))^2)))   # same sigma both ways
```

## BFGS or conjugate gradient?

Both use only gradients, never the true Hessian (the matrix of second derivatives).

- **BFGS** builds up an approximation of the inverse Hessian from successive gradient changes. That costs $O(p^2)$ memory for $p$ parameters, but it captures curvature, so it usually needs far fewer iterations.
- **Conjugate gradient (CG)** keeps only a couple of vectors, so its memory is $O(p)$. It's the choice when $p$ is huge. On small problems it tends to need many more evaluations.

In the lab, with the gradient left to finite differences, CG used 208 function and 35 gradient evaluations, BFGS 41 and 15. In the snippet above the gap is similar. For two parameters, BFGS is the obvious pick. Giving an analytic gradient (a correct one) saves the extra function calls finite differences need and is more accurate.

For the one-parameter case, and why gradient methods stall on flat objectives, see [grid search, golden section and BFGS](/notes/golden-section-search-and-flat-objectives/).

## Key takeaways

- Minimise the negative log-likelihood: sums don't underflow the way products do.
- Check every hand-written gradient numerically. "Converged" is not the same as "correct".
- Watch R's operator precedence: `1/2*x` is $x/2$.
- Reparametrise constrained parameters (for example $\sigma = e^s$) so the optimiser can't wander out of the valid region.
- BFGS is usually faster for small problems; CG saves memory for very large ones.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 2 (2020).
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 2.
- Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006: Chapter 5 (conjugate gradient), Chapter 6 (quasi-Newton / BFGS), Chapter 8 (finite-difference gradients).
- R documentation, `optim()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html
