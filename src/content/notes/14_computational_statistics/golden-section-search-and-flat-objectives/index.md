---
title: "Tuning one parameter: grid search, golden section, and why BFGS can get stuck"
slug: golden-section-search-and-flat-objectives
date: 2020-02-02
topic: Computational Statistics
tags: [university-lab, optimization, golden-section-search, hyperparameter-tuning, bfgs, r]
summary: "How golden-section search finds the minimum of a one-parameter function in a handful of evaluations, and why a gradient method like BFGS stops dead on a piecewise-constant objective."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "Computational Statistics (2nd ed.), Chapter 2: Optimization and solving nonlinear equations"
    author: "Geof H. Givens, Jennifer A. Hoeting"
    type: book
  - title: "Algorithms for Minimization without Derivatives"
    author: "Richard P. Brent"
    type: book
  - title: "R documentation: optimize() and optim()"
    author: "R Core Team"
    url: "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optimize.html"
    type: docs


status: draft
---

A lot of model tuning is "find the one number that makes validation error smallest". The lazy way is to try every value on a grid. In the second computational statistics lab we tuned the smoothness of a `loess` fit by test MSE three ways: a grid, R's `optimize()`, and `optim()` with BFGS. The grid needed 400 model fits, `optimize()` needed 13, and BFGS didn't move at all. Each of those results teaches something.

## Grid search: simple, safe, expensive

Evaluate the objective at $\lambda_1, \lambda_2, \dots, \lambda_m$ and keep the best. With a step of 0.1 on $[0.1, 40]$ that's $m = 400$ fits. It can't be fooled by a strange-shaped curve (as long as the grid is fine enough), but the cost grows linearly with the resolution you want, and exponentially with the number of parameters.

In the lab the grid found the minimum test MSE of about 0.1386 at $\lambda = 11.7$.

## Golden-section search: shrink a bracket

If the function is **unimodal** on $[a, b]$ (it goes down, then up, with one minimum), you can throw away part of the interval after every evaluation.

Pick two interior points $x_1 < x_2$. If $f(x_1) < f(x_2)$, the minimum can't be to the right of $x_2$, so the new interval is $[a, x_2]$. Otherwise it's $[x_1, b]$. The trick is where to put $x_1$ and $x_2$:

$$
x_1 = b - r(b - a), \qquad x_2 = a + r(b - a), \qquad r = \frac{\sqrt5 - 1}{2} \approx 0.618,
$$

where $a$ and $b$ are the current bracket ends and $r$ is the golden ratio minus one. With this choice, one of the old interior points lands exactly where a new interior point is needed. So each step costs **one** new evaluation and shrinks the interval by a factor of 0.618. To get from length $L$ down to a tolerance $\epsilon$ you need about $\log(L/\epsilon) / \log(1/0.618)$ steps. For $L = 40$ and $\epsilon = 0.01$, that's about 17.

```r
golden <- function(f, a, b, tol = 1e-3) {
  r <- (sqrt(5) - 1) / 2
  x1 <- b - r * (b - a); x2 <- a + r * (b - a)
  f1 <- f(x1); f2 <- f(x2)
  while (b - a > tol) {
    if (f1 < f2) { b <- x2; x2 <- x1; f2 <- f1; x1 <- b - r * (b - a); f1 <- f(x1) }
    else         { a <- x1; x1 <- x2; f1 <- f2; x2 <- a + r * (b - a); f2 <- f(x2) }
  }
  (a + b) / 2
}
golden(function(x) (x - 2)^2, 0, 5)   # 1.999766
```

R's `optimize()` is Brent's method: golden section combined with parabolic interpolation (fit a parabola through three points and jump to its minimum when that looks safe). That's why it used only 13 evaluations in the lab and landed at $\lambda \approx 12.68$ with MSE $0.1390$, almost the same as the grid's best.

The catch: on a function with several dips, golden section happily converges to *a* local minimum, whichever one the bracket closes on. A coarse grid first, then golden section inside the best cell, is a good combination.

## Why BFGS stopped at its starting point

BFGS is a quasi-Newton method: it follows the gradient and builds up an approximation of the curvature. If you don't give `optim()` a gradient, it estimates one with finite differences, $\big(f(\lambda + h) - f(\lambda - h)\big)/(2h)$ with a small $h$ (by default $10^{-3}$).

Started at $\lambda = 35$, BFGS reported 1 function evaluation, 1 gradient evaluation, "converged", and returned 35 with MSE 0.2426, much worse than the true minimum. Our counter showed 3 calls in total: one for the function value and two for the finite-difference gradient (`optim()`'s counts don't include those).

The reason is the shape of the objective. In `loess`, the smoothing parameter is turned into a span, and the span decides how many neighbouring points go into each local fit, which is a whole number. So the test MSE as a function of $\lambda$ is a **staircase**: flat over small intervals, with jumps in between. On a flat step, $f(\lambda + h) = f(\lambda - h)$, the estimated gradient is exactly 0, and BFGS thinks it's already at a minimum.

In our lab report we said this happened "because our variable (lambda) is discrete". That's not quite it. $\lambda$ can be any real number. It's the *objective* that's piecewise constant in $\lambda$, because the quantity that actually changes the fit (the number of neighbours) is discrete. The fix is the same either way: use a derivative-free method (golden section, Nelder-Mead, a grid), or tune the discrete quantity directly.

```r
calls <- 0
f <- function(x) { calls <<- calls + 1; (floor(x) - 12)^2 / 100 + 0.14 }  # a staircase

calls <- 0; o <- optimize(f, lower = 0.1, upper = 40, tol = 0.01)
c(best = o$minimum, calls = calls)        # 12.005 after 18 calls

calls <- 0; b <- optim(par = 35.5, fn = f, method = "BFGS")
c(best = b$par, calls = calls)            # 35.5 after 3 calls: stuck on a flat step
```

For optimisers with more than one parameter, and how to feed them gradients safely, see [maximum likelihood with optim()](/notes/maximum-likelihood-by-numerical-optimisation/).

## Key takeaways

- Grid search is hard to fool but its cost grows with resolution and dimension.
- Golden-section search shrinks a bracket by 0.618 per evaluation, but only finds the right answer if the function is unimodal in the bracket. `optimize()` adds parabolic steps for speed.
- Gradient methods with finite-difference gradients see a gradient of 0 on any flat region, so they stop immediately on step-shaped objectives.
- Before choosing an optimiser, ask what the objective looks like: smooth, noisy, or piecewise constant.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 2 (2020).
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 2.
- Richard P. Brent, *Algorithms for Minimization without Derivatives*, Prentice-Hall, 1973 (the method behind `optimize()`).
- Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006, Chapter 6 (quasi-Newton methods) and Chapter 8 (finite-difference derivatives).
- R documentation, `optimize()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optimize.html
- R documentation, `optim()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html
