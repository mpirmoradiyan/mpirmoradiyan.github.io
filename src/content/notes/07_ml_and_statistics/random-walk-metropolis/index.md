---
title: "Random-walk Metropolis: a general MCMC sampler and how to tune it"
slug: random-walk-metropolis
date: 2020-05-10
topic: ML & Statistics
tags: [university-lab, bayesian-inference, mcmc, metropolis-hastings, poisson-regression]
summary: "How the random-walk Metropolis algorithm samples any posterior you can evaluate up to a constant, why a Laplace-approximation covariance makes a good proposal, and how the step size controls the acceptance rate and mixing."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Data Analysis (3rd ed.), Sections 11.2 and 12.2 (Metropolis algorithm, efficient proposals)"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book
  - title: "Weak convergence and optimal scaling of random walk Metropolis algorithms"
    author: "Gareth O. Roberts, Andrew Gelman, Walter R. Gilks"
    url: "https://doi.org/10.1214/aoap/1034625254"
    type: paper


status: draft
---

Gibbs sampling needs conditionals you can draw from. Often you don't have them. Random-walk Metropolis needs much less: just a function that returns the log posterior at any point, up to an additive constant. In the third Bayesian Learning lab (pair work) we wrote one general sampler and used it for a Poisson regression on auction data. The part that took the most fiddling was the step size.

## The algorithm

Start at some $\theta^{(0)}$. At each step:

1. Propose a move: $\theta_p \sim N(\theta^{(t-1)}, c\,\Sigma)$.
2. Compute the acceptance ratio $r = \dfrac{p(\theta_p \mid y)}{p(\theta^{(t-1)} \mid y)}$.
3. With probability $\min(1, r)$ set $\theta^{(t)} = \theta_p$; otherwise stay: $\theta^{(t)} = \theta^{(t-1)}$.

Here $\Sigma$ sets the shape of the proposal and $c > 0$ scales its size. Because the normalising constant $p(y)$ appears in both numerator and denominator of $r$, it cancels. That's why you only need the posterior up to a constant. And because the normal proposal is symmetric, no proposal-density correction is needed (that would be Metropolis-*Hastings*).

Uphill moves are always accepted; downhill moves are accepted sometimes, in proportion to how far down they go. That's what makes the chain spend time in each region in proportion to its posterior probability, instead of just climbing to the mode.

In practice, compute everything on the log scale: accept if $\log u < \log p(\theta_p \mid y) - \log p(\theta^{(t-1)} \mid y)$ with $u \sim U(0,1)$. Taking `exp` of a log-posterior directly overflows or underflows quickly.

## A good proposal shape for free

If the parameters are correlated or on very different scales, a round proposal $N(\theta, c I)$ is terrible: it's too big in some directions and too small in others. A neat trick is to first run the [Laplace approximation](/notes/laplace-approximation-logistic-regression/) (find the posterior mode with `optim` and take the inverse negative Hessian) and use that covariance as $\Sigma$. The proposal then has roughly the shape of the posterior, and only the scalar $c$ needs tuning.

## Writing it once, for any model

The sampler shouldn't know anything about Poisson regression. In R the clean way is to pass the log-posterior as a function and forward extra arguments with `...`:

```r
rw_metropolis <- function(log_post, theta0, Sigma, c = 1, n_iter = 5000, ...) {
  L <- t(chol(c * Sigma)); d <- length(theta0)
  draws <- matrix(NA, n_iter, d)
  theta <- theta0; lp <- log_post(theta, ...)
  accepted <- 0
  for (i in 1:n_iter) {
    prop    <- theta + drop(L %*% rnorm(d))
    lp_prop <- log_post(prop, ...)
    if (log(runif(1)) < lp_prop - lp) {         # accept with prob min(1, r)
      theta <- prop; lp <- lp_prop; accepted <- accepted + 1
    }
    draws[i, ] <- theta
  }
  list(draws = draws, accept_rate = accepted / n_iter)
}

set.seed(6)   # example: Poisson regression with a N(0, 10^2 I) prior
X <- cbind(1, rnorm(300)); y <- rpois(300, exp(X %*% c(0.5, 0.8)))
log_post_pois <- function(b, X, y)
  sum(y * (X %*% b) - exp(X %*% b)) + sum(dnorm(b, 0, 10, log = TRUE))
fit <- optim(c(0, 0), log_post_pois, X = X, y = y, method = "BFGS",
             control = list(fnscale = -1), hessian = TRUE)
for (c in c(0.1, 1, 10)) {
  out <- rw_metropolis(log_post_pois, fit$par, solve(-fit$hessian), c = c, X = X, y = y)
  cat("c =", c, " acceptance =", round(out$accept_rate, 2), "\n")   # 0.84, 0.54, 0.15
}
```

Our lab version worked, but it wasn't really general: the proposal covariance was read from a global variable instead of being an argument, and the `...` was accepted but never passed on to the log-posterior. Passing both through explicitly is what makes the sampler reusable. Caching `lp` for the current state also halves the number of posterior evaluations.

The Poisson log-likelihood in that function is $\sum_i [y_i x_i^\top\beta - \exp(x_i^\top\beta)]$, dropping the $-\log y_i!$ term because it doesn't depend on $\beta$.

## Tuning $c$

The acceptance rate tells you whether $c$ is sensible:

- **$c$ too small**: almost every proposal is accepted (84% above), but each step is tiny, so the chain crawls and successive draws are strongly correlated.
- **$c$ too large**: proposals often land in the tails and get rejected (15% above). The chain sits still for many iterations, and the trace plot looks like a staircase. This is what our trace plots looked like at first: the parameters got stuck.
- **In between**: theory for random-walk Metropolis on roughly normal targets says the most efficient acceptance rate is about 0.44 in one dimension, dropping towards 0.23 in many dimensions (Roberts, Gelman and Gilks, 1997). With a Laplace-shaped proposal, $c \approx 2.38^2/d$ (with $d$ the number of parameters) is a common starting point. In practice, aiming for roughly 0.2-0.4 and checking the trace plots is good enough.

Acceptance rate is only a proxy. What you actually care about is the *effective sample size*: how many independent draws your correlated chain is worth. A chain of 5,000 sticky draws might be worth 100.

## Checking it

Run the chain, drop a burn-in, and look at trace plots for every parameter. Then compare the MCMC histograms with the Laplace approximation's normals. For a Poisson regression with plenty of data they should match closely; where they don't (skewed or bounded parameters), trust the MCMC.

## Key takeaways

- Random-walk Metropolis only needs the log posterior up to a constant.
- Accept/reject on the log scale.
- Use the Laplace covariance as the proposal shape and tune a single scale $c$.
- Aim for an acceptance rate of roughly 0.2-0.4; too high means tiny steps, too low means a stuck chain.
- Make the sampler general: pass the log-posterior, the proposal covariance and `...` explicitly.

## Resources

- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Section 11.2 (Metropolis and Metropolis-Hastings) and Section 12.2 (efficient Metropolis jumping rules): http://www.stat.columbia.edu/~gelman/book/
- Gareth O. Roberts, Andrew Gelman, Walter R. Gilks, "Weak convergence and optimal scaling of random walk Metropolis algorithms", *Annals of Applied Probability* 7(1) (1997): https://doi.org/10.1214/aoap/1034625254
- Nicholas Metropolis et al., "Equation of state calculations by fast computing machines", *Journal of Chemical Physics* 21 (1953): https://doi.org/10.1063/1.1699114
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
