---
title: "Metropolis-Hastings: proposals, the Hastings correction, tuning and checking convergence"
slug: metropolis-hastings-and-gelman-rubin
date: 2020-02-17
topic: Computational Statistics
tags: [university-lab, mcmc, metropolis-hastings, gelman-rubin, monte-carlo-integration, bayesian-computation, bayesian-inference, poisson-regression]
summary: "How Metropolis-Hastings samples a density known only up to a constant, why the proposal matters so much, how to tune a random-walk step size by its acceptance rate, and how the Gelman-Rubin statistic compares chains to judge convergence."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Monte Carlo sampling methods using Markov chains and their applications"
    author: "W. K. Hastings"
    url: "https://doi.org/10.1093/biomet/57.1.97"
    type: paper
  - title: "Inference from iterative simulation using multiple sequences"
    author: "Andrew Gelman, Donald B. Rubin"
    url: "https://doi.org/10.1214/ss/1177011136"
    type: paper
  - title: "Weak convergence and optimal scaling of random walk Metropolis algorithms"
    author: "Gareth O. Roberts, Andrew Gelman, Walter R. Gilks"
    url: "https://doi.org/10.1214/aoap/1034625254"
    type: paper
  - title: "Bayesian Data Analysis (3rd ed.), Sections 11.2 and 12.2 (Metropolis algorithm, efficient proposals)"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book
  - title: "Computational Statistics (2nd ed.), Chapter 7: Markov chain Monte Carlo"
    author: "Geof H. Givens, Jennifer A. Hoeting"
    type: book


status: draft
---

I met Metropolis twice in one semester. In the computational statistics lab we sampled $f(x) \propto x^5 e^{-x}$ with two different proposals, and one chain came out badly wrong. A few months later, in Bayesian Learning, we wrote a general random-walk sampler for a Poisson regression, and this time the fiddly part was the step size. Same algorithm, two different lessons, so here they are together.

The starting problem: rejection sampling needs an envelope that covers the whole target, and that gets hopeless in more than a few dimensions. Markov chain Monte Carlo (MCMC) takes another route. Build a random walk whose long-run distribution is the target, and use its path as (correlated) samples.

## The algorithm

We want samples from a density $f$ that we can only evaluate up to a constant. We pick a **proposal** $q(x' \mid x)$: a distribution for the next candidate $x'$ given where we are now, $x$. Then, from some start $x_0$:

1. Draw a candidate $x' \sim q(\cdot \mid x_t)$.
2. Compute the acceptance probability
$$
\alpha = \min\left(1,\; \frac{f(x')\, q(x_t \mid x')}{f(x_t)\, q(x' \mid x_t)}\right).
$$
3. With probability $\alpha$ set $x_{t+1} = x'$; otherwise $x_{t+1} = x_t$ (repeat the current value).

The unknown normalising constant of $f$ appears on top and bottom, so it cancels. That's the whole point. For a Bayesian posterior it means you never need $p(y)$.

The factor $q(x_t \mid x')/q(x' \mid x_t)$ is the **Hastings correction**. If the proposal likes suggesting moves in one direction more than the other, this factor balances it out, so the chain doesn't drift towards wherever the proposal likes to go. For a symmetric proposal (like a normal random walk) it equals 1, and you get the original Metropolis algorithm.

The way I picture it: moves uphill (to higher $f$) are always accepted, and moves downhill are accepted with probability equal to the density ratio. So the chain doesn't just climb to the mode; it spends time in each region in proportion to its probability. Formally, these rules give *detailed balance*, $f(x)\,P(x \to x') = f(x')\,P(x' \to x)$, which makes $f$ a stationary distribution of the chain.

One practical habit: work on the log scale. Accept if $\log u < \log \alpha$ with $u \sim U(0,1)$. Densities like $x^5 e^{-x}$, or any posterior with a few hundred data points, overflow or underflow quickly, while their logs don't.

## The proposal makes or breaks it

The first lab compared two proposals for the same target:

- A **log-normal** $LN(x_t, 1)$, meaning $\log x' \sim N(x_t, 1)$. The current value goes in as the mean *on the log scale*, so from $x_t = 6$ the proposals are centred around $e^6 \approx 400$, where the target is basically zero. Almost everything is rejected, the chain sits still for long stretches, and the trace plot had long flat segments. The sample mean came out at about 3.75.
- A **chi-square** $\chi^2(\lfloor x_t + 1 \rfloor)$, whose mean is about $x_t + 1$. Proposals land near the current value on the natural scale. The trace plot looked like a fuzzy band around a stable level from the start, and the sample mean came out at about 6.0.

The target is a Gamma distribution with shape 6 and scale 1 (density $x^{\alpha-1}e^{-x/\beta}/(\Gamma(\alpha)\beta^\alpha)$ with $\alpha = 6$, $\beta = 1$), so its mean is $\alpha\beta = 6$. The chi-square chain got it right and the log-normal chain was far off. Both chains are "correct" in the limit, but the bad one would need an enormous number of iterations to get there.

In our report we called $\beta$ the "rate factor". In $x^{\alpha-1}e^{-x/\beta}$, $\beta$ is the **scale**. With a rate $\theta$ the density has $e^{-\theta x}$ and the mean is $\alpha/\theta$. Here both are 1, so the number didn't change, but mixing them up is a classic way to get a wrong mean.

## Random-walk Metropolis and tuning the step size

The second lab used the most common proposal: a normal step around the current point,

$$
\theta_p \sim N(\theta^{(t-1)}, c\,\Sigma),
$$

where $\Sigma$ sets the shape of the proposal and $c > 0$ its size. It's symmetric, so no Hastings correction is needed.

**A good shape for free.** If the parameters are correlated or on very different scales, a round proposal $N(\theta, cI)$ is too big in some directions and too small in others. A neat trick is to first run the [Laplace approximation](/notes/laplace-approximation-logistic-regression/) (find the posterior mode with `optim` and take the inverse of the negative Hessian) and use that covariance as $\Sigma$. Then only the single number $c$ needs tuning.

**Writing it once, for any model.** The sampler shouldn't know anything about Poisson regression. In R you pass the log-posterior as a function and forward extra arguments with `...`:

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

The Poisson log-likelihood there is $\sum_i [y_i x_i^\top\beta - \exp(x_i^\top\beta)]$; the $-\log y_i!$ term is dropped because it doesn't depend on $\beta$. Our lab version worked but wasn't really general: the proposal covariance came from a global variable, and `...` was accepted but never passed on. Caching `lp` for the current state also halves the number of posterior evaluations.

**Reading the acceptance rate.**

- **$c$ too small**: almost everything is accepted (84% above), but each step is tiny, so the chain crawls and neighbouring draws are strongly correlated.
- **$c$ too large**: proposals often land in the tails and get rejected (15% above). The chain sits still and the trace plot looks like a staircase. That's how our trace plots looked at first.
- **In between**: for random-walk Metropolis on roughly normal targets, the most efficient acceptance rate is about 0.44 in one dimension, dropping towards 0.23 in many dimensions (Roberts, Gelman and Gilks, 1997). With a Laplace-shaped proposal, $c \approx 2.38^2/d$ ($d$ = number of parameters) is a common starting point. Aiming for roughly 0.2-0.4 and looking at trace plots is good enough in practice.

The acceptance rate is only a proxy. What you really care about is the *effective sample size*: how many independent draws your correlated chain is worth. 5,000 sticky draws might be worth only 100.

## Using the chain: Monte Carlo integration

Once the chain has settled, averages along it estimate expectations:

$$
\int_0^\infty x\, f(x)\, dx = E_f[X] \approx \frac{1}{N}\sum_{t=1}^{N} x_t,
$$

where $f$ is the normalised target, $x_t$ are the values after burn-in and $N$ is how many we keep. In our report we called this integral "the expected value of $f(x)$". It's the expected value of $X$ under the density $f$. The expected value of $f(X)$ would be $\int f(x)^2\,dx$, something else.

For the regression, a nice sanity check is to compare the MCMC histograms with the Laplace approximation's normals. With plenty of data they should match closely; where they don't (skewed or bounded parameters), trust the MCMC.

## Checking convergence: Gelman-Rubin

A single trace plot can look fine while the chain is stuck in one region. Gelman and Rubin's idea: run $k$ chains from spread-out starting points and compare them. If they've all reached the same distribution, the variation *between* chains should match the variation *within* them. For $k$ chains of length $n$, with $\bar x_j$ the mean of chain $j$, $s_j^2$ its variance and $\bar x$ the overall mean:

$$
W = \frac{1}{k}\sum_{j=1}^k s_j^2, \qquad B = \frac{n}{k-1}\sum_{j=1}^k (\bar x_j - \bar x)^2,
$$

$$
\hat V = \frac{n-1}{n} W + \frac{1}{n} B, \qquad \hat R = \sqrt{\hat V / W}.
$$

$W$ is the average within-chain variance, $B$ the between-chain variance (scaled by $n$), and $\hat V$ an estimate of the target variance that's too big if the chains haven't mixed. $\hat R$ (the potential scale reduction factor) near 1 means the chains agree. Values clearly above 1 mean run longer; the old rule of thumb was 1.1, newer guidance says 1.01. In the lab, ten chi-square chains started at 1 to 10 gave $\hat R = 1$ (point estimate and upper limit, from `coda::gelman.diag`).

```r
log_target <- function(x) 5 * log(x) - x                # x^5 e^(-x), unnormalised
mh_chisq <- function(n, x0) {
  x <- numeric(n); x[1] <- x0
  for (t in 2:n) {
    cur <- x[t - 1]; prop <- rchisq(1, df = floor(cur + 1))
    log_a <- log_target(prop) - log_target(cur) +
      dchisq(cur, floor(prop + 1), log = TRUE) -        # q(cur | prop)
      dchisq(prop, floor(cur + 1), log = TRUE)          # q(prop | cur)
    x[t] <- if (log(runif(1)) < log_a) prop else cur
  }
  x
}
gelman_rubin <- function(ch) {                           # ch: n x k matrix
  n <- nrow(ch); W <- mean(apply(ch, 2, var)); B <- n * var(colMeans(ch))
  sqrt(((n - 1) / n * W + B / n) / W)
}
set.seed(4)
chains <- sapply(1:10, function(s) mh_chisq(5000, x0 = s))
gelman_rubin(chains)                                     # about 1.000
mean(chains[-(1:500), ])                                 # about 6
```

$\hat R \approx 1$ is necessary, not sufficient. If all chains start in the same wrong region and never leave it, they'll agree with each other and still be wrong. Spread out the starting points.

When every full conditional can be sampled directly, you don't need a proposal at all: see [Gibbs sampling for a random-walk smoother](/notes/gibbs-sampling-full-conditionals/) and [Gibbs sampling for normal mixtures](/notes/gibbs-sampling-and-normal-mixtures/).

## Key takeaways

- MH needs $f$ only up to a constant; the Hastings ratio $q(x_t\mid x')/q(x'\mid x_t)$ corrects for asymmetric proposals and is 1 for a random walk.
- A proposal that puts most candidates where $f$ is tiny gives a sticky chain and badly biased averages in any realistic run.
- For random-walk Metropolis, use the Laplace covariance as the shape and tune one scale $c$; aim for an acceptance rate of roughly 0.2-0.4.
- $\int x f(x)\,dx$ is $E[X]$, estimated by the chain's average after burn-in. In $x^{\alpha-1}e^{-x/\beta}$, $\beta$ is a scale, not a rate.
- Gelman-Rubin compares between- and within-chain variance; $\hat R$ near 1 with dispersed starts is good evidence of convergence.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 4 (2020).
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
- W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications", *Biometrika* 57(1), 1970: https://doi.org/10.1093/biomet/57.1.97
- Nicholas Metropolis et al., "Equation of state calculations by fast computing machines", *Journal of Chemical Physics* 21 (1953): https://doi.org/10.1063/1.1699114
- Andrew Gelman and Donald B. Rubin, "Inference from iterative simulation using multiple sequences", *Statistical Science* 7(4), 1992: https://doi.org/10.1214/ss/1177011136
- Aki Vehtari et al., "Rank-normalization, folding, and localization: an improved $\hat R$ for assessing convergence of MCMC", *Bayesian Analysis* 16(2), 2021: https://doi.org/10.1214/20-BA1221
- Gareth O. Roberts, Andrew Gelman, Walter R. Gilks, "Weak convergence and optimal scaling of random walk Metropolis algorithms", *Annals of Applied Probability* 7(1) (1997): https://doi.org/10.1214/aoap/1034625254
- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Section 11.2 (Metropolis and Metropolis-Hastings) and Section 12.2 (efficient Metropolis jumping rules): http://www.stat.columbia.edu/~gelman/book/
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 7.
