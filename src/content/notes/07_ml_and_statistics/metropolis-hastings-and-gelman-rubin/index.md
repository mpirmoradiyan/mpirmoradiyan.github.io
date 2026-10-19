---
title: "Metropolis-Hastings: proposals, the Hastings correction and checking convergence"
slug: metropolis-hastings-and-gelman-rubin
date: 2020-02-17
topic: ML & Statistics
tags: [university-lab, mcmc, metropolis-hastings, gelman-rubin, monte-carlo-integration, bayesian-computation]
summary: "How the Metropolis-Hastings algorithm samples a density known only up to a constant, why the choice of proposal matters so much, and how the Gelman-Rubin statistic compares several chains to judge convergence."
sources:
  - title: "732A90 Computational Statistics (course lab)"
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
  - title: "Computational Statistics (2nd ed.), Chapter 7: Markov chain Monte Carlo"
    author: "Geof H. Givens, Jennifer A. Hoeting"
    type: book


status: draft
---

Rejection sampling needs an envelope that covers the whole target, which gets hopeless in more than a few dimensions. Markov chain Monte Carlo (MCMC) takes a different route: build a random walk whose long-run distribution is the target, and use the path as (correlated) samples. Metropolis-Hastings is the general recipe. In the fourth computational statistics lab we sampled $f(x) \propto x^5 e^{-x}$ with two different proposals, and the difference between them was dramatic.

## The algorithm

We want samples from a density $f$ that we can evaluate up to a constant. We choose a **proposal** $q(x' \mid x)$: a distribution for the next candidate $x'$ given the current state $x$. Then, starting from some $x_0$:

1. Draw a candidate $x' \sim q(\cdot \mid x_t)$.
2. Compute the acceptance probability
$$
\alpha = \min\left(1,\; \frac{f(x')\, q(x_t \mid x')}{f(x_t)\, q(x' \mid x_t)}\right).
$$
3. With probability $\alpha$ set $x_{t+1} = x'$; otherwise $x_{t+1} = x_t$ (repeat the current value).

The unknown normalising constant of $f$ cancels in the ratio, which is the whole point. The factor $q(x_t \mid x')/q(x' \mid x_t)$ is the **Hastings correction**. If the proposal is more likely to suggest moves in one direction than the other, this factor balances it out so the chain doesn't drift toward wherever the proposal likes to go. For a symmetric proposal (like a normal random walk) it equals 1 and you get the original Metropolis algorithm.

Why it works, in one sentence: these rules make the chain satisfy *detailed balance*, $f(x)\,P(x \to x') = f(x')\,P(x' \to x)$, which makes $f$ a stationary distribution of the chain.

A practical tip: compute $\log \alpha$ and compare with $\log U$. Densities like $x^5 e^{-x}$ can overflow or underflow for extreme $x$, while their logs won't.

## The proposal makes or breaks it

The lab compared two proposals for the same target:

- A **log-normal** $LN(x_t, 1)$, meaning $\log x' \sim N(x_t, 1)$. The current value goes in as the mean *on the log scale*, so from $x_t = 6$ the proposal is centred around $e^6 \approx 400$, where the target is essentially zero. Most proposals are rejected, the chain sticks in place for long stretches, and when it moves it jumps wildly. Its trace plot had long flat segments, and the sample mean came out at about 3.75.
- A **chi-square** $\chi^2(\lfloor x_t + 1 \rfloor)$, whose mean is about $x_t + 1$. Proposals land near the current value on the natural scale. The trace plot looked like a fuzzy band around a stable level from the start, and the sample mean came out at about 6.0001.

The target is a Gamma distribution with shape 6 and scale 1 (density $x^{\alpha-1}e^{-x/\beta}/(\Gamma(\alpha)\beta^\alpha)$ with $\alpha = 6$, $\beta = 1$), whose mean is $\alpha\beta = 6$. So the chi-square chain got it right and the log-normal chain was far off.

In our lab report we called $\beta$ the "rate factor". In the parametrisation we wrote down, $x^{\alpha-1}e^{-x/\beta}$, $\beta$ is the **scale**. With a rate $\theta$ the density has $e^{-\theta x}$ and the mean is $\alpha/\theta$. Here $\beta = \theta = 1$, so the number didn't change, but mixing up the two is a classic way to get a wrong mean.

## Monte Carlo integration with the chain

Once the chain has settled, averages along it estimate expectations:

$$
\int_0^\infty x\, f(x)\, dx = E_f[X] \approx \frac{1}{N}\sum_{t=1}^{N} x_t,
$$

where $f$ here is the normalised target density, $x_t$ are the chain values after burn-in and $N$ is how many we keep. In the lab report we described this integral as "the expected value of $f(x)$". That's a slip worth fixing: it's the expected value of $X$ under the density $f$. The expected value of $f(X)$ would be $\int f(x)^2\,dx$, something different.

## Checking convergence: Gelman-Rubin

A single trace plot can look fine while the chain is stuck in one region. Gelman and Rubin's idea is to run $k$ chains from spread-out starting points and compare. If they've all converged to the same distribution, the variation *between* chains should match the variation *within* them. For $k$ chains of length $n$, with $\bar x_j$ the mean of chain $j$, $s_j^2$ its variance and $\bar x$ the overall mean:

$$
W = \frac{1}{k}\sum_{j=1}^k s_j^2, \qquad B = \frac{n}{k-1}\sum_{j=1}^k (\bar x_j - \bar x)^2,
$$

$$
\hat V = \frac{n-1}{n} W + \frac{1}{n} B, \qquad \hat R = \sqrt{\hat V / W}.
$$

$W$ is the average within-chain variance, $B$ the between-chain variance (scaled by $n$), and $\hat V$ an estimate of the target variance that's too big if the chains haven't mixed. $\hat R$ (the potential scale reduction factor) near 1 means the chains agree; values clearly above 1 (a common rule of thumb is 1.1, newer guidance says 1.01) mean run longer. In the lab, ten chi-square chains started at 1 to 10 gave $\hat R = 1$ (point estimate and upper confidence limit, from `coda::gelman.diag`).

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

For tuning a symmetric random-walk proposal, see [random-walk Metropolis](/notes/random-walk-metropolis/). For the case where every conditional can be sampled directly, see [Gibbs sampling for a random-walk smoother](/notes/gibbs-sampling-full-conditionals/).

## Key takeaways

- MH needs $f$ only up to a constant; the Hastings ratio $q(x_t\mid x')/q(x'\mid x_t)$ corrects for asymmetric proposals.
- A proposal that puts most candidates where $f$ is tiny gives a sticky chain and biased averages, even though the algorithm is "correct" in the limit.
- $\int x f(x)\,dx$ is $E[X]$, estimated by the chain's average after burn-in.
- In $x^{\alpha-1}e^{-x/\beta}$, $\beta$ is a scale, not a rate.
- Gelman-Rubin compares between- and within-chain variance; $\hat R$ near 1 with dispersed starts is good evidence of convergence.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 4 (2020).
- W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications", *Biometrika* 57(1), 1970: https://doi.org/10.1093/biomet/57.1.97
- Andrew Gelman and Donald B. Rubin, "Inference from iterative simulation using multiple sequences", *Statistical Science* 7(4), 1992: https://doi.org/10.1214/ss/1177011136
- Aki Vehtari et al., "Rank-normalization, folding, and localization: an improved $\hat R$ for assessing convergence of MCMC", *Bayesian Analysis* 16(2), 2021: https://doi.org/10.1214/20-BA1221
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 7.
