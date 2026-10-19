---
title: "Gibbs sampling: deriving full conditionals for a random-walk smoother"
slug: gibbs-sampling-full-conditionals
date: 2020-02-17
topic: Computational Statistics
tags: [university-lab, mcmc, gibbs-sampling, bayesian-inference, full-conditionals, smoothing]
summary: "How to get each full conditional distribution from a joint posterior by completing the square, how a Gibbs sampler sweeps through them, and the indexing bug that quietly turns it into something else."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "Explaining the Gibbs sampler"
    author: "George Casella, Edward I. George"
    url: "https://doi.org/10.1080/00031305.1992.10475878"
    type: paper
  - title: "Computational Statistics (2nd ed.), Chapter 7: Markov chain Monte Carlo"
    author: "Geof H. Givens, Jennifer A. Hoeting"
    type: book
  - title: "Bayesian Data Analysis (3rd ed.), Chapter 11"
    author: "Andrew Gelman et al."
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book


status: draft
---

In the fourth computational statistics lab we had 50 days of noisy measurements and wanted a smooth curve through them, with uncertainty. The joint posterior over all 50 levels looked messy. But each level on its own, *given all the others*, turned out to be a plain normal you can sample directly. That's the setting for Gibbs sampling: cycle through the parameters, drawing each one from its conditional, and the sequence of full vectors is a Markov chain with the joint posterior as its target. Our derivation was right. Our code had a bug that I want to remember.

## The model

We observe $Y_1, \dots, Y_n$ (one per day) and assume each is a noisy version of an unknown smooth level $\mu_i$:

$$
Y_i \mid \mu_i \sim N(\mu_i, \sigma^2), \qquad i = 1, \dots, n.
$$

The prior says the level changes slowly, like a random walk:

$$
p(\mu_1) \propto 1, \qquad \mu_{i+1} \mid \mu_i \sim N(\mu_i, \sigma^2), \qquad i = 1, \dots, n-1.
$$

Here $\sigma^2$ is a known variance (0.2 in the lab, for both noise and steps) and $p(\mu_1) \propto 1$ is a flat prior on the starting level.

## Posterior, up to a constant

Bayes' theorem: posterior $\propto$ likelihood $\times$ prior. All factors are Gaussian, so

$$
p(\vec\mu \mid \vec Y) \propto \exp\!\left(-\frac{1}{2\sigma^2}\left[\sum_{i=1}^n (Y_i - \mu_i)^2 + \sum_{i=1}^{n-1} (\mu_{i+1} - \mu_i)^2\right]\right).
$$

The first sum pulls each $\mu_i$ toward its observation; the second pulls neighbours toward each other. That tug of war is the smoothing.

## Full conditionals by "keep only the terms with $\mu_i$"

For the conditional of $\mu_i$ given everything else, treat all other $\mu$'s as constants and drop every factor that doesn't involve $\mu_i$. For an interior day ($1 < i < n$) three squared terms survive:

$$
p(\mu_i \mid \vec\mu_{-i}, \vec Y) \propto \exp\!\left(-\frac{1}{2\sigma^2}\Big[(\mu_i - Y_i)^2 + (\mu_i - \mu_{i-1})^2 + (\mu_i - \mu_{i+1})^2\Big]\right).
$$

A sum of $k$ squares $(\mu - a_j)^2$ equals $k(\mu - \bar a)^2$ plus terms without $\mu$, where $\bar a$ is the average of the $a_j$. (That's completing the square.) So with $k = 3$:

$$
\mu_i \mid \vec\mu_{-i}, \vec Y \;\sim\; N\!\left(\frac{Y_i + \mu_{i-1} + \mu_{i+1}}{3},\; \frac{\sigma^2}{3}\right).
$$

At the two ends only two terms survive ($k = 2$):

$$
\mu_1 \mid \cdot \sim N\!\left(\frac{Y_1 + \mu_2}{2}, \frac{\sigma^2}{2}\right), \qquad \mu_n \mid \cdot \sim N\!\left(\frac{Y_n + \mu_{n-1}}{2}, \frac{\sigma^2}{2}\right).
$$

Nice and intuitive: each level is drawn around the average of its own observation and its neighbours, with less spread the more terms pin it down.

## The sweep, and the bug

One Gibbs iteration updates $\mu_1, \mu_2, \dots, \mu_n$ in turn, **always using the most recent value of every other component**. So when updating $\mu_i$ in iteration $t$, $\mu_{i-1}$ is the value just drawn in iteration $t$, and $\mu_{i+1}$ is the value from iteration $t-1$.

Our lab code stored the draws in a matrix with one row per iteration, initialised to zeros, and wrote the update for row $t$ using `mu[t, i-1]` and `mu[t, i+1]`. The first is fine. The second is still 0, because row $t$ hasn't reached column $i+1$ yet, and nothing ever copied the previous row forward. So every iteration restarted from zeros, and the "chain" was really 1000 independent single sweeps from an all-zero start, with the right neighbour always 0. On top of that, the interior mean was divided by 2 instead of 3, which happened to partly compensate (with the right-hand neighbour stuck at 0, dividing by 3 would have pulled every level toward zero).

The fitted curve still looked smooth and plausible, which is the scary part. A trace plot of a single component and a comparison with an independent method (for example the exact posterior mean, which for this Gaussian model is a linear smoother you can compute directly) would have exposed it. The fix is to keep one current state vector, update it in place, and copy it into the output matrix after each full sweep:

```r
gibbs_rw <- function(y, s2, n_iter = 1000) {
  n <- length(y)
  draws <- matrix(0, n_iter, n)
  cur <- y                                         # current state of the chain
  for (t in 1:n_iter) {
    for (i in 1:n) {
      if (i == 1)      { m <- (y[1] + cur[2]) / 2;                  v <- s2 / 2 }
      else if (i == n) { m <- (y[n] + cur[n - 1]) / 2;              v <- s2 / 2 }
      else             { m <- (y[i] + cur[i - 1] + cur[i + 1]) / 3; v <- s2 / 3 }
      cur[i] <- rnorm(1, m, sqrt(v))               # in place: newest values are used
    }
    draws[t, ] <- cur                              # store the state after the sweep
  }
  draws
}
set.seed(5)
x <- 1:50
y <- 0.3 + 1.2 * (1 - exp(-x / 15)) + rnorm(50, sd = 0.3)   # a noisy curve
draws <- gibbs_rw(y, s2 = 0.2)
post_mean <- colMeans(draws[-(1:100), ])          # drop a short burn-in

D <- diff(diag(50))                               # first-difference matrix
exact <- solve(diag(50) + crossprod(D), y)        # exact posterior mean
max(abs(post_mean - exact))                       # about 0.03: just Monte Carlo noise
```

The exact mean comes from writing the posterior as one multivariate normal: its precision matrix is $(I + D^\top D)/\sigma^2$, where $D$ takes differences of neighbours, so its mean is $(I + D^\top D)^{-1}\vec Y$.

## Things to watch

- **Correlated components mix slowly.** Neighbouring $\mu$'s are strongly tied, so one-at-a-time updates take small steps. Block updates (drawing all $\vec\mu$ jointly from its multivariate normal) mix much faster for models like this.
- **Burn-in.** Drop early iterations if the start was far from the posterior; look at trace plots and autocorrelation of a few components.
- **Gibbs is a special case of Metropolis-Hastings** where every proposal is accepted, because proposing from the exact conditional makes the acceptance ratio equal to 1. See [Metropolis-Hastings](/notes/metropolis-hastings-and-gelman-rubin/).

For Gibbs sampling with hidden labels (data augmentation) in a mixture model, see [Gibbs sampling and normal mixtures](/notes/gibbs-sampling-and-normal-mixtures/).

## Key takeaways

- Full conditionals come from the joint posterior: keep only factors that involve the parameter, then recognise the distribution (here, by completing the square).
- For this random-walk smoother, each $\mu_i$ is normal around the average of $Y_i$ and its neighbours, with variance $\sigma^2/k$ for $k$ terms.
- Each update must use the latest values of all other components. Storing iterations in rows without carrying the state forward silently breaks the chain.
- A plausible-looking result isn't proof the sampler is right: check against a known answer when you can.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 4 (2020).
- George Casella and Edward I. George, "Explaining the Gibbs sampler", *The American Statistician* 46(3), 1992: https://doi.org/10.1080/00031305.1992.10475878
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 7.
- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., CRC Press, 2013, Chapter 11: http://www.stat.columbia.edu/~gelman/book/
