---
title: "Gibbs sampling, from a normal model to a mixture with data augmentation"
slug: gibbs-sampling-and-normal-mixtures
date: 2020-05-10
topic: Computational Statistics
tags: [university-lab, bayesian-inference, mcmc, gibbs-sampling, mixture-models, data-augmentation]
summary: "How Gibbs sampling builds a joint posterior out of easy conditional draws, and how adding a hidden label per observation turns a messy mixture model into a chain of simple updates."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Data Analysis (3rd ed.), Section 11.1 (Gibbs sampler) and Chapter 22 (finite mixture models)"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book


status: draft
---

Gibbs sampling is the MCMC method that feels most like cheating. If the joint posterior of several parameters is hard, but each parameter's distribution *given all the others* is easy, you just cycle through those easy conditionals, and the sequence of draws converges to the joint posterior. In the third Bayesian Learning lab (pair work) we used it on daily precipitation data: first a plain normal model, then a two-component normal mixture.

## The basic idea

Say the parameters are $\theta = (\theta_1, \theta_2)$. One Gibbs iteration is:

1. draw $\theta_1^{(t)} \sim p(\theta_1 \mid \theta_2^{(t-1)}, y)$,
2. draw $\theta_2^{(t)} \sim p(\theta_2 \mid \theta_1^{(t)}, y)$.

Each conditional is called a *full conditional*. The draws form a Markov chain whose stationary distribution is the joint posterior $p(\theta_1, \theta_2 \mid y)$. Every draw is accepted, so there's no tuning, unlike [random-walk Metropolis](/notes/random-walk-metropolis/).

## Normal model with a semi-conjugate prior

Model $y_i \sim N(\mu, \sigma^2)$ with *independent* priors $\mu \sim N(\mu_0, \tau_0^2)$ and $\sigma^2 \sim \text{Inv-}\chi^2(\nu_0, \sigma_0^2)$. Because the priors are independent (not the fully conjugate version), the joint posterior isn't a standard distribution. But the full conditionals are:

$$
\mu \mid \sigma^2, y \sim N(\mu_n, \tau_n^2), \quad \frac{1}{\tau_n^2} = \frac{n}{\sigma^2} + \frac{1}{\tau_0^2}, \quad \mu_n = w\bar y + (1-w)\mu_0, \quad w = \frac{n/\sigma^2}{n/\sigma^2 + 1/\tau_0^2},
$$

$$
\sigma^2 \mid \mu, y \sim \text{Inv-}\chi^2\!\left(\nu_0 + n,\; \frac{\nu_0\sigma_0^2 + \sum_{i=1}^n (y_i - \mu)^2}{\nu_0 + n}\right).
$$

Here $n$ is the sample size, $\bar y$ the sample mean, and $w$ the weight the data get relative to the prior (data precision over total precision). Alternate these two draws and you're done.

To check convergence, look at trace plots (each parameter against iteration: you want a "hairy caterpillar" with no trends) and cumulative means that settle down. With large $n$ this chain mixes almost instantly, because $\mu$ and $\sigma^2$ are nearly independent in the posterior.

## Mixtures and data augmentation

Precipitation data are skewed: lots of small values, a long tail of big ones. One normal fits that badly. A two-component mixture is more flexible:

$$
p(y_i) = \pi\, N(y_i \mid \mu_1, \sigma_1^2) + (1-\pi)\, N(y_i \mid \mu_2, \sigma_2^2),
$$

where $\pi$ is the weight of component 1. The likelihood is a product of sums, which is unpleasant. The trick is to invent a hidden label $z_i \in \{1, 2\}$ for each observation saying which component it came from. Given the labels, everything splits into two ordinary normal models. Given the parameters, each label has a simple probability. So the Gibbs sampler alternates:

1. $\pi \mid z \sim \text{Beta}(\alpha_1 + n_1, \alpha_2 + n_2)$, where $n_k$ is how many points currently have label $k$.
2. For each component $k$: draw $\mu_k$ and $\sigma_k^2$ from the normal-model conditionals above, **using only the points with $z_i = k$**.
3. For each observation: $\Pr(z_i = 1 \mid \cdot) = \frac{\pi N(y_i \mid \mu_1, \sigma_1^2)}{\pi N(y_i \mid \mu_1, \sigma_1^2) + (1-\pi) N(y_i \mid \mu_2, \sigma_2^2)}$.

Adding latent variables to make conditionals easy is called *data augmentation*. The same idea is behind EM for mixtures.

**A bug in my draft.** In step 2, my code computed each component's variance update with `sum((x - mu1)^2)` over *all* observations instead of just the ones assigned to that component. The mean update used the right subset, the variance didn't. That makes both variances look like the overall variance and blurs the two components together. It's an easy slip because the normal-model code is copied and only half-adapted.

## Code

```r
set.seed(5)
y <- c(rnorm(300, 10, 5), rnorm(200, 50, 20)); n <- length(y)
mu0 <- mean(y); tau2_0 <- 100^2; nu0 <- 4; s2_0 <- var(y) / 4; alpha <- c(1, 1)
rinvchisq <- function(df, s2) df * s2 / rchisq(1, df)

z <- sample(1:2, n, replace = TRUE); mu <- quantile(y, c(.25, .75)); s2 <- rep(var(y), 2)
keep <- matrix(NA, 2000, 5)
for (it in 1:2000) {
  nk <- tabulate(z, 2)
  w  <- rgamma(2, alpha + nk); pi1 <- w[1] / sum(w)           # Beta draw via gammas
  for (k in 1:2) {
    yk <- y[z == k]                                            # only this component's points
    prec  <- nk[k] / s2[k] + 1 / tau2_0
    mu[k] <- rnorm(1, (sum(yk) / s2[k] + mu0 / tau2_0) / prec, sqrt(1 / prec))
    s2[k] <- rinvchisq(nu0 + nk[k], (nu0 * s2_0 + sum((yk - mu[k])^2)) / (nu0 + nk[k]))
  }
  p1 <- pi1 * dnorm(y, mu[1], sqrt(s2[1]))
  p2 <- (1 - pi1) * dnorm(y, mu[2], sqrt(s2[2]))
  z  <- ifelse(runif(n) < p1 / (p1 + p2), 1, 2)                # new labels
  keep[it, ] <- c(pi1, mu, s2)
}
round(colMeans(keep[-(1:500), ]), 2)   # 0.61 10.14 50.11 27.95 406.21
```

The true values were $\pi = 0.6$, means 10 and 50, variances 25 and 400, so the sampler recovers them after a burn-in.

## Two things to watch with mixtures

- **Plot the mixture, not the components.** To compare with a histogram, the fitted density is $\pi N(\mu_1, \sigma_1^2) + (1-\pi)N(\mu_2, \sigma_2^2)$ on one curve. Plotting the two normals separately (as in my draft) doesn't show the fit, and neither does a normal with the data's own mean and variance.
- **Label switching.** Nothing in the model says which component is "1". In long runs, or with symmetric priors, the chain can swap the labels, and then posterior means of $\mu_1$ average over both components. Starting values that separate the components and checking trace plots help; an ordering constraint like $\mu_1 < \mu_2$ is the usual fix.

## Key takeaways

- Gibbs sampling: cycle through full conditionals; every draw is accepted.
- Semi-conjugate priors give easy conditionals even when the joint posterior isn't standard.
- Mixtures become easy by adding a hidden label per observation (data augmentation).
- Each component's updates must use only the points currently assigned to it.
- Watch for label switching and compare the full mixture density with the data.

## Resources

- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Section 11.1 (Gibbs sampler) and Chapter 22 (finite mixture models): http://www.stat.columbia.edu/~gelman/book/
- Stuart Geman, Donald Geman, "Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images", *IEEE TPAMI* 6(6) (1984): https://doi.org/10.1109/TPAMI.1984.4767596
- Matthew Stephens, "Dealing with label switching in mixture models", *JRSS B* 62(4) (2000): https://doi.org/10.1111/1467-9868.00265
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
