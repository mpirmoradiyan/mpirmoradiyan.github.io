---
title: "Grid approximation: the posterior of a one-parameter model by brute force"
slug: grid-approximation-of-a-posterior
date: 2020-04-14
topic: Bayesian Methods
tags: [university-lab, bayesian-inference, grid-approximation, von-mises, directional-data]
summary: "How to get the posterior of a single parameter in any model, conjugate or not, by evaluating prior times likelihood on a fine grid, and how to normalise it so it's a real density."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Data Analysis (3rd ed.), Chapter 10 (introduction to computation)"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book
  - title: "Von Mises distribution (Wikipedia)"
    author: "Wikipedia"
    url: "https://en.wikipedia.org/wiki/Von_Mises_distribution"
    type: docs


status: draft
---

Most models don't have a neat conjugate posterior. But if there's only one unknown parameter, you don't need anything clever: evaluate prior times likelihood on a fine grid of values and you have the posterior, up to a constant. In the first Bayesian Learning lab (pair work) we did this for wind directions modelled with a von Mises distribution, which is a good example because nothing about it is conjugate-friendly.

## The idea

Bayes' rule says

$$
p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)} \propto p(y \mid \theta)\, p(\theta),
$$

where $p(y \mid \theta)$ is the likelihood, $p(\theta)$ the prior and $p(y)$ the normalising constant (the evidence). The hard part is always $p(y) = \int p(y \mid \theta) p(\theta)\, d\theta$. On a grid $\theta_1, \dots, \theta_G$ with spacing $\Delta$, the integral becomes a sum:

$$
p(y) \approx \sum_{g=1}^{G} p(y \mid \theta_g)\, p(\theta_g)\, \Delta.
$$

So the recipe is: compute the unnormalised posterior at every grid point, then divide by (sum × spacing). After that, the curve integrates to 1 and you can read off the mode, mean, quantiles and so on.

## The von Mises example

Angles live on a circle, so a normal distribution doesn't fit them. The von Mises distribution is the circular version:

$$
p(y \mid \mu, \kappa) = \frac{\exp[\kappa \cos(y - \mu)]}{2\pi I_0(\kappa)}, \qquad -\pi \le y \le \pi,
$$

where $\mu$ is the mean direction, $\kappa > 0$ is the concentration (large $\kappa$ means the angles cluster tightly around $\mu$), and $I_0$ is the modified Bessel function of order zero (`besselI(k, 0)` in R). With $\mu$ known and an exponential prior $\kappa \sim \text{Exp}(\lambda)$, the posterior for $n$ independent angles is

$$p(\kappa \mid y) \propto \frac{\exp\!\left[\kappa \sum_{i=1}^n \cos(y_i - \mu)\right]}{\left(2\pi I_0(\kappa)\right)^n}\; \lambda e^{-\lambda\kappa}
\;\propto\; \frac{\exp\!\left[\kappa \left(\sum_i \cos(y_i - \mu) - \lambda\right)\right]}{I_0(\kappa)^n}.$$

No standard distribution has this shape, but it's one line to evaluate.

## Two traps we hit

**A misplaced parenthesis.** With $\lambda = 1$ the exponent is $\kappa\left(\sum_i \cos(y_i-\mu) - 1\right)$. The code in our submitted report had `exp(k * (sum(cos(y - mu)) - k))`, which is $\kappa\sum\cos - \kappa^2$. That's a different prior (a half-normal-like penalty instead of an exponential one), so the posterior and its mode were off. The formula in the text was right; the code didn't match it. Working on the log scale and adding `log_lik + log_prior` as two separate terms makes this kind of slip much harder.

**Not normalising properly.** We first plotted the unnormalised curve. Dividing by `sum(post)` alone isn't enough either: that gives probabilities per grid cell, which depend on how fine the grid is. To get a density you divide by `sum(post) * spacing`.

## Code

```r
y  <- c(2.1, 2.5, 1.9, 2.4, -2.9, 2.2)   # made-up angles in radians
mu <- 2.3
dk <- 0.01
kappa <- seq(0.01, 10, by = dk)

log_lik   <- sapply(kappa, function(k)
  k * sum(cos(y - mu)) - length(y) * log(2 * pi * besselI(k, 0)))
log_prior <- dexp(kappa, rate = 1, log = TRUE)
log_post  <- log_lik + log_prior

post <- exp(log_post - max(log_post))    # subtract the max to avoid underflow
post <- post / (sum(post) * dk)          # now it integrates to 1

kappa[which.max(post)]                   # posterior mode: 2.17
sum(kappa * post) * dk                   # posterior mean: 2.66
```

Notice the angle $-2.9$: on a circle it's close to $\pi$, so it's not far from the others at all. The cosine handles that automatically, which is the whole point of using a circular distribution.

## When the grid stops working

A grid with 1,000 points per parameter is fine in one dimension. In two dimensions it's a million evaluations, in five it's $10^{15}$. That's why the rest of the course moves to normal approximations and MCMC. Still, for a single parameter (or for checking an MCMC sampler on a 1-D slice), the grid is the most honest tool there is: no approximation except the grid spacing.

Two practical tips:

- Make sure the grid covers all the mass. If the curve isn't near zero at both ends, widen the range.
- Always compute on the log scale. Likelihoods of even moderate datasets underflow to 0 in double precision.

## Key takeaways

- For one parameter, the posterior is just prior × likelihood evaluated on a grid, normalised by (sum × spacing).
- Normalising by the sum alone gives cell probabilities, not a density.
- Work with log-likelihood and log-prior as separate terms; it avoids underflow and typos in the exponent.
- Grids explode with dimension, which is the motivation for Laplace approximations and MCMC.

## Resources

- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Section 10.1 (numerical integration and grid methods): http://www.stat.columbia.edu/~gelman/book/
- Richard McElreath, *Statistical Rethinking*, 2nd ed., Section 2.4.3 (grid approximation): https://xcelab.net/rm/
- Wikipedia, "Von Mises distribution": https://en.wikipedia.org/wiki/Von_Mises_distribution
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
