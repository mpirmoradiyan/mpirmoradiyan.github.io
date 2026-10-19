---
title: "Bayesian AR(1) in Stan: why the mean gets lost when phi nears 1, and latent AR for counts"
slug: bayesian-ar1-in-stan
date: 2020-05-17
topic: Bayesian Methods
tags: [university-lab, bayesian-inference, time-series, autoregressive-model, stan, poisson-regression]
summary: "How the AR(1) coefficient controls memory and stationarity, why the process mean becomes almost impossible to estimate as phi approaches 1, and how to put a latent AR(1) process under Poisson counts in Stan."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Stan User's Guide: Time-Series Models (autoregressive models)"
    author: "Stan Development Team"
    url: "https://mc-stan.org/docs/stan-users-guide/time-series.html"
    type: docs
  - title: "Time Series Analysis and Its Applications (4th ed.), Chapter 3"
    author: "Robert H. Shumway, David S. Stoffer"
    url: "https://www.stat.pitt.edu/stoffer/tsa4/"
    type: book


status: draft
---

The last Bayesian Learning lab (pair work) was our first real use of Stan. We simulated AR(1) series, tried to recover their parameters, and then used an AR(1) process as a hidden layer under count data. The result that stuck with me: with the same amount of data, one parameter was estimated to within $\pm 0.3$ for one series and was basically unknown for the other. The reason is worth understanding.

## The AR(1) process

$$
x_t = \mu + \phi\,(x_{t-1} - \mu) + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma^2),
$$

where $\mu$ is the level the series is pulled towards, $\phi$ controls how much of yesterday's deviation from $\mu$ survives to today, and $\sigma^2$ is the variance of each new shock.

- $\phi = 0$: no memory, just noise around $\mu$.
- $\phi$ close to 1: long memory. The series wanders away from $\mu$ and comes back slowly.
- $\phi$ close to $-1$: the deviation flips sign each step, so the series zig-zags.
- $|\phi| < 1$ is needed for the process to be *stationary*. Then its long-run mean is $\mu$ and its long-run variance is $\sigma^2/(1-\phi^2)$, which blows up as $|\phi| \to 1$. At $\phi = 1$ it's a random walk and $\mu$ drops out of the equation entirely.

## Why $\mu$ gets lost

We simulated two series of length 200, one with $\phi = 0.3$ and one with $\phi = 0.95$, and fit both in Stan. For the first, the posterior for $\mu$ was tight: mean 9.94 with a 95% interval of about $(9.65, 10.24)$ around the true 10, and $\phi$ came out at 0.28. For the second, $\phi$ was estimated fine (about 0.98), but the 95% interval for $\mu$ ran from about $-80$ to $80$.

Our report noted that $\mu$ was "harder to estimate" without saying why. Here's why. Information about the mean comes from seeing the series cross back and forth over it many times. With strong persistence, 200 correlated points behave like very few independent ones. The long-run standard deviation of the sample mean is roughly

$$
\text{sd}(\bar x) \approx \frac{\sigma}{(1-\phi)\sqrt{T}},
$$

where $T$ is the series length. The factor $1/(1-\phi)$ is about 1.4 for $\phi = 0.3$ but 20 for $\phi = 0.95$. A quick simulation confirms it:

```r
sim_ar1 <- function(T, mu, phi, sigma) {
  x <- numeric(T); x[1] <- mu
  for (t in 2:T) x[t] <- mu + phi * (x[t - 1] - mu) + rnorm(1, 0, sigma)
  x
}
set.seed(7); T <- 200; sigma <- sqrt(2)
for (phi in c(0.3, 0.95)) {
  means <- replicate(2000, mean(sim_ar1(T, mu = 10, phi = phi, sigma = sigma)))
  cat(sprintf("phi = %.2f  sd of sample mean: simulated %.2f, formula %.2f\n",
              phi, sd(means), sigma / ((1 - phi) * sqrt(T))))
}
# phi = 0.30  sd of sample mean: simulated 0.14, formula 0.14
# phi = 0.95  sd of sample mean: simulated 1.87, formula 2.00
```

That explains a standard deviation of about 2 for $\mu$. An interval of $\pm 80$ needs one more ingredient: the model behind those numbers didn't constrain $\phi$ to $(-1, 1)$, and its posterior for $\phi$ reached past 1 (upper limit 1.007). Near and beyond $\phi = 1$ the term $(1-\phi)\mu$ that ties the series to its mean vanishes, so the likelihood becomes flat in $\mu$ and the posterior of $\mu$ is left to the prior, which was vague. Stan also reported divergent transitions in this region, its way of saying the posterior geometry is awkward. Declaring `real<lower=-1, upper=1> phi;` (as in my own Stan file) keeps the sampler out of that region.

The lesson I took: a wide posterior isn't a sampler failure. It's the data honestly saying they contain little information about that parameter.

## A latent AR(1) under counts

The second part used a series of infection counts. Counts need a Poisson model, but the rate clearly drifts over time. A neat way to model that is to let the log-rate follow an AR(1):

$$
c_t \mid x_t \sim \text{Poisson}\!\left(e^{x_t}\right), \qquad x_t = \mu + \phi(x_{t-1} - \mu) + \varepsilon_t.
$$

Here $c_t$ is the observed count and $x_t$ the hidden log-intensity. This is a simple *state-space model*: a hidden process with its own dynamics, seen through a noisy observation model. In Stan the hidden $x_t$ are just more parameters:

```stan
data { int<lower=0> N; array[N] int<lower=0> c; }
parameters {
  real mu;
  real<lower=-1, upper=1> phi;
  real<lower=0> sigma;
  vector[N] x;
}
model {
  mu ~ normal(0, 10);
  sigma ~ normal(0, 1);
  x[1] ~ normal(mu, sigma / sqrt(1 - phi^2));          // stationary start
  x[2:N] ~ normal(mu + phi * (x[1:(N - 1)] - mu), sigma);
  c ~ poisson_log(x);                                   // rate = exp(x)
}
```

Two details I'd fix compared with our lab files. First, we left $x_1$ without any distribution, which gives it an improper flat prior; drawing it from the stationary distribution is cleaner. Second, `poisson_log(x)` is the numerically stable way to write `poisson(exp(x))`. Plotting the posterior mean of $e^{x_t}$ with a 95% band over the data shows a smooth intensity that follows the counts without chasing every spike. How smooth depends on the prior for $\sigma$: a tight prior on small values forces a smoother curve.

## Key takeaways

- In AR(1), $\phi$ is memory; $|\phi| < 1$ means stationary, with variance $\sigma^2/(1-\phi^2)$.
- As $\phi \to 1$, the mean $\mu$ is weakly identified: $\text{sd}(\bar x) \approx \sigma / ((1-\phi)\sqrt T)$.
- Constrain $\phi$ to $(-1, 1)$ if you want a stationary model; otherwise $\mu$ can become completely unidentified.
- A latent AR(1) on the log scale is an easy way to model counts with a drifting rate.
- Give the first latent state a proper (stationary) distribution.

## Resources

- Stan User's Guide, "Time-Series Models": https://mc-stan.org/docs/stan-users-guide/time-series.html
- Stan Functions Reference, `poisson_log`: https://mc-stan.org/docs/functions-reference/
- Robert H. Shumway, David S. Stoffer, *Time Series Analysis and Its Applications*, 4th ed., Chapter 3: https://www.stat.pitt.edu/stoffer/tsa4/
- Michael Betancourt, "A conceptual introduction to Hamiltonian Monte Carlo" (2017), on divergences: https://arxiv.org/abs/1701.02434
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
