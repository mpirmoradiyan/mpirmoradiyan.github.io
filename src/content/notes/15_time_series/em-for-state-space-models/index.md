---
title: "Learning noise variances in a state space model with EM"
slug: em-for-state-space-models
date: 2020-09-25
topic: Time Series & State Space
tags: [university-lab, em-algorithm, maximum-likelihood, state-space-models, kalman-smoother, time-series]
summary: "What the EM algorithm actually maximizes, why the M-step for noise variances in a linear Gaussian state space model is just 'average the smoothed squared noise', and why the trend variance is often fixed by hand."
sources:
  - title: "732A80 Time Series and Sequence Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "An approach to time series smoothing and forecasting using the EM algorithm"
    author: "Robert H. Shumway, David S. Stoffer (Journal of Time Series Analysis, 1982)"
    url: "https://doi.org/10.1111/j.1467-9892.1982.tb00349.x"
    type: paper
  - title: "Time Series Analysis by State Space Methods (2nd ed.), Section 7.3.4"
    author: "James Durbin, Siem Jan Koopman"
    type: book
  - title: "Introduction to Time Series Modeling, Section 12.3"
    author: "Genshiro Kitagawa"
    type: book


status: draft
---

A [structural time series model](/notes/structural-time-series-models/) has noise variances that decide how fast the trend and season can change and how noisy the observations are. Guess them badly and the [Kalman filter](/notes/kalman-filter/) gives poor predictions. The EM algorithm learns them from data, and for this model each update has a short closed form. In my course lab I learned the seasonal and observation noise variances on the sea-level data this way. I also got the objective wrong on paper, which is the main thing I want to fix here.

## What EM is trying to maximize

We want the maximum likelihood estimate of the parameters $\theta$, here $\theta = (\sigma_\gamma^2, \sigma_\varepsilon^2)$:

$$
\hat\theta = \arg\max_\theta \log p(y_{1:n} \mid \theta).
$$

That's the *marginal* likelihood of the observed data, with the hidden states integrated out. The filter can evaluate it, but maximizing it directly is awkward. EM is a way to climb it.

In my lab report I wrote the objective as maximizing $\log p(\alpha_{1:n}, y_{1:n} \mid y_{1:n}, \theta)$. That doesn't make sense: it conditions on the data and includes the data as an unknown at the same time, and it puts the hidden states in the objective as if we observed them. The states are what EM *averages over*, not what it maximizes.

## The two steps

EM works with the *complete-data* log-likelihood $\log p(\alpha_{1:n}, y_{1:n} \mid \theta)$: what the log-likelihood would be if we could see the states. Since we can't, we take its expectation under our current best guess of the states.

**E-step.** With the current parameters $\theta_r$, compute

$$
\mathcal{Q}(\theta; \theta_r) = \mathbb{E}\big[\log p(\alpha_{1:n}, y_{1:n} \mid \theta) \,\big|\, y_{1:n}, \theta_r\big].
$$

The expectation is over the smoothed distribution of the states, $p(\alpha_{1:n} \mid y_{1:n}, \theta_r)$. In practice this means: run the Kalman filter and the [smoother](/notes/kalman-smoother/) with $\theta_r$.

**M-step.** Set $\theta_{r+1} = \arg\max_\theta \mathcal{Q}(\theta; \theta_r)$.

The key property is that each iteration never decreases $\log p(y_{1:n} \mid \theta)$. So you can monitor the filter's log-likelihood across iterations as a sanity check: it should go up and level off. If it goes down, there's a bug.

## The M-step in closed form

For a linear Gaussian model, the complete-data log-likelihood splits into Gaussian terms for each noise sequence. The part that involves the observation noise is

$$
-\frac{n}{2}\log\sigma_\varepsilon^2 - \frac{1}{2\sigma_\varepsilon^2}\sum_{t=1}^n \varepsilon_t^2, \qquad \varepsilon_t = y_t - Z\alpha_t.
$$

Taking the expectation replaces $\varepsilon_t^2$ with $\mathbb{E}[\varepsilon_t^2 \mid y_{1:n}] = \hat\varepsilon_{t|n}^2 + \text{Var}(\varepsilon_t \mid y_{1:n})$, the squared smoothed mean plus the smoothed variance. Setting the derivative to zero:

$$
\sigma_\varepsilon^2 = \frac{1}{n}\sum_{t=1}^{n}\left(\hat\varepsilon_{t|n}^2 + \text{Var}(\varepsilon_t \mid y_{1:n})\right).
$$

The seasonal state noise works the same way. With $\eta_t = (\eta_t^\mu, \eta_t^\gamma)^\top$ and a diagonal $Q$, take the $\gamma$ entry:

$$
\sigma_\gamma^2 = \frac{1}{n-1}\sum_{t=1}^{n-1}\left( (\hat\eta^\gamma_{t|n})^2 + \text{Var}(\eta^\gamma_t \mid y_{1:n}) \right).
$$

There are $n-1$ state noise terms because $\eta_t$ links $\alpha_t$ to $\alpha_{t+1}$. (I divided by $n$ in my lab. With 800 training points that's a tiny difference, but $n-1$ is the right count.)

So basically: "the new variance is the average of the smoothed squared noise, plus a correction for how unsure we are about that noise". The variance term matters. Dropping it would systematically shrink the estimates, because the smoothed means are less spread out than the true noise.

```python
import numpy as np

def m_step(eps_hat, eps_var, eta_g_hat, eta_g_var):
    """Inputs are smoother outputs for the observation noise (length n)
    and for the seasonal state noise (length n-1)."""
    sigma2_eps = np.mean(eps_hat**2 + eps_var)
    sigma2_gamma = np.mean(eta_g_hat**2 + eta_g_var)
    return sigma2_gamma, sigma2_eps

# EM loop sketch:
# for r in range(100):
#     run Kalman filter + smoother with (sigma2_gamma, sigma2_eps)   # E-step
#     record the filter log-likelihood (should not decrease)
#     sigma2_gamma, sigma2_eps = m_step(...)                          # M-step
```

## What happened on the sea-level data

I started from arbitrary values ($\sigma_\gamma = 1$, $\sigma_\varepsilon = 1$, with $\sigma_\mu = 0.01$ fixed) and ran 100 iterations on the first 800 points. The estimates settled at about $\sigma_\gamma \approx 0.24$ and $\sigma_\varepsilon \approx 3.82$. The training log-likelihood went from about $-3191$ with the starting values to about $-1600$ with the learned ones (both without the constant $\log 2\pi$ term). So the data wanted much less seasonal drift and much more observation noise than my starting guess.

One more thing I'd do differently: to show convergence, plot each parameter and the log-likelihood against the iteration number. I tried a 3D "surface" of log-likelihood over the visited parameter values, but EM visits a path, not a grid, so that plot doesn't show a real likelihood surface.

## Why not learn the trend variance too?

It's just as easy to add $\sigma_\mu^2$ to the M-step. The catch is that the maximum likelihood value tends to be too large for good long-range forecasts. No simple trend + season model fits real data perfectly, and the likelihood makes up for the mismatch by letting the trend absorb short-term wiggles. A wiggly trend then extrapolates badly. Kitagawa suggests two fixes: fix $\sigma_\mu^2$ to a small value by hand (what we did), or add a stationary AR component to the model so that it soaks up the short-term structure instead of the trend.

## Key takeaways

- EM maximizes the marginal likelihood $\log p(y_{1:n} \mid \theta)$; the hidden states are averaged over, not maximized.
- E-step = Kalman filter + smoother with the current parameters.
- M-step for a noise variance = average of (smoothed mean squared + smoothed variance) of that noise.
- The log-likelihood must not decrease between iterations; use that as a bug check.
- Learning every variance by maximum likelihood can hurt long-range forecasts when the model is misspecified.

## Resources

- 732A80 Time Series and Sequence Learning, Linköping University (course lab 2, autumn 2020).
- Shumway & Stoffer, "An approach to time series smoothing and forecasting using the EM algorithm", *Journal of Time Series Analysis* 3(4), 1982. https://doi.org/10.1111/j.1467-9892.1982.tb00349.x
- Durbin & Koopman, *Time Series Analysis by State Space Methods*, 2nd ed., Oxford University Press, 2012, Section 7.3.4.
- Kitagawa, *Introduction to Time Series Modeling*, CRC Press, 2010, Section 12.3.
