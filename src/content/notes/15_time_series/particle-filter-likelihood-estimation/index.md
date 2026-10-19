---
title: "Estimating the likelihood with a particle filter, and using it to fit a parameter"
slug: particle-filter-likelihood-estimation
date: 2020-10-16
topic: Time Series & State Space
tags: [university-lab, particle-filter, sequential-monte-carlo, maximum-likelihood, epidemic-models, parameter-estimation]
summary: "How a bootstrap particle filter gives an estimate of the data likelihood as a by-product, why that estimate is noisy and unbiased only on the natural scale, and how I used it to grid-search one parameter of a stochastic SEIR model."
sources:
  - title: "732A80 Time Series and Sequence Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Elements of Sequential Monte Carlo, Sections 2–3"
    author: "Christian A. Naesseth, Fredrik Lindsten, Thomas B. Schön"
    url: "https://arxiv.org/abs/1903.04797"
    type: paper
  - title: "A Tutorial on Particle Filtering and Smoothing: Fifteen years later"
    author: "Arnaud Doucet, Adam M. Johansen"
    url: "https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf"
    type: paper
  - title: "Estimates of the peak-day and the number of infected individuals during the covid-19 outbreak in the Stockholm region, Sweden February – April 2020"
    author: "Folkhälsomyndigheten (Public Health Agency of Sweden)"
    url: "https://www.folkhalsomyndigheten.se/publicerat-material/publikationsarkiv/e/estimates-of-the-peak-day-and-the-number-of-infected-individuals-during-the-covid-19-outbreak-in-the-stockholm-region-sweden-february--april-2020/"
    type: paper


status: draft
---

Here was my problem in the lab: I had a simulator-style epidemic model with one unknown parameter, and I wanted the value that best explains the data. Maximum likelihood needs $p(y_{1:n} \mid \theta)$. For a linear Gaussian model the [Kalman filter](/notes/kalman-filter/) hands you that number exactly. For a nonlinear model there's no formula. The nice surprise: a [particle filter](/notes/particle-filters-sis-and-bootstrap/) gives a random estimate of it almost for free. I used this to pick the transmission parameter of a simplified stochastic SEIR model fitted to daily ICU admissions in Stockholm in 2020. The course was clear that the model was a toy, so the point here is the method, not the epidemiology.

## The model in one paragraph

The hidden state holds the number of susceptible ($S$), exposed ($E$) and infectious ($I$) people, plus a log contact rate $z_t$ that does a Gaussian random walk. Each day, binomial draws move people $S \to E \to I \to R$, with the $S \to E$ probability growing with $\rho\, e^{z_t} I_t / \text{population}$, where $\rho$ is a transmission parameter. The observation is the number of new ICU cases, drawn as $\text{Binomial}(I_t, p_{ic})$ with a small probability $p_{ic}$. So the observations are small non-negative counts. That's one reason a Gaussian observation model would be a poor fit: it allows negative and non-integer values, and its spread doesn't grow with the level the way count data does.

## The likelihood estimate

Write the likelihood with the chain rule, $p(y_{1:n}) = \prod_t p(y_t \mid y_{1:t-1})$. In a bootstrap particle filter, at step $t$ the particles $\alpha_t^i$ are (approximately) draws from $p(\alpha_t \mid y_{1:t-1})$, before we look at $y_t$. So the average of their unnormalized weights estimates the one-step predictive density:

$$
p(y_t \mid y_{1:t-1}) = \int p(y_t \mid \alpha_t)\, p(\alpha_t \mid y_{1:t-1})\, d\alpha_t \approx \frac{1}{N}\sum_{i=1}^N w_t^i, \qquad w_t^i = p(y_t \mid \alpha_t^i).
$$

Multiplying over time:

$$
\hat Z = \prod_{t=1}^n \left( \frac{1}{N}\sum_{i=1}^N w_t^i \right), \qquad \log \hat Z = \sum_{t=1}^n \log\left( \frac{1}{N}\sum_{i=1}^N w_t^i \right).
$$

In code it's one extra line in the filter loop, done in log space for safety. The $-\log N$ is easy to forget. It's what turns the sum of weights into an average.

```python
import numpy as np

def log_mean_exp(logw):
    c = logw.max()
    return c + np.log(np.mean(np.exp(logw - c)))

# inside the bootstrap particle filter loop, after computing log-weights:
#     logZ += log_mean_exp(logw)          # log of (1/N) * sum_i w_t^i
#     W = np.exp(logw - logw.max()); W /= W.sum()
```

## It's random, so look at its spread

Run the filter twice and you get two different $\log \hat Z$ values. In the lab I first ran the filter 20 times with $N = 200$ at the default $\rho = 0.3$ and drew a box plot of the log-likelihood estimates. Seeing that spread is important before using the estimate to compare parameter values: a difference smaller than the spread doesn't mean much.

The variance shrinks as $N$ grows, and grows with the length of the series. A rough rule from the theory is that the variance of $\log \hat Z$ scales like $n/N$, so a longer series needs more particles for the same precision.

## Unbiased, but on which scale?

A nice and slightly surprising fact: $\hat Z$ is an *unbiased* estimate of the likelihood, $\mathbb{E}[\hat Z] = p(y_{1:n} \mid \theta)$, for any $N$. This is what makes particle MCMC methods work.

But $\log \hat Z$ is *not* unbiased for $\log p(y_{1:n})$. Because $\log$ is concave, Jensen's inequality gives

$$
\mathbb{E}[\log \hat Z] \le \log \mathbb{E}[\hat Z] = \log p(y_{1:n} \mid \theta).
$$

So when you average several $\log \hat Z$ values, as I did, the result is biased downward, and the bias is bigger where the estimate is noisier. To estimate the log-likelihood itself from $K$ runs, the log of the average is better: $\log\big(\frac{1}{K}\sum_k \hat Z_k\big)$, computed with log-mean-exp. For finding the maximum it often doesn't matter much, but if the noise level changes a lot across parameter values, the downward bias can shift where the peak appears.

## Grid search over one parameter

With an estimated likelihood in hand, the simplest way to fit one parameter is a grid. I used 50 values of $\rho$ between 0.1 and 0.9, and for each one averaged $\log \hat Z$ over 30 runs with $N = 1000$ particles. The averaged curve peaked at $\rho \approx 0.57$ with an average log-likelihood of about $-394$. For comparison, a single run at the default $\rho = 0.3$ with $N = 200$ gave about $-433$. So the data clearly preferred faster transmission than the default, within this toy model.

A few things that help with this kind of search:

- **Average or increase $N$.** A single run per grid point gives a jagged curve with a misleading argmax.
- **Reuse random numbers.** Using the same random seed at every grid point (common random numbers) makes neighbouring estimates noisy in a similar way, so their *differences* are less noisy. Resampling breaks this partly, because a small change in $\rho$ can change which particles get copied.
- **Beyond one parameter.** Grids get expensive fast. Then you'd move to particle MCMC (which relies on the unbiasedness above) or to methods that optimize a smoothed likelihood estimate.

## Key takeaways

- A bootstrap particle filter estimates $p(y_t \mid y_{1:t-1})$ as the average unnormalized weight; multiply over time to get $\hat Z$.
- Don't forget the $1/N$, and do everything in log space.
- $\hat Z$ is unbiased for the likelihood, but $\log \hat Z$ is biased downward; average on the natural scale if you want the log-likelihood.
- Check the spread of the estimate before comparing parameter values; average runs, use more particles, or use common random numbers.
- A grid over one parameter is a fine first step for a simulator-based model.

## Resources

- 732A80 Time Series and Sequence Learning, Linköping University (course lab 3, autumn 2020).
- Naesseth, Lindsten & Schön, "Elements of Sequential Monte Carlo", 2019. https://arxiv.org/abs/1903.04797
- Doucet & Johansen, "A Tutorial on Particle Filtering and Smoothing: Fifteen years later", 2009/2011. https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
- Folkhälsomyndigheten report on covid-19 estimates for the Stockholm region (basis of the SEIR model used in the lab). https://www.folkhalsomyndigheten.se/publicerat-material/publikationsarkiv/e/estimates-of-the-peak-day-and-the-number-of-infected-individuals-during-the-covid-19-outbreak-in-the-stockholm-region-sweden-february--april-2020/
