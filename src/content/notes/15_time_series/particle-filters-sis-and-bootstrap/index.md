---
title: "Particle filters: sequential importance sampling, weight collapse and resampling"
slug: particle-filters-sis-and-bootstrap
date: 2020-10-16
topic: Time Series & State Space
tags: [university-lab, particle-filter, sequential-monte-carlo, importance-sampling, state-space-models, effective-sample-size]
summary: "How a particle filter tracks the hidden state of a nonlinear, non-Gaussian model with weighted samples, why plain sequential importance sampling collapses, and how resampling (the bootstrap particle filter) fixes it."
sources:
  - title: "732A80 Time Series and Sequence Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "A Tutorial on Particle Filtering and Smoothing: Fifteen years later"
    author: "Arnaud Doucet, Adam M. Johansen"
    url: "https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf"
    type: paper
  - title: "Elements of Sequential Monte Carlo"
    author: "Christian A. Naesseth, Fredrik Lindsten, Thomas B. Schön"
    url: "https://arxiv.org/abs/1903.04797"
    type: paper
  - title: "Novel approach to nonlinear/non-Gaussian Bayesian state estimation"
    author: "N. J. Gordon, D. J. Salmond, A. F. M. Smith (IEE Proceedings F, 1993)"
    url: "https://doi.org/10.1049/ip-f-2.1993.0015"
    type: paper


status: draft
---

The [Kalman filter](/notes/kalman-filter/) is exact, but only for linear models with Gaussian noise. Many real models aren't like that. In my course lab the model was a stochastic SEIR epidemic model, where the states are counts of people (susceptible, exposed, infectious) that move between groups with binomial randomness, and the observations are daily ICU admissions. No closed-form filter exists. A particle filter replaces the Gaussian belief with a cloud of weighted samples ("particles") and pushes that cloud through time.

## The setting

A general state space model has a transition density and an observation density:

$$
\alpha_t \sim p(\alpha_t \mid \alpha_{t-1}), \qquad y_t \sim p(y_t \mid \alpha_t).
$$

We want the filtering distribution $p(\alpha_t \mid y_{1:t})$. A particle filter approximates it by $N$ samples $\alpha_t^i$ with normalized weights $W_t^i$:

$$
p(\alpha_t \mid y_{1:t}) \approx \sum_{i=1}^N W_t^i\, \delta_{\alpha_t^i}(\alpha_t), \qquad \mathbb{E}[\alpha_t \mid y_{1:t}] \approx \sum_{i=1}^N W_t^i \alpha_t^i.
$$

$\delta_{\alpha_t^i}$ is a point mass at particle $i$. All we need is to be able to *sample* from the transition and *evaluate* the observation density. That's exactly what a simulator like the SEIR model gives you.

## Sequential importance sampling (SIS)

Importance sampling: draw samples from a distribution you can sample from (the *proposal*), then reweight them by "target density / proposal density". In the simplest version, the proposal is the model itself. Each particle's whole path is simulated from the dynamics, and its weight measures how well that path explains the data:

$$
w_t^i = w_{t-1}^i \cdot p(y_t \mid \alpha_t^i), \qquad W_t^i = \frac{w_t^i}{\sum_j w_t^j}.
$$

The important part is the *product*: a particle's weight is the likelihood of the whole observed sequence along its path, $\prod_{s \le t} p(y_s \mid \alpha_s^i)$.

The problem is that this product collapses. After a few dozen steps, one path explains the data much better than all the others, and one weight is near 1 while the rest are near 0. You still pay for $N$ particles, but effectively you have one.

## Measuring collapse: effective sample size

The *effective sample size* is

$$
\text{ESS}_t = \frac{1}{\sum_{i=1}^N (W_t^i)^2}.
$$

With equal weights it's $N$. With all weight on one particle it's 1. For SIS you expect ESS to fall quickly toward 1 and stay there.

## The mistake in my SIS code

In my lab code, the SIS weight at each step was only $p(y_t \mid \alpha_t^i)$. I didn't multiply by (in log space: add) the previous weight. That isn't SIS. It weights each particle by the latest observation only, while the particles themselves are never corrected. So it's neither a proper importance sampler nor a resampling filter.

The ESS plot was the clue I missed at the time. My "SIS" ESS dropped early on and then climbed back up to around 90 out of 100 near the end of the series. With accumulated weights that shouldn't happen. My version recovered only because late in the series the observations were close to zero, and nearly every particle explained a zero equally well, so the latest-only weights became nearly equal. With proper accumulated weights, those equal late likelihoods would leave the old (hugely unequal) weight ratios unchanged, and ESS would stay low. The correct line is `logW[t] = logW[t-1] + log_lik(y[t], particles[t])`.

## Resampling: the bootstrap particle filter

The fix for collapse is to *resample*: before propagating, draw $N$ new particles from the current ones with probabilities $W_{t-1}^i$. Good particles get copied, bad ones disappear, and all weights become equal again. The bootstrap particle filter (BPF) does this at every step:

1. Resample indices $a^i \sim \text{Categorical}(W_{t-1}^{1:N})$.
2. Propagate: $\alpha_t^i \sim p(\alpha_t \mid \alpha_{t-1}^{a^i})$.
3. Weight: $w_t^i = p(y_t \mid \alpha_t^i)$, then normalize.

Here using only the current likelihood is correct, because resampling has already reset the old weights to equal. In my BPF the ESS stayed high for most of the run, and the filter means tracked the true simulated S, E, I and R curves much better than my SIS version did.

## Do it in log space

Likelihoods of counts can be tiny, so $\exp(\log w)$ underflows to zero. Subtract the maximum before exponentiating (the log-sum-exp trick):

```python
import numpy as np

def normalize_log_weights(logw):
    c = logw.max()
    w = np.exp(logw - c)              # largest weight becomes exp(0) = 1
    return w / w.sum()

def bootstrap_pf(y, sample_init, sample_next, log_lik, N, rng):
    """sample_init(N) -> (d, N); sample_next(particles) -> (d, N);
    log_lik(y_t, particles) -> (N,)."""
    particles = sample_init(N)
    means, ess = [], []
    for t in range(len(y)):
        if t > 0:
            idx = rng.choice(N, size=N, p=W)          # resample
            particles = sample_next(particles[:, idx])  # propagate
        W = normalize_log_weights(log_lik(y[t], particles))
        means.append(particles @ W)                   # filter mean
        ess.append(1.0 / np.sum(W**2))
    return np.array(means), np.array(ess)
```

Resampling isn't free: it adds a bit of extra randomness and reduces diversity (many copies of the same particle). That's why some implementations only resample when ESS drops below, say, $N/2$. And for a long series the *path* history still collapses to a few ancestors, which matters for smoothing but not for filtering.

## Key takeaways

- A particle filter represents $p(\alpha_t \mid y_{1:t})$ with weighted samples; it only needs to simulate the dynamics and evaluate the observation likelihood.
- SIS multiplies weights over time, and they collapse onto one particle; ESS shows this.
- Resampling at each step (bootstrap PF) resets weights and keeps the particle cloud useful.
- In SIS, accumulate log-weights; in the BPF, use only the current log-likelihood after resampling. Mixing these up gives a filter that is neither.
- Always normalize weights in log space.

## Resources

- 732A80 Time Series and Sequence Learning, Linköping University (course lab 3, autumn 2020).
- Doucet & Johansen, "A Tutorial on Particle Filtering and Smoothing: Fifteen years later", 2009/2011. https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
- Naesseth, Lindsten & Schön, "Elements of Sequential Monte Carlo", 2019. https://arxiv.org/abs/1903.04797
- Gordon, Salmond & Smith, "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", *IEE Proceedings F*, 1993. https://doi.org/10.1049/ip-f-2.1993.0015
- Next: [estimating the likelihood with a particle filter](/notes/particle-filter-likelihood-estimation/).
