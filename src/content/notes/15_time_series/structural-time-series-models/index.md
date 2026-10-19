---
title: "Structural time series: writing trend plus season as a state space model"
slug: structural-time-series-models
date: 2020-09-25
topic: Time Series & State Space
tags: [university-lab, time-series, state-space-models, seasonality, trend, structural-models]
summary: "How to describe a series as trend + seasonal + noise, write each part as a small linear state space model, and glue the parts together with block-diagonal matrices."
sources:
  - title: "732A80 Time Series and Sequence Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Time Series Analysis by State Space Methods (2nd ed.), Chapter 3"
    author: "James Durbin, Siem Jan Koopman"
    type: book
  - title: "Introduction to Time Series Modeling, Chapters 11–12"
    author: "Genshiro Kitagawa"
    type: book
  - title: "NASA Global Mean Sea Level data"
    author: "NASA"
    url: "https://climate.nasa.gov/vital-signs/sea-level/"
    type: docs


status: draft
---

With an AR model I had to remove the trend by hand before fitting anything. A structural time series model goes the other way: it models the trend and the seasonal pattern directly, as hidden components that can drift slowly over time. The trick that makes this practical is writing everything as one *linear Gaussian state space model*. After that, the Kalman filter and smoother do all the work. I built one of these in a course lab for the sea-level data.

## The decomposition

The idea is to say the observation is a sum of interpretable parts:

$$
y_t = \mu_t + \gamma_t + \varepsilon_t,
$$

where $\mu_t$ is the trend (the slow level), $\gamma_t$ is the seasonal component (a pattern that repeats every $s$ steps), and $\varepsilon_t \sim N(0, \sigma_\varepsilon^2)$ is observation noise. None of $\mu_t$ or $\gamma_t$ is observed. We only see their noisy sum.

## The state space form

A linear Gaussian state space model has two equations:

$$
\alpha_{t+1} = T\alpha_t + R\eta_t, \qquad \eta_t \sim N(0, Q),
$$
$$
y_t = Z\alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \sigma_\varepsilon^2).
$$

- $\alpha_t$ is the hidden *state*, a vector of dimension $d$.
- $T$ ($d \times d$) moves the state one step forward.
- $\eta_t$ is the state noise, of dimension $d_\eta$, with covariance $Q$ ($d_\eta \times d_\eta$).
- $R$ ($d \times d_\eta$) decides which states the noise enters. Often $d_\eta < d$, because some states are just copies of old values and get no noise.
- $Z$ ($1 \times d$) picks out the combination of states we observe.

The job is to find a $T$, $R$, $Z$ for each component.

## Trend component

A simple smooth trend says the trend's second difference is small noise:

$$
\mu_{t+1} = 2\mu_t - \mu_{t-1} + \eta^{\mu}_t.
$$

Without noise, this is a straight line: $\mu_{t+1} - \mu_t = \mu_t - \mu_{t-1}$, so the slope stays the same. The noise lets the slope wander slowly. To make it first-order, keep two values in the state, $\alpha^{\mu}_t = (\mu_t, \mu_{t-1})^\top$:

$$
T_\mu = \begin{bmatrix} 2 & -1 \\ 1 & 0 \end{bmatrix}, \quad R_\mu = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad Z_\mu = \begin{bmatrix} 1 & 0 \end{bmatrix}.
$$

The second row of $T_\mu$ just copies $\mu_t$ down so it's available as "the previous value" next time. The variance $\sigma_\mu^2$ of $\eta^\mu_t$ controls how bendy the trend is.

## Seasonal component

The seasonal effects over one full period should add up to roughly zero (otherwise part of the season is really trend). So:

$$
\gamma_{t+1} = -\sum_{j=0}^{s-2} \gamma_{t-j} + \eta^{\gamma}_t.
$$

If the noise is zero, the $s$ most recent seasonal values sum to exactly zero, and the pattern repeats every $s$ steps. The state holds the last $s-1$ values, $(\gamma_t, \gamma_{t-1}, \dots, \gamma_{t-s+2})^\top$, so its dimension is $s-1$. The matrices are

$$
T_\gamma = \begin{bmatrix} -1 & -1 & \cdots & -1 & -1 \\ 1 & 0 & \cdots & 0 & 0 \\ 0 & 1 & \cdots & 0 & 0 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & 0 \end{bmatrix}, \quad R_\gamma = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad Z_\gamma = \begin{bmatrix} 1 & 0 & \cdots & 0 \end{bmatrix}.
$$

Same pattern as before: the first row computes the new value, the rest shift old values down.

## Choosing the period $s$

The sea-level data has one observation about every 10 days, so $s$ should be the number of observations per year. Here I made a small mistake in the lab. I counted observations per calendar year and averaged, but I included the last year, which only had 2 observations. That pulled the mean down to about 35.6, and I rounded to $s = 36$. The complete years had 36 or 37 observations each, averaging about 36.85, so the right choice is $s = 37$. The time step in the data (about 0.0271 years) gives the same answer: $1/0.0271 \approx 36.8$. The lesson: when you count "per year", leave out partial years at the ends.

## Gluing components together

Independent components combine by stacking their states. Then $T$ and $R$ are block-diagonal, and the $Z$ rows are put side by side:

$$
T = \begin{bmatrix} T_\mu & 0 \\ 0 & T_\gamma \end{bmatrix}, \quad R = \begin{bmatrix} R_\mu & 0 \\ 0 & R_\gamma \end{bmatrix}, \quad Z = \begin{bmatrix} Z_\mu & Z_\gamma \end{bmatrix}, \quad Q = \begin{bmatrix} \sigma_\mu^2 & 0 \\ 0 & \sigma_\gamma^2 \end{bmatrix}.
$$

So $Z\alpha_t = \mu_t + \gamma_t$, which is what we wanted. This modularity is the nice part: adding another component (say, a stationary AR term to soak up short-term wiggles) is just one more block.

```python
import numpy as np
from scipy.linalg import block_diag

def trend_block():
    return np.array([[2., -1.], [1., 0.]]), np.array([[1.], [0.]]), np.array([[1., 0.]])

def seasonal_block(s):
    T = np.zeros((s - 1, s - 1))
    T[0, :] = -1.0                       # new value = minus sum of the last s-1
    T[1:, :-1] = np.eye(s - 2)           # shift old values down
    R = np.zeros((s - 1, 1)); R[0, 0] = 1.0
    return T, R, R.T.copy()

T_mu, R_mu, Z_mu = trend_block()
T_g, R_g, Z_g = seasonal_block(s=37)
T = block_diag(T_mu, T_g)                # (38, 38)
R = block_diag(R_mu, R_g)                # (38, 2)
Z = np.hstack([Z_mu, Z_g])               # (1, 38)
```

## The initial state

The model also needs a starting distribution $\alpha_1 \sim N(a_1, P_1)$. A sensible choice is to start the trend at the first observation with zero slope (so $\mu_1 = \mu_0 = y_1$ in the mean), set the seasonal states to zero, and use a large diagonal $P_1$ to say "I'm not sure". The filter quickly overrides a vague start.

## Key takeaways

- A structural model writes $y_t$ as trend + season + noise, each a hidden component that can drift.
- Each component is a tiny state space model; the "copy old values down" rows turn higher-order recursions into first-order ones.
- A seasonal component of period $s$ needs $s-1$ states, because the last value is fixed by the sum-to-zero rule.
- Components combine with block-diagonal $T$, $R$, $Q$ and a concatenated $Z$.
- Count the period from complete cycles only.

## Resources

- 732A80 Time Series and Sequence Learning, Linköping University (course lab 2, autumn 2020).
- Durbin & Koopman, *Time Series Analysis by State Space Methods*, 2nd ed., Oxford University Press, 2012, Chapter 3.
- Kitagawa, *Introduction to Time Series Modeling*, CRC Press, 2010, Chapters 11–12 (trend and seasonal models).
- NASA Global Mean Sea Level data. https://climate.nasa.gov/vital-signs/sea-level/
- Next steps: [the Kalman filter](/notes/kalman-filter/) and [the Kalman smoother](/notes/kalman-smoother/).
