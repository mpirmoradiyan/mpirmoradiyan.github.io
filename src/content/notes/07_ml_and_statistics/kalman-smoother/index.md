---
title: "Kalman smoothing: using later data to clean up earlier estimates"
slug: kalman-smoother
date: 2020-09-25
topic: ML & Statistics
tags: [university-lab, kalman-smoother, state-space-models, time-series, missing-data]
summary: "What the Kalman smoother computes, how its backward pass differs from the filter, how it behaves over a gap of missing data and after the last observation, and a mix-up between filtered and smoothed estimates that I made in my lab."
sources:
  - title: "732A80 Time Series and Sequence Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Time Series Analysis by State Space Methods (2nd ed.), Sections 4.4–4.5"
    author: "James Durbin, Siem Jan Koopman"
    type: book
  - title: "Bayesian Filtering and Smoothing, Chapter 8 (Rauch–Tung–Striebel smoother)"
    author: "Simo Särkkä"
    url: "https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf"
    type: book


status: draft
---

The [Kalman filter](/notes/kalman-filter/) estimates the state at time $t$ using data up to $t$. That's what you want in real time. But when you analyse a finished data set, for example to pull out the trend and the seasonal pattern, you have the data *after* $t$ too, and it carries information about $t$. The smoother uses all of it. In my course lab I got the code for the smoother from the course, but I mixed up its outputs with the filter's, and that mistake taught me what the difference actually looks like.

## Filtering vs smoothing

- **Filter:** $p(\alpha_t \mid y_{1:t})$, with mean $a_{t|t}$ and covariance $P_{t|t}$. Only past and present data.
- **Smoother:** $p(\alpha_t \mid y_{1:n})$ for $t \le n$, with mean $\hat\alpha_{t|n}$ and covariance $V_{t|n}$. All the data.

The smoothed estimate is never more uncertain than the filtered one: $V_{t|n} \preceq P_{t|t}$. More data can't hurt. At the last time point they're equal, since there's nothing after $n$ to add.

## The backward pass

The classic form is the Rauch–Tung–Striebel (RTS) smoother. Run the filter forward and store $a_{t|t}, P_{t|t}$ and the predictions $a_{t+1|t}, P_{t+1|t}$. Then go backward from $t = n-1$ to $1$:

$$
J_t = P_{t|t} T^\top P_{t+1|t}^{-1},
$$
$$
\hat\alpha_{t|n} = a_{t|t} + J_t\,(\hat\alpha_{t+1|n} - a_{t+1|t}),
$$
$$
V_{t|n} = P_{t|t} + J_t\,(V_{t+1|n} - P_{t+1|t})\,J_t^\top,
$$

starting from $\hat\alpha_{n|n} = a_{n|n}$, $V_{n|n} = P_{n|n}$.

How I read it: $\hat\alpha_{t+1|n} - a_{t+1|t}$ is "how much the full data changed my mind about time $t+1$, compared to what I predicted from time $t$". The gain $J_t$ translates that correction back to time $t$. It's large when the filter was uncertain at $t$ and the step from $t$ to $t+1$ was fairly predictable.

(The course code used a different but equivalent form, the disturbance smoother from Durbin & Koopman, which also returns smoothed estimates of the noise terms $\eta_t$ and $\varepsilon_t$. Those are handy for [EM](/notes/em-for-state-space-models/).)

```python
import numpy as np

def rts_smoother(a_filt, P_filt, a_pred, P_pred, T):
    """a_filt[t] = a_{t|t}, P_filt[t] = P_{t|t}; a_pred[t] = a_{t|t-1}, P_pred[t] = P_{t|t-1}."""
    n = len(a_filt)
    a_sm, V_sm = a_filt.copy(), P_filt.copy()
    for t in range(n - 2, -1, -1):
        J = P_filt[t] @ T.T @ np.linalg.inv(P_pred[t + 1])
        a_sm[t] = a_filt[t] + J @ (a_sm[t + 1] - a_pred[t + 1])
        V_sm[t] = P_filt[t] + J @ (V_sm[t + 1] - P_pred[t + 1]) @ J.T
    return a_sm, V_sm
```

## Getting components out

With $s_t = Z\alpha_t$ (for example trend + season), the smoothed signal is

$$
\mathbb{E}[s_t \mid y_{1:n}] = Z\hat\alpha_{t|n}, \qquad \text{Var}(s_t \mid y_{1:n}) = Z V_{t|n} Z^\top.
$$

And to get one component, use only its part of $Z$. For the structural model, $Z_\mu$ applied to the trend states gives $\hat\mu_{t|n}$, and $Z_\gamma$ applied to the seasonal states gives $\hat\gamma_{t|n}$.

## My mix-up

In my lab report I wrote the right formulas for the smoothed signal, but in the code I plugged in the *filtered* means and covariances (the filter's outputs, which the smoother result object also carries) instead of the smoothed ones. So my "smoothed" signal, trend and seasonal plots were really filter plots. And in the missing-data part I plotted the smoother result's one-step predictions, which are also just the filter's, so my "filter" and "smoother" plots were the same picture.

The fix is simple: use $\hat\alpha_{t|n}$ and $V_{t|n}$. But it's worth knowing what the right plots should show, because that's how you'd catch this kind of bug.

## What to expect over a gap of missing data

Say observations 301–400 are missing.

- **Filter:** through the gap it just predicts. The estimate follows the model dynamics (trend continues, season repeats), and the uncertainty band grows steadily until the first observation after the gap, where it snaps back.
- **Smoother:** it knows what happened at both ends. The estimate bridges the gap and joins up with the data on both sides. The band is narrow near both edges and widest in the middle of the gap, like a lens shape.

If your filter and smoother plots look identical over a gap, something is wrong.

## What to expect after the last observation

For $t > n$ there's no data after $t$ (or after $n$) to pull information back. So the smoothed estimate equals the filter's forecast there: $\hat\alpha_{t|n} = a_{t|n}$. In my report I said the smoother "uses what we predicted" to smooth the future part. It's simpler than that: past the last observation the smoother has nothing to add, and the two curves coincide. The smoother only changes things *before* $n$.

## Key takeaways

- Filter: data up to $t$. Smoother: all data. The smoother is at least as certain as the filter.
- The RTS smoother is a backward pass over stored filter results.
- To get a component (trend, season), multiply its part of $Z$ with the smoothed state.
- Over a gap, the smoother bridges it with a lens-shaped band; after the last observation, smoother and filter agree.
- If filter and smoother plots look the same, check which arrays you're plotting.

## Resources

- 732A80 Time Series and Sequence Learning, Linköping University (course lab 2, autumn 2020).
- Durbin & Koopman, *Time Series Analysis by State Space Methods*, 2nd ed., Oxford University Press, 2012, Sections 4.4–4.5 (state and disturbance smoothing).
- Särkkä, *Bayesian Filtering and Smoothing*, Cambridge University Press, 2013, Chapter 8. https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
