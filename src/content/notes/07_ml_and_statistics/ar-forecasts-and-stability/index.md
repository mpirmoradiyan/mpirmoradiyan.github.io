---
title: "Long-range AR forecasts: why they fade to the mean, and when an AR model is stable"
slug: ar-forecasts-and-stability
date: 2020-09-13
topic: ML & Statistics
tags: [university-lab, time-series, autoregressive-model, forecasting, stationarity, eigenvalues]
summary: "How to forecast many steps ahead with an AR(p) model by feeding predictions back in, why the forecast decays to the mean, and how stability really depends on the roots (or companion-matrix eigenvalues), not on the size of each coefficient."
sources:
  - title: "732A80 Time Series and Sequence Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Time Series Analysis and Its Applications (4th ed.), Sections 3.1 and 3.4"
    author: "Robert H. Shumway, David S. Stoffer"
    url: "https://www.stat.pitt.edu/stoffer/tsa4/"
    type: book
  - title: "Forecasting: Principles and Practice (3rd ed.), Section 9.3"
    author: "Rob J. Hyndman, George Athanasopoulos"
    url: "https://otexts.com/fpp3/"
    type: book


status: draft
---

One-step-ahead prediction with an AR model is easy because you always have the real past values. Forecasting a year ahead is different: the values you need haven't happened yet, so you feed your own predictions back in. In my course lab, when I ran this recursion for a long time on the detrended sea-level data, the forecast flattened out to zero. That's expected, but my explanation of *why* was wrong, so this note is about getting it right.

## Forecasting by recursion

For an AR($p$) model $y_t = a_1 y_{t-1} + \dots + a_p y_{t-p} + \varepsilon_t$, the best forecast (in mean squared error) of a future value given data up to time $n$ is the conditional mean

$$
\hat y_{n+h|n} = \mathbb{E}[y_{n+h} \mid y_{1:n}].
$$

Because the future noise has mean zero, this is computed by running the model forward with the noise set to zero, and using forecasts wherever real values are missing:

$$
\hat y_{n+h|n} = a_1 \hat y_{n+h-1|n} + \dots + a_p \hat y_{n+h-p|n}, \qquad \hat y_{s|n} = y_s \text{ for } s \le n.
$$

So it's a deterministic recursion, not a random simulation. (If you add random draws of $\varepsilon_t$, you get one possible future path instead of the expected one.)

```python
import numpy as np

def forecast_ar(y, theta, m):
    """Conditional-mean forecast m steps past the end of y."""
    p = len(theta)
    hist = list(y[-p:][::-1])            # most recent value first
    out = []
    for _ in range(m):
        nxt = float(np.dot(theta, hist))
        out.append(nxt)
        hist = [nxt] + hist[:-1]         # shift: newest in, oldest out
    return np.array(out)
```

## The companion form makes it clear

Put the last $p$ values in a state vector $\mathbf{x}_t = (y_t, y_{t-1}, \dots, y_{t-p+1})^\top$. Then the noise-free recursion is a matrix multiplication,

$$
\mathbf{x}_{t+1} = A\,\mathbf{x}_t, \qquad A = \begin{bmatrix} a_1 & a_2 & \cdots & a_{p-1} & a_p \\ 1 & 0 & \cdots & 0 & 0 \\ 0 & 1 & \cdots & 0 & 0 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & 0 \end{bmatrix}.
$$

$A$ is called the *companion matrix*. Its first row computes the new value, and the rows below just shift the old values down. After $h$ steps the forecast state is $A^h \mathbf{x}_n$.

Now everything depends on the eigenvalues $\lambda_1, \dots, \lambda_p$ of $A$. Roughly, $A^h$ behaves like $\lambda^h$ for each eigenvalue. If every $|\lambda_i| < 1$, then $A^h \to 0$ and the forecast decays to zero, which is the mean of a zero-mean AR process. The largest $|\lambda_i|$ sets how fast: close to 1 means slow decay, and complex eigenvalues give damped oscillations on the way down. If any $|\lambda_i| > 1$, the forecast blows up.

The same condition is often written with the characteristic polynomial: the model is stable (stationary) when all roots $z$ of

$$
1 - a_1 z - a_2 z^2 - \dots - a_p z^p = 0
$$

lie *outside* the unit circle, $|z| > 1$. The two statements are equivalent: the roots $z$ are exactly $1/\lambda_i$.

## The mistake I made

In my lab report I wrote that the forecast goes to zero because all the coefficients $a_i$ are between $-1$ and $1$, which makes the process stable. That's not right. The size of each coefficient on its own tells you very little. What matters is the eigenvalues of $A$ (equivalently, the roots above), and they depend on all the coefficients together.

Two quick counterexamples:

- $a = (0.6, 0.6)$: both coefficients are inside $(-1, 1)$, but one eigenvalue is about $1.13$, so the model is unstable.
- $a = (1.5, -0.7)$: the first coefficient is bigger than 1, but the eigenvalues have modulus about $0.84$, so the model is stable (and its forecasts oscillate while they decay).

```python
def companion(theta):
    p = len(theta)
    A = np.zeros((p, p))
    A[0, :] = theta                # new value = weighted sum of old ones
    A[1:, :-1] = np.eye(p - 1)     # shift the old values down
    return A

def is_stable(theta):
    return np.max(np.abs(np.linalg.eigvals(companion(theta)))) < 1

print(is_stable([0.6, 0.3]))    # True
print(is_stable([0.6, 0.6]))    # False, though both are in (-1, 1)
print(is_stable([1.5, -0.7]))   # True, though 1.5 > 1
```

A useful necessary condition to remember: for a stable AR model, $a_1 + \dots + a_p < 1$. If that sum is 1 or more, the model has a unit root or worse. But it's not sufficient on its own.

## So should the forecast always go to zero?

For a stable AR model, yes: it goes to the process mean. That's zero here only because I detrended the data and fitted without an intercept. With an intercept $c$, the forecast converges to $c / (1 - a_1 - \dots - a_p)$. In the original units, it goes back to the fitted trend line. This is a real limitation: a stationary model "forgets" the starting point and can only tell you the long-run average far ahead. Any persistent seasonal pattern or trend has to be modelled explicitly, for example with a [structural time series model](/notes/structural-time-series-models/).

The prediction *uncertainty* goes the other way. It grows with $h$ and levels off at the variance of the process itself, which is what you'd guess if you knew nothing about the recent past.

## Key takeaways

- Multi-step AR forecasts are the noise-free recursion, using forecasts in place of unknown values.
- Write the model in companion form; forecasts are $A^h \mathbf{x}_n$.
- Stability means all eigenvalues of $A$ have $|\lambda| < 1$ (all characteristic roots outside the unit circle). Individual coefficients in $(-1, 1)$ don't guarantee it.
- Stable AR forecasts decay to the process mean; the slowest eigenvalue sets the speed.

## Resources

- 732A80 Time Series and Sequence Learning, Linköping University (course lab 1, autumn 2020).
- Shumway & Stoffer, *Time Series Analysis and Its Applications*, 4th ed., Sections 3.1 (causality and roots) and 3.4 (forecasting). https://www.stat.pitt.edu/stoffer/tsa4/
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., Section 9.3. https://otexts.com/fpp3/
