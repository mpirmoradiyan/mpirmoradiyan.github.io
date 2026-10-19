---
title: "Kernel regression with several distances: summing vs multiplying kernels"
slug: kernel-regression-sum-vs-product
date: 2020-06-08
topic: ML & Statistics
tags: [university-lab, kernel-regression, nadaraya-watson, bandwidth, smoothing, spark]
summary: "How Nadaraya-Watson kernel regression predicts a value as a weighted average of nearby observations, and why combining distance, date and time kernels by product behaves very differently from combining them by sum."
sources:
  - title: "732A54 Big Data Analytics (course lab)"
    author: "Linköping University"
    type: course
  - title: "The Elements of Statistical Learning, Chapter 6 (Kernel Smoothing Methods)"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book
  - title: "Pattern Recognition and Machine Learning, Section 6.3.1 (Nadaraya-Watson model)"
    author: "Christopher M. Bishop"
    type: book


status: draft
---

Kernel regression is one of the simplest ways to predict something: take a weighted average of the observations you already have, and give more weight to the ones that are "close" to the point you're predicting. The interesting part shows up when "close" has several meanings at once. In my Big Data course (a pair lab) we predicted the temperature at a place, date and hour from Swedish weather station data, using three kernels: one for geographic distance, one for the day of the year, one for the time of day. The big lesson was that *how you combine* the three kernels changes the answer a lot.

## The basic estimator

The Nadaraya–Watson estimator predicts at a point $x$ as

$$
\hat y(x) = \frac{\sum_{i=1}^{n} k(x, x_i)\, y_i}{\sum_{i=1}^{n} k(x, x_i)},
$$

where $x_i$ are the observed inputs (station, date, time), $y_i$ the observed temperatures, $n$ the number of observations, and $k(x, x_i) \ge 0$ a weight that is large when $x_i$ is close to $x$. The denominator makes the weights sum to one, so $\hat y$ is a weighted average and always lies between the smallest and largest $y_i$.

A common choice for one distance $u$ is the Gaussian kernel with bandwidth $h$:

$$
k_h(u) = \exp\!\left(-\frac{u^2}{h^2}\right).
$$

At $u = 0$ the weight is 1; at $u = h$ it has dropped to $e^{-1} \approx 0.37$; by $u = 3h$ it's basically zero. (Some books write $\exp(-u^2 / 2h^2)$. That only rescales what $h$ means.) So $h$ is "how far away something can be and still count". Small $h$: only very close points matter, the prediction is noisy. Large $h$: everything counts, the prediction drifts towards the overall average.

## Three distances

For weather we had:

- $u_{\text{dist}}$: great-circle distance in km between the target location and the station (haversine formula).
- $u_{\text{date}}$: difference in day of the year, wrapped around the year end, so 30 December and 2 January are 3 days apart: $u = \min(|d_1 - d_2|,\ 365 - |d_1 - d_2|)$, with $d$ the day of the year.
- $u_{\text{time}}$: difference in hours, wrapped around midnight the same way: $\min(|t_1 - t_2|,\ 24 - |t_1 - t_2|)$.

Each gets its own bandwidth, and in each case it's in the units of that distance: km, days, hours. In our report I wrote the distance bandwidth as "30 km² to 100 km²". That's wrong: the haversine distance is a length, so the bandwidth is in km, not km².

One more rule: only use observations from *before* the date you predict for. Otherwise you're using the future to predict the past.

## Sum or product?

To get one weight per observation we need to combine three kernel values $k_1, k_2, k_3 \in [0, 1]$:

$$
k_{\text{sum}} = k_1 + k_2 + k_3, \qquad k_{\text{prod}} = k_1 \cdot k_2 \cdot k_3.
$$

The way I think about it:

- **Product = "close in every way".** If any one kernel is near zero, the whole weight is near zero. A reading from the right station at the right hour but in July gets almost nothing when you predict for January.
- **Sum = "close in any way".** One good match is enough to get a sizeable weight. That July reading at the right station and hour still gets weight of about 2 out of a possible 3.

Since almost every observation is close in *some* sense, the sum lets huge numbers of only-partly-relevant readings into the average. The prediction gets pulled towards the overall mean of the data.

That's exactly what we saw. For a winter date (early January) in the revised version of our lab, the summed kernel predicted around 5 to 6 °C across the hours of the day, while the product kernel gave about −1 to −0.5 °C. The product answer is the plausible one for a Swedish January. The summed answer looks like a yearly average, because summer readings got a lot of weight just for being at the right place or the right hour.

There's a second effect: with the product, the daily curve is driven by readings that are close in all three senses, so its shape follows that specific date. With the sum, the time kernel alone adds a daily wiggle on top of a mostly flat, washed-out average.

## A tiny demo

```python
import numpy as np

def k(u, h):
    return np.exp(-(u / h) ** 2)

rng = np.random.default_rng(1)
n = 5000
dist = rng.uniform(0, 600, n)        # km from target
day = rng.integers(0, 183, n)        # days from target date (wrapped)
hour = rng.uniform(0, 12, n)         # hours from target time (wrapped)
# Fake temperatures: cold near the target date, warm half a year away
temp = -2 + 18 * (day / 182) + rng.normal(0, 2, n)

k1, k2, k3 = k(dist, 100), k(day, 15), k(hour, 3)
for name, w in [("sum", k1 + k2 + k3), ("product", k1 * k2 * k3)]:
    print(name, round(np.sum(w * temp) / np.sum(w), 2))
```

With this seed, the product estimate comes out at about −1.1, close to the temperatures within a couple of weeks of the target date. The sum comes out at about 5.9, far above anything near that date, because readings from months away still got a lot of weight from the distance and hour kernels.

## Key takeaways

- Nadaraya–Watson: prediction = kernel-weighted average of observed values.
- The bandwidth $h$ is in the units of its distance and sets how far "close" reaches.
- Wrap cyclic distances (day of year, hour of day) so the year end and midnight aren't treated as far apart.
- Product of kernels = close in every dimension; sum = close in any dimension. The sum pulls predictions towards the global mean.
- Only use observations from before the prediction time.

## Resources

- 732A54 Big Data Analytics, Linköping University, Spark lab 3 (machine learning with Spark), 2020.
- Hastie, Tibshirani, Friedman, [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), Chapter 6, Kernel Smoothing Methods.
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 6.3.1, Nadaraya–Watson model.
- See also [Broadcast the small side](/notes/broadcast-joins-in-spark/) for how to run this on the full dataset in Spark.
