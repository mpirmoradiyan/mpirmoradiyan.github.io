---
title: "Bootstrap and jackknife: standard errors and four kinds of confidence interval"
slug: bootstrap-and-jackknife-confidence-intervals
date: 2020-02-23
topic: Statistical Inference
tags: [university-lab, bootstrap, jackknife, confidence-intervals, resampling, standard-error]
summary: "How resampling the data gives a standard error and bias estimate for almost any statistic, how normal, basic, percentile and BCa bootstrap intervals differ, and how the jackknife relates to them."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "An Introduction to the Bootstrap"
    author: "Bradley Efron, Robert J. Tibshirani"
    type: book
  - title: "Bootstrap Methods and their Application"
    author: "A. C. Davison, D. V. Hinkley"
    type: book
  - title: "R package boot: boot.ci()"
    author: "Angelo Canty, Brian Ripley"
    url: "https://cran.r-project.org/package=boot"
    type: docs


status: draft
---

How uncertain is a sample mean, a median, a ratio, or a fitted coefficient? For the mean there's a formula, but for most statistics there isn't. The bootstrap answers it by brute force: pretend the sample is the population, resample from it many times, and watch how much the statistic moves. In the fifth computational statistics lab we did this for the mean price of 110 houses, compared four bootstrap intervals, and checked the result against the jackknife.

## The bootstrap in one paragraph

We have data $x_1, \dots, x_n$ and a statistic $\hat\theta = s(x_1, \dots, x_n)$. Draw $n$ values **with replacement** from the data, compute the statistic on that resample, and repeat $B$ times to get $\hat\theta^*_1, \dots, \hat\theta^*_B$. The spread of these values approximates the sampling distribution of $\hat\theta$. From them:

$$
\widehat{\text{se}} = \text{sd}(\hat\theta^*_1, \dots, \hat\theta^*_B), \qquad \widehat{\text{bias}} = \bar\theta^* - \hat\theta, \qquad \hat\theta_{\text{bc}} = \hat\theta - \widehat{\text{bias}} = 2\hat\theta - \bar\theta^*,
$$

where $\bar\theta^*$ is the average of the bootstrap values and $\hat\theta_{\text{bc}}$ is the bias-corrected estimate. For the mean the bias is zero in theory, so $\bar\theta^*$ just wobbles around $\hat\theta$. In the lab, the mean price was about 1080.47, the average of 3000 bootstrap means 1080.48, and their standard deviation about 35.6.

## Four intervals

Let $\hat\theta^*_{(\alpha)}$ be the $\alpha$-quantile of the bootstrap values, and $z_\alpha$ the standard normal quantile.

- **Normal (first-order):** $\hat\theta - \widehat{\text{bias}} \pm z_{0.975}\,\widehat{\text{se}}$. Assumes the sampling distribution is roughly normal. Symmetric by construction.
- **Percentile:** $\big[\hat\theta^*_{(0.025)},\; \hat\theta^*_{(0.975)}\big]$. Just read off the middle 95% of the bootstrap values. Respects skewness, and it's unchanged if you transform the parameter (for example take logs).
- **Basic (reverse percentile):** $\big[2\hat\theta - \hat\theta^*_{(0.975)},\; 2\hat\theta - \hat\theta^*_{(0.025)}\big]$. It uses the bootstrap to estimate the distribution of the *error* $\hat\theta - \theta$ and flips it around $\hat\theta$. So if the bootstrap distribution has a long right tail, the basic interval gets a long *left* tail.
- **BCa (bias-corrected and accelerated):** still percentiles, but at adjusted levels $\alpha_1, \alpha_2$ instead of 0.025 and 0.975:
$$
\alpha_1 = \Phi\!\left(\hat z_0 + \frac{\hat z_0 + z_{0.025}}{1 - \hat a(\hat z_0 + z_{0.025})}\right),
$$
and the same with $z_{0.975}$ for $\alpha_2$. Here $\Phi$ is the standard normal CDF, $\hat z_0 = \Phi^{-1}(\text{share of } \hat\theta^* < \hat\theta)$ corrects for median bias, and the acceleration $\hat a$ (usually estimated from jackknife values) corrects for the standard error changing with $\theta$. It's the most accurate of the four in theory, and it needs a larger $B$ (a few thousand) because it uses extreme quantiles.

For a mean of 110 reasonably behaved prices all four came out close. In one version of our report, percentile gave (1015, 1152), basic (1009, 1146) and normal (1011, 1150). In the other version, with $B = 2000$, BCa gave (1017, 1157), shifted a little right, as you'd expect for right-skewed prices.

Two corrections to our report. First, one section was titled "BCa" but the code called `boot.ci(..., type = "basic")`. Basic and BCa are different intervals; for BCa it's `type = "bca"`. Second, a draft comparison table in our lab folder worked out a "mean" and "variance" for each interval by taking its midpoint and assuming the width was $\pm 2$ standard deviations. That's only meaningful for the normal interval. Percentile and BCa intervals don't have to be symmetric around the estimate, and that asymmetry is exactly the information they add.

## The jackknife

The jackknife leaves out one observation at a time. With $\hat\theta_{(i)}$ the statistic computed without $x_i$, and $\bar\theta_{(\cdot)}$ their average:

$$
\widehat{\text{se}}_{\text{jack}} = \sqrt{\frac{n-1}{n}\sum_{i=1}^n \big(\hat\theta_{(i)} - \bar\theta_{(\cdot)}\big)^2}.
$$

The factor $(n-1)$ is there because the leave-one-out values are much closer to each other than independent resamples would be.

For the mean, a bit of algebra shows that $\widehat{\text{se}}^2_{\text{jack}} = s^2/n$ exactly, where $s^2$ is the usual sample variance with $n-1$. The ideal bootstrap (infinite $B$) gives $\frac{n-1}{n}\cdot\frac{s^2}{n}$, slightly smaller, because resampling uses the plug-in variance with $n$. In the lab the jackknife variance of the mean was 1320.91 and the bootstrap variances were 1268.5 and 1284.3 in the two runs. The jackknife being larger is expected; most of the gap here is Monte Carlo noise from finite $B$, since the $(n-1)/n$ factor alone only accounts for about 1%.

The jackknife is cheap ($n$ evaluations) and deterministic, but it only works well for smooth statistics. For the median it's inconsistent, because removing one point barely moves the median, or moves it in jumps.

```r
library(boot)
set.seed(7)
price <- rgamma(110, shape = 4, scale = 270)     # right-skewed, like house prices
b <- boot(price, statistic = function(d, i) mean(d[i]), R = 3000)
c(estimate = b$t0, bias = mean(b$t) - b$t0, se = sd(b$t))
boot.ci(b, type = c("norm", "basic", "perc", "bca"))

n <- length(price)
theta_i <- sapply(1:n, function(i) mean(price[-i]))           # leave-one-out means
se_jack <- sqrt((n - 1) / n * sum((theta_i - mean(theta_i))^2))
c(se_boot = sd(b$t), se_jack = se_jack, formula = sd(price) / sqrt(n))  # last two equal
```

For bootstrap bands around a whole fitted curve, see [bootstrap bands](/notes/bootstrap-confidence-and-prediction-bands/). For testing a hypothesis by resampling, see [permutation tests](/notes/permutation-tests/).

## Key takeaways

- The bootstrap approximates the sampling distribution by resampling the data with replacement. It gives a standard error and a bias estimate for almost any statistic.
- Normal intervals are symmetric; percentile intervals follow the bootstrap distribution; basic intervals flip it; BCa adjusts the percentile levels for bias and skewness.
- Check which `type` you actually pass to `boot.ci()`.
- The jackknife (leave-one-out) gives exactly $s^2/n$ for the mean, slightly above the bootstrap's value. It fails for non-smooth statistics like the median.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 5 (2020).
- Bradley Efron and Robert J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993, Chapters 6 (standard errors), 10-11 (bias, jackknife), 12-14 (confidence intervals, BCa).
- A. C. Davison and D. V. Hinkley, *Bootstrap Methods and their Application*, Cambridge University Press, 1997, Chapter 5.
- Bradley Efron, "Better bootstrap confidence intervals", *Journal of the American Statistical Association* 82(397), 1987.
- R package `boot` (Canty and Ripley): https://cran.r-project.org/package=boot
