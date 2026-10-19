---
title: "Bootstrap and jackknife: standard errors, confidence intervals and bands around a curve"
slug: bootstrap-and-jackknife-confidence-intervals
date: 2020-02-23
topic: Statistical Inference
tags: [university-lab, bootstrap, jackknife, confidence-intervals, prediction-intervals, resampling, standard-error, uncertainty, regression]
summary: "How resampling the data gives a standard error, a bias estimate and confidence intervals for almost any statistic, how normal, basic, percentile and BCa intervals differ, how the jackknife compares, and how the same trick puts confidence and prediction bands around a fitted curve."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "An Introduction to the Bootstrap"
    author: "Bradley Efron, Robert J. Tibshirani"
    type: book
  - title: "Bootstrap Methods and their Application"
    author: "A. C. Davison, D. V. Hinkley"
    type: book
  - title: "The Elements of Statistical Learning, Section 8.2 (The Bootstrap and Maximum Likelihood)"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book
  - title: "R package boot: boot(), boot.ci(), envelope()"
    author: "Angelo Canty, Brian Ripley"
    url: "https://cran.r-project.org/package=boot"
    type: docs


status: draft
---

How sure am I about a sample mean? There's a formula for that. But what about a median, a ratio, or the curve a regression tree draws? Usually there's no formula at all. I ran into this twice in my first year at LiU: once with the mean price of 110 houses, once with a regression tree on 48 points. Both times the answer was the bootstrap. The idea is almost cheeky: pretend your sample *is* the population, resample from it many times, and watch how much your answer moves.

## The bootstrap in one paragraph

Say I have data $x_1, \dots, x_n$ and some statistic $\hat\theta = s(x_1, \dots, x_n)$. I draw $n$ values **with replacement** from the data, compute the statistic on that new sample, and do it again, $B$ times. Now I have $\hat\theta^*_1, \dots, \hat\theta^*_B$. How spread out they are tells me how much $\hat\theta$ would wobble if I could collect new data. From them:

$$
\widehat{\text{se}} = \text{sd}(\hat\theta^*_1, \dots, \hat\theta^*_B), \qquad \widehat{\text{bias}} = \bar\theta^* - \hat\theta, \qquad \hat\theta_{\text{bc}} = \hat\theta - \widehat{\text{bias}} = 2\hat\theta - \bar\theta^*,
$$

where $\bar\theta^*$ is the average of the bootstrap values and $\hat\theta_{\text{bc}}$ is the bias-corrected estimate. For the mean the bias is zero in theory, so $\bar\theta^*$ just wobbles around $\hat\theta$. In the lab the mean price was about 1080.47, the average of 3000 bootstrap means 1080.48, and their standard deviation about 35.6.

## Four ways to turn it into an interval

Next question: how do I get a 95% interval out of those $B$ numbers? It turns out there are several answers. Let $\hat\theta^*_{(\alpha)}$ be the $\alpha$-quantile of the bootstrap values, and $z_\alpha$ the standard normal quantile.

- **Normal (first-order):** $\hat\theta - \widehat{\text{bias}} \pm z_{0.975}\,\widehat{\text{se}}$. Assumes the sampling distribution is roughly normal. Symmetric by construction.
- **Percentile:** $\big[\hat\theta^*_{(0.025)},\; \hat\theta^*_{(0.975)}\big]$. Just read off the middle 95% of the bootstrap values. It follows skewness, and it doesn't change if you transform the parameter (for example take logs).
- **Basic (reverse percentile):** $\big[2\hat\theta - \hat\theta^*_{(0.975)},\; 2\hat\theta - \hat\theta^*_{(0.025)}\big]$. This one confused me at first. It uses the bootstrap to estimate the distribution of the *error* $\hat\theta - \theta$ and flips it around $\hat\theta$. So if the bootstrap values have a long right tail, the basic interval gets a long *left* tail.
- **BCa (bias-corrected and accelerated):** still percentiles, but at adjusted levels $\alpha_1, \alpha_2$ instead of 0.025 and 0.975:
$$
\alpha_1 = \Phi\!\left(\hat z_0 + \frac{\hat z_0 + z_{0.025}}{1 - \hat a(\hat z_0 + z_{0.025})}\right),
$$
and the same with $z_{0.975}$ for $\alpha_2$. Here $\Phi$ is the standard normal CDF, $\hat z_0 = \Phi^{-1}(\text{share of } \hat\theta^* < \hat\theta)$ corrects for median bias, and the acceleration $\hat a$ (usually estimated from jackknife values) corrects for the standard error changing with $\theta$. In theory it's the most accurate of the four, and it needs a larger $B$ (a few thousand) because it reads quantiles far out in the tails.

For a mean of 110 fairly well-behaved prices, all four came out close. In one version of our report, percentile gave (1015, 1152), basic (1009, 1146) and normal (1011, 1150). In the other version, with $B = 2000$, BCa gave (1017, 1157), shifted a little right, as you'd expect for right-skewed prices.

Looking back, our report had two mistakes. First, one section was titled "BCa" but the code called `boot.ci(..., type = "basic")`. Those are different intervals; for BCa it's `type = "bca"`. Second, a draft table worked out a "mean" and "variance" for each interval by taking its midpoint and assuming the width was $\pm 2$ standard deviations. That only makes sense for the normal interval. Percentile and BCa intervals don't have to be symmetric around the estimate, and that asymmetry is exactly the extra information they give.

## The jackknife

The jackknife is the older, simpler cousin. Instead of random resamples, leave out one observation at a time. With $\hat\theta_{(i)}$ the statistic computed without $x_i$, and $\bar\theta_{(\cdot)}$ their average:

$$
\widehat{\text{se}}_{\text{jack}} = \sqrt{\frac{n-1}{n}\sum_{i=1}^n \big(\hat\theta_{(i)} - \bar\theta_{(\cdot)}\big)^2}.
$$

Why the big factor $(n-1)$? Because the leave-one-out values are much closer to each other than independent resamples would be, so their spread has to be scaled up.

For the mean, a bit of algebra shows $\widehat{\text{se}}^2_{\text{jack}} = s^2/n$ exactly, where $s^2$ is the usual sample variance with $n-1$. The ideal bootstrap (infinite $B$) gives $\frac{n-1}{n}\cdot\frac{s^2}{n}$, slightly smaller, because resampling uses the plug-in variance with $n$. In the lab the jackknife variance of the mean was 1320.91 and the bootstrap variances were 1268.5 and 1284.3 in the two runs. The jackknife being larger is expected. Most of the gap here is Monte Carlo noise from finite $B$, since the $(n-1)/n$ factor alone only explains about 1%.

The jackknife is cheap ($n$ evaluations) and gives the same answer every time, but it only works well for smooth statistics. For the median it's inconsistent: removing one point barely moves the median, or moves it in jumps.

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

## From one number to a whole curve

In the machine learning lab the "statistic" wasn't a number. It was a fitted regression tree, a whole curve. Same trick, though: refit the model on many fake datasets and look at how much the curve moves at each $x$. Here I had to ask myself which question I was actually answering.

- **Confidence band**: where is the *true curve* $f(x) = E[Y \mid X = x]$? Only the uncertainty in the fit counts.
- **Prediction band**: where will a *new observation* $Y_{new}$ at $x$ land? That's the fit uncertainty plus the noise around the curve:

$$
\operatorname{Var}(Y_{new} - \hat f(x)) = \operatorname{Var}(\hat f(x)) + \sigma^2,
$$

where $\sigma^2$ is the noise variance and the new noise is independent of the fit. The first term shrinks with more data; the second doesn't. So a prediction band never collapses onto the curve. A sanity check I used: about 95% of the observed points should fall inside a 95% prediction band (in my lab 2 of 48 fell outside, which is fine). Nothing like that holds for a confidence band.

There are also two ways to make the fake datasets:

- **Non-parametric:** resample the rows $(x_i, y_i)$ with replacement, refit, predict on a grid of $x$ values. At each grid point, the 2.5% and 97.5% quantiles of the $B$ predictions give a pointwise 95% confidence band. No distribution assumed. But some resamples miss whole regions of $x$, so the band can get bumpy, especially for a step-shaped model like a tree.
- **Parametric:** assume a noise model, for example $y_i \sim N(\hat f(x_i), \hat\sigma^2)$, with $\hat f$ the fit to the real data and $\hat\sigma$ the residual standard deviation. Keep the $x_i$ fixed, simulate new $y_i$, refit, predict. The band comes out smoother, but only if the noise model is roughly right. My residuals were skewed, so the normal assumption was doubtful, and a smooth band can look more trustworthy than it is.

For a **prediction band** with the parametric bootstrap, add fresh noise to each bootstrap prediction, $\hat f^{*b}(x) + \varepsilon^{*}$ with $\varepsilon^{*} \sim N(0, \hat\sigma^2)$, then take quantiles as before.

```r
set.seed(1)
d <- data.frame(x = sort(runif(60, 0, 10)))
d$y <- sin(d$x) + rnorm(60, sd = 0.4)
grid <- data.frame(x = seq(0, 10, length.out = 50))
fit_curve <- function(data) predict(lm(y ~ poly(x, 5), data = data), newdata = grid)

np <- boot(d, function(data, i) fit_curve(data[i, ]), R = 1000)   # resample rows

fit0 <- lm(y ~ poly(x, 5), data = d)                             # keep x, simulate y
s <- sigma(fit0)
gen <- function(data, mle) { data$y <- rnorm(nrow(data), fitted(fit0), s); data }
pb_conf <- boot(d, fit_curve, R = 1000, sim = "parametric", ran.gen = gen)
pb_pred <- boot(d, function(data) fit_curve(data) + rnorm(nrow(grid), 0, s),
                R = 1000, sim = "parametric", ran.gen = gen)

band  <- function(b) apply(b$t, 2, quantile, probs = c(0.025, 0.975))
width <- function(b) mean(diff(band(b)))
c(nonparam_conf = width(np), param_conf = width(pb_conf), param_pred = width(pb_pred))
# about 0.69, 0.51 and 1.64: the prediction band is roughly 3 times wider
```

Things that tripped me up with bands:

- **Pointwise vs simultaneous.** Quantiles at each $x$ separately give a *pointwise* band: at any single $x$ the true curve is inside with about 95% probability. The whole curve staying inside everywhere at once is less likely. `boot::envelope()` returns both: `$point` is pointwise and `$overall` is a wider simultaneous band.
- **`boot.ci()` is for one number.** When the statistic returns a vector of predictions, `boot.ci()` only looks at one element (the first, by default). For bands, take quantiles across the columns of `b$t`.
- **Wide bands aren't a bug.** Bumpy, wide bands for a tree on 48 points mostly say "this model is unstable on this little data", which is useful to know.

For testing a hypothesis by resampling instead, see [permutation tests](/notes/permutation-tests/).

## Key takeaways

- The bootstrap approximates the sampling distribution by resampling the data with replacement. It gives a standard error and a bias estimate for almost any statistic.
- Normal intervals are symmetric; percentile intervals follow the bootstrap distribution; basic intervals flip it; BCa adjusts the percentile levels for bias and skewness. Check which `type` you actually pass to `boot.ci()`.
- The jackknife (leave-one-out) gives exactly $s^2/n$ for the mean, slightly above the bootstrap's value. It fails for non-smooth statistics like the median.
- Around a fitted curve, a confidence band is about the curve and a prediction band is about new points (it adds the noise variance). Non-parametric bootstrap resamples rows; parametric bootstrap simulates from a noise model and is only as good as that model.
- Quantiles at each $x$ give pointwise bands; simultaneous bands are wider.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 5 (2020).
- 732A99 Machine Learning, Linköping University, Block 1 Lab 2 (autumn 2019).
- Bradley Efron and Robert J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993, Chapters 6 (standard errors, parametric bootstrap), 9 (regression), 10-11 (bias, jackknife), 12-14 (confidence intervals, BCa).
- A. C. Davison and D. V. Hinkley, *Bootstrap Methods and their Application*, Cambridge University Press, 1997, Chapter 5.
- Bradley Efron, "Better bootstrap confidence intervals", *Journal of the American Statistical Association* 82(397), 1987.
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 8.2: https://hastie.su.domains/ElemStatLearn/
- R package `boot` (Canty and Ripley), documentation for `boot`, `boot.ci`, `envelope`: https://cran.r-project.org/package=boot
