---
title: "Hotelling's T²: testing a mean vector, confidence ellipses and simultaneous intervals"
slug: hotelling-t2-simultaneous-intervals
date: 2019-11-28
topic: ML & Statistics
tags: [university-lab, hotelling-t2, confidence-region, bonferroni, multivariate-statistics, hypothesis-testing]
summary: "How Hotelling's T² tests several means at once, why its confidence region is an ellipse, and how T² and Bonferroni simultaneous intervals compare with ordinary one-at-a-time intervals."
sources:
  - title: "732A97 Multivariate Statistical Methods (course assignment)"
    author: "Linköping University"
    type: course
  - title: "Applied Multivariate Statistical Analysis (6th ed.), Chapter 5"
    author: "Richard A. Johnson, Dean W. Wichern"
    type: book


status: draft
---

Say you measure two things on each bird, tail length and wing length, and want to know if the population means match some reference pair $\mu_0$. Two separate $t$-tests feel natural, but they ignore that the two lengths are correlated, and they give you two error rates instead of one. Hotelling's $T^2$ tests both means in one go, and its confidence region turns out to be an ellipse. In a group assignment we drew that ellipse together with T² and Bonferroni intervals, and seeing the ellipse next to the box of intervals is what made it make sense for me.

## The test statistic

For $n$ observations of a $p$-dimensional vector with sample mean $\bar x$ and sample covariance $S$, to test $H_0: \mu = \mu_0$:

$$
T^2 = n\,(\bar x - \mu_0)^\top S^{-1} (\bar x - \mu_0).
$$

It's the squared [Mahalanobis distance](/notes/mahalanobis-distance-outliers/) between $\bar x$ and $\mu_0$, scaled by $n$ (because the covariance of $\bar x$ is $\Sigma / n$). With $p = 1$ it reduces to the square of the usual $t$ statistic.

If the data are multivariate normal, then under $H_0$

$$
T^2 \sim \frac{(n-1)\,p}{n-p}\, F_{p,\,n-p},
$$

so we reject at level $\alpha$ when $T^2 > \frac{(n-1)p}{n-p} F_{p,n-p}(\alpha)$, where $F_{p,n-p}(\alpha)$ is the upper $\alpha$ quantile.

## The confidence region is an ellipse

Turning the test around, the $100(1-\alpha)\%$ confidence region for $\mu$ is every $\mu$ that wouldn't be rejected:

$$
n\,(\bar x - \mu)^\top S^{-1} (\bar x - \mu) \le \frac{(n-1)p}{n-p} F_{p,n-p}(\alpha).
$$

That's an ellipse (an ellipsoid for $p > 2$) centered at $\bar x$. Its axes point along the eigenvectors $e_i$ of $S$, and the half-length along axis $i$ is

$$
\sqrt{\lambda_i}\;\sqrt{\frac{(n-1)p}{n(n-p)} F_{p,n-p}(\alpha)},
$$

with $\lambda_i$ the matching eigenvalue. That's how we drew it: a circle scaled by those half-lengths and rotated by the eigenvector matrix. If the variables are positively correlated, the ellipse tilts along the diagonal.

## Intervals for each mean

The ellipse is the honest joint answer, but people want an interval per variable. Three options, for component $j$ with standard error $\sqrt{s_{jj}/n}$:

- **One at a time**: $\bar x_j \pm t_{n-1}(\alpha/2)\sqrt{s_{jj}/n}$. Each has 95% coverage alone, but all $p$ together cover less.
- **Bonferroni**: $\bar x_j \pm t_{n-1}\!\big(\alpha/(2p)\big)\sqrt{s_{jj}/n}$. Joint coverage at least $1-\alpha$ for these $p$ intervals.
- **T² simultaneous**: $\bar x_j \pm \sqrt{\frac{(n-1)p}{n-p}F_{p,n-p}(\alpha)}\sqrt{s_{jj}/n}$. This is the shadow of the ellipse on each axis. It holds jointly for *every* linear combination $a^\top \mu$, not just the $p$ components.

Because T² intervals cover infinitely many combinations, they're the widest. Bonferroni intervals are shorter when you only care about the $p$ means, which is what we found too. The T² ones pay off when you go "data snooping": look at the data, then decide to examine some difference or combination, and still keep the coverage promise.

## The ellipse sees things the box doesn't

The ellipse and the box of intervals can disagree. With correlated variables, a $\mu_0$ can sit inside every single interval yet outside the ellipse, because the *combination* of deviations is unlikely. The example below is built that way:

```r
library(MASS)
set.seed(8)
X <- mvrnorm(45, mu = c(193, 280), Sigma = matrix(c(120, 120, 120, 210), 2))
n <- nrow(X); p <- ncol(X); alpha <- 0.05
xbar <- colMeans(X); S <- cov(X)
mu0 <- c(196, 277)

T2   <- n * t(xbar - mu0) %*% solve(S) %*% (xbar - mu0)
crit <- (n - 1) * p / (n - p) * qf(1 - alpha, p, n - p)
c(T2 = T2, critical = crit)                         # about 8.50 vs 6.58: reject H0

se <- sqrt(diag(S) / n)
half <- rbind(one_at_a_time = qt(1 - alpha / 2, n - 1) * se,
              bonferroni    = qt(1 - alpha / (2 * p), n - 1) * se,
              T2            = sqrt(crit) * se)
round(half, 2)                                      # widths: one-at-a-time < Bonferroni < T2
abs(xbar - mu0) < half["bonferroni", ]              # TRUE TRUE: each component looks fine alone
```

Here $\bar x \approx (194.0, 279.2)$, so $\mu_0$ is slightly high in the first variable and slightly low in the second. Each difference is small on its own, but the two variables are positively correlated, so "one up, one down" is unlikely. T² rejects. The intervals alone would have missed it.

## Checking the normality assumption

The $F$ distribution above assumes multivariate normality. In the assignment we checked a Q–Q plot for each variable and concluded the population "can be considered normal". That's a weaker check than it sounds: normal marginals don't guarantee joint normality. A better quick check is a chi-square plot: compute each observation's squared Mahalanobis distance $d_i^2$ to $\bar x$, sort them, and plot against $\chi^2_p$ quantiles. A straight line supports multivariate normality. (With large $n$, the T² test is also fairly robust because $\bar x$ is close to normal anyway.)

## Key takeaways

- $T^2 = n(\bar x-\mu_0)^\top S^{-1}(\bar x-\mu_0)$ tests all $p$ means at once and accounts for correlation.
- Under normality $T^2 \sim \frac{(n-1)p}{n-p}F_{p,n-p}$; the confidence region is an ellipse along the eigenvectors of $S$.
- Bonferroni intervals are shorter for the $p$ components; T² intervals are wider but cover every linear combination.
- A point can be inside every interval and still outside the ellipse.
- Check joint normality with a chi-square plot, not only per-variable Q–Q plots.

## Resources

- 732A97 Multivariate Statistical Methods, Linköping University (group assignment, autumn 2019)
- Richard A. Johnson and Dean W. Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., Pearson (2007): Section 5.2 (Hotelling's $T^2$), Section 5.4 (confidence regions and simultaneous comparisons), Section 4.6 (chi-square plots)
- Harold Hotelling, "The generalization of Student's ratio", *Annals of Mathematical Statistics* 2(3) (1931): https://doi.org/10.1214/aoms/1177732979
