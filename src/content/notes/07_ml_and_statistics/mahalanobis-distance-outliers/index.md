---
title: "Finding multivariate outliers: Euclidean, standardized and Mahalanobis distance"
slug: mahalanobis-distance-outliers
date: 2019-11-28
topic: ML & Statistics
tags: [university-lab, mahalanobis-distance, outlier-detection, multivariate-statistics, chi-square, multiple-testing]
summary: "Why plain Euclidean distance is a poor way to spot unusual observations in correlated data, how Mahalanobis distance fixes it, and how to turn it into an outlier test with the right degrees of freedom and multiple-testing correction."
sources:
  - title: "732A97 Multivariate Statistical Methods (course assignment)"
    author: "Linköping University"
    type: course
  - title: "Applied Multivariate Statistical Analysis (6th ed.), Sections 1.5, 4.6 and 4.7"
    author: "Richard A. Johnson, Dean W. Wichern"
    type: book


status: draft
---

An observation can be unusual in two ways: it has an extreme value in some variable, or it has an odd *combination* of values that are each normal on their own. A tall person who weighs 50 kg isn't extreme in height or in weight, but the pair is rare. Euclidean distance from the mean only catches the first kind. Mahalanobis distance catches both. We used these distances in two group assignments on national athletics records (seven running events, 55 countries), and the mistakes we made along the way are worth writing down.

## Three distances from the mean

Let $x$ be one observation (a vector of $p$ variables), $\bar x$ the sample mean vector and $S$ the sample covariance matrix.

**Euclidean:**
$$
d_E^2 = (x - \bar x)^\top (x - \bar x) = \sum_{j=1}^{p} (x_j - \bar x_j)^2 .
$$
Every variable counts in its own units. With running records, the marathon is measured in minutes with a spread of tens of minutes, the 100 m in seconds with a spread of tenths. So the marathon decides almost everything.

**Standardized** (divide each variable by its variance):
$$
d_V^2 = (x - \bar x)^\top V^{-1} (x - \bar x), \qquad V = \operatorname{diag}(s_1^2, \dots, s_p^2).
$$
Now each variable contributes on the same scale, so this version treats the variables equally. In our report we wrote the opposite ("this method does not treat the variables equally"). That was backwards. What $d_V$ ignores is the *correlation* between variables.

**Mahalanobis:**
$$
d_M^2 = (x - \bar x)^\top S^{-1} (x - \bar x).
$$
This also uses the covariances. Picture the data cloud as an ellipse. $d_M$ measures distance in units of the ellipse: one unit along the long axis is a big step in raw terms, one unit along the short axis is a small one. So a point that's a little off the main diagonal of strongly correlated data can be very far in Mahalanobis terms. That's how a country with an unusual *profile* (fast at short races but relatively slow at long ones, say) shows up as an outlier even though none of its individual records is extreme.

## A bug worth remembering: keep the signs

In the first assignment we centered the data with `abs(x - mean(x))`. For $d_E$ and $d_V$ this is harmless, because only squares appear. For $d_M$ it's wrong. The cross terms $(x_j - \bar x_j)(x_l - \bar x_l)\,[S^{-1}]_{jl}$ depend on whether the two deviations have the same sign, and that's exactly the information that flags an unusual combination. Taking absolute values throws it away. In the second assignment we fixed it and a new country appeared among the outliers. Its pattern across events was unusual, which is what Mahalanobis distance is meant to detect.

```r
library(MASS)
set.seed(7)
S <- matrix(c(1, 0.9, 0.9, 1), 2)                 # two strongly correlated variables
X <- rbind(mvrnorm(50, c(0, 0), S),
           c(1.5, -1.5))                          # last row: odd combination, not odd values
Xc <- sweep(X, 2, colMeans(X))                    # centre, keeping the signs
C  <- cov(X)
n <- nrow(X); p <- ncol(X)

d_euc <- rowSums(Xc^2)
d_std <- rowSums(scale(X)^2)
d_mah <- mahalanobis(X, colMeans(X), C)           # squared Mahalanobis distance
d_abs <- rowSums((abs(Xc) %*% solve(C)) * abs(Xc))  # the abs() mistake

rank_last <- function(d) rank(-d)[n]              # where does the odd row rank (1 = most extreme)?
c(euclid = rank_last(d_euc), standardized = rank_last(d_std),
  mahalanobis = rank_last(d_mah), with_abs = rank_last(d_abs))
# about 8, 8, 1, 7: only the correct Mahalanobis distance puts it first
pchisq(d_mah[n], df = p, lower.tail = FALSE) < 0.001 / n   # TRUE: still an outlier after Bonferroni
```

## Turning distance into a test

If the data are multivariate normal, $d_M^2$ for an observation follows (approximately, for large $n$) a chi-square distribution with **$p$ degrees of freedom**, where $p$ is the number of variables. So a p-value is `pchisq(d2, df = p, lower.tail = FALSE)`. For small samples there's an exact result based on the Beta distribution, but the chi-square version is the usual first check.

Two corrections to what we did in the second assignment:

- **Degrees of freedom.** We used $p - 1 = 6$. It should be $p = 7$. Each of the seven variables contributes one squared standardized deviation. Nothing is lost here the way one degree of freedom is lost when estimating a variance. Using 6 makes the distances look more extreme than they are.
- **What to correct for.** We applied Bonferroni by dividing $\alpha$ by 7, the number of races. But the tests are run once per **country**, not per race: each country's $d_M^2$ is one test that already combines all seven races. With 55 countries, the Bonferroni level is $\alpha/55$. (For many tests, [Benjamini–Hochberg](/notes/benjamini-hochberg-fdr/) is a less strict alternative.) (We also called the 0.001 a "confidence interval"; it's a significance level.)

And one interpretation point: being flagged doesn't mean an observation is an error. In athletics data, the countries that stand out are often small nations with few athletes, which is a real feature of the data, not something to delete.

## Key takeaways

- Euclidean distance is dominated by the variables with the largest spread; standardizing fixes the scale but ignores correlation.
- Mahalanobis distance $(x-\bar x)^\top S^{-1}(x-\bar x)$ uses the covariance, so it catches unusual combinations.
- Never take absolute values of the deviations; the signs carry the correlation information.
- Under normality $d_M^2 \approx \chi^2_p$, with $p$ the number of variables; correct for the number of observations tested.

## Resources

- 732A97 Multivariate Statistical Methods, Linköping University (group assignments, autumn 2019)
- Richard A. Johnson and Dean W. Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., Pearson (2007): Section 1.5 (statistical distance), Section 4.6 (chi-square plots for assessing normality), Section 4.7 (detecting outliers)
- P. C. Mahalanobis, "On the generalised distance in statistics", *Proceedings of the National Institute of Sciences of India* 2(1) (1936)
- R documentation for `mahalanobis`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mahalanobis.html
