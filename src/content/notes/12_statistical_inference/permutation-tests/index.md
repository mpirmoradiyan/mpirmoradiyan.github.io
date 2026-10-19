---
title: "Permutation tests: p-values by shuffling, and what a p-value can't tell you"
slug: permutation-tests
date: 2020-02-23
topic: Statistical Inference
tags: [university-lab, hypothesis-testing, permutation-test, p-value, statistical-power, bootstrap]
summary: "How a permutation test builds the null distribution of any statistic by shuffling, how to read the resulting p-value correctly, how to estimate power by simulation, and why a bootstrap distribution is not a null distribution."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "Randomization and social affairs: the 1970 draft lottery"
    author: "Stephen E. Fienberg"
    url: "https://doi.org/10.1126/science.171.3968.255"
    type: paper
  - title: "An Introduction to the Bootstrap, Chapters 15-16"
    author: "Bradley Efron, Robert J. Tibshirani"
    type: book
  - title: "Permutation p-values should never be zero"
    author: "Belinda Phipson, Gordon K. Smyth"
    url: "https://doi.org/10.2202/1544-6115.1585"
    type: paper


status: draft
---

A permutation test answers "could this pattern be just chance?" without assuming any distribution. You don't need a formula for the null distribution of your statistic: you make it by shuffling. In the fifth computational statistics lab we used it (and the bootstrap) to ask whether the 1970 US draft lottery was random. Looking back, our report got the logic of the hypothesis test backwards in one place, so this note is partly a correction.

## The idea

Say we have pairs $(x_i, y_i)$, $i = 1, \dots, n$, and a statistic $T$ that measures association, like a slope or a correlation. The null hypothesis $H_0$ is "$y$ has nothing to do with $x$".

If $H_0$ is true, then the particular way the $y$ values are lined up with the $x$ values is just one of the $n!$ equally likely arrangements. So:

1. Compute $t_{\text{obs}} = T(x, y)$ on the real data.
2. Shuffle $y$ (a random permutation, **without** replacement), recompute $T$. Repeat $B$ times to get $t_1^*, \dots, t_B^*$.
3. The two-sided p-value is the share of shuffles at least as extreme as the real data:
$$
p = \frac{1 + \#\{b : |t_b^*| \ge |t_{\text{obs}}|\}}{B + 1}.
$$

The $+1$ in numerator and denominator counts the observed arrangement as one of the possible ones. It keeps $p$ from being exactly 0, which a finite number of shuffles can never justify.

```r
set.seed(6)
n <- 366; x <- 1:n
y <- sample(n) - 0.08 * (x - mean(x))            # ranks plus a weak downward drift
stat <- function(x, y) cor(x, y, method = "spearman")
t_obs <- stat(x, y)
B <- 2000
t_perm <- replicate(B, stat(x, sample(y)))       # shuffle y: a permutation
p <- (1 + sum(abs(t_perm) >= abs(t_obs))) / (B + 1)
c(t_obs = t_obs, p = p)
```

The statistic can be anything you can compute. That's the big selling point.

## What a p-value means (and doesn't)

The p-value is the probability, *assuming $H_0$ is true*, of a statistic at least as extreme as the one observed. A small p-value is evidence against $H_0$. A large p-value means the data are compatible with $H_0$; it doesn't show that $H_0$ is true.

In our group's report we wrote, in one version, "we assume $H_0: T > 0$", got 0.001 as the share of bootstrap values of $T$ above zero, rejected $H_0$, and concluded that there was no trend and "the randomness may be proved". That's wrong in two ways:

1. **The hypotheses were the wrong way round.** "The lottery is random" means no trend, so $H_0$ should be $T = 0$. Only 0.1% of the bootstrap values of $T$ were positive, so almost all resamples gave a negative slope. That's evidence *for* a (downward) trend, i.e. against randomness. Our other version of the report, which said a low p-value means rejecting "the lottery is random", read it correctly.
2. **You can't prove $H_0$.** Even when a test doesn't reject, the most you can say is that the data don't show a detectable departure from it.

## A bootstrap distribution is not a null distribution

The bootstrap resamples the *observed* pairs, so the bootstrap values of $T$ scatter around $t_{\text{obs}}$. They describe the uncertainty in $T$, which is useful for a confidence interval. If a 95% bootstrap interval for $T$ excludes 0, that's evidence of a trend at roughly the 5% level. But the bootstrap values themselves were not generated under $H_0$, so "the share above 0" is not a p-value in the usual sense. A permutation test, in contrast, breaks the link between $x$ and $y$ on purpose, so its distribution *is* the null distribution.

Two coding slips in our permutation code are easy to make, so I'll note them:

- One version drew the shuffled $x$ with `sample(n, replace = TRUE)`. With replacement, that's a bootstrap draw, not a permutation. Use `sample(y)`.
- Another version refitted a smoother on the shuffled data and then computed $T$ on the *fitted values* through the same smoothing function again. That's not wrong as such, but it's simpler and clearer to compute $T$ on the shuffled data directly, exactly as on the real data.

Our permutation p-values (around 0.11 to 0.16 in different versions) didn't reach 5%. Fienberg (1971), using rank correlation, found clear evidence that the 1970 lottery wasn't random: the capsules were reportedly not mixed well, so late-year birthdays tended to get low draft numbers. A test with low power can easily miss a real effect, which leads to the next point.

## Power by simulation

The **power** of a test is the probability that it rejects $H_0$ when $H_0$ is false, for a particular kind and size of departure. To estimate it, simulate many data sets with a known effect, run the test on each, and count the rejections:

```r
power_at <- function(drift, reps = 200, B = 200) {
  mean(replicate(reps, {
    yy <- sample(n) - drift * (x - mean(x))      # data with a known trend
    tt <- replicate(B, stat(x, sample(yy)))
    (1 + sum(abs(tt) >= abs(stat(x, yy)))) / (B + 1) < 0.05
  }))
}
sapply(c(0, 0.05, 0.1), power_at)                # drift 0 gives the type I error, ~0.05
```

At drift 0 the rejection rate should be close to the significance level (here 5%). That's a useful sanity check of the whole test.

In the lab we generated one data set per effect size, recorded its p-value and saw they were all below 0.05. That shows the test detects strong effects, but a single data set per effect size isn't really an estimate of power. In one report version we also wrote that power 1 means "all our rejections are correct, which solidifies the opinion that the lottery is not random". That mixes things up: power is about the test's behaviour on simulated data with a known effect. It says nothing directly about the real data.

For what the bootstrap is good for (standard errors and confidence intervals), see [bootstrap and jackknife](/notes/bootstrap-and-jackknife-confidence-intervals/).

## Key takeaways

- A permutation test makes the null distribution by shuffling one variable against the other, without replacement. Any statistic works.
- Use $p = (1 + \#\text{extreme})/(B+1)$; never report $p = 0$ from a finite number of shuffles.
- $H_0$ is the "nothing going on" hypothesis. Small $p$ is evidence against it; large $p$ doesn't prove it.
- Bootstrap distributions are centred on the observed statistic, not on $H_0$. Use them for confidence intervals.
- Power is the rejection rate over many simulated data sets with a known effect; check the rate at zero effect too.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 5 (2020).
- Stephen E. Fienberg, "Randomization and social affairs: the 1970 draft lottery", *Science* 171(3968), 1971: https://doi.org/10.1126/science.171.3968.255
- Bradley Efron and Robert J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993, Chapters 15 (permutation tests) and 16 (hypothesis testing with the bootstrap).
- Belinda Phipson and Gordon K. Smyth, "Permutation p-values should never be zero", *Statistical Applications in Genetics and Molecular Biology* 9(1), 2010: https://doi.org/10.2202/1544-6115.1585
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 9 (bootstrapping).
