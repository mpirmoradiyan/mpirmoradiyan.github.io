---
title: "Posterior draws, transformed quantities, and equal-tail vs HPD intervals"
slug: equal-tail-vs-hpd-credible-intervals
date: 2020-04-14
topic: ML & Statistics
tags: [university-lab, bayesian-inference, credible-intervals, monte-carlo, posterior-simulation]
summary: "Once you can draw from a posterior, the posterior of any function of the parameters comes for free, and you can summarise it with an equal-tail or a highest-posterior-density interval. The two only agree when the posterior is symmetric."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Data Analysis (3rd ed.), Chapters 2-3"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book
  - title: "Gini coefficient (Wikipedia), section on the log-normal distribution"
    author: "Wikipedia"
    url: "https://en.wikipedia.org/wiki/Gini_coefficient"
    type: docs


status: draft
---

The first Bayesian Learning lab was mostly about one habit: once you can draw random numbers from a posterior, you can answer almost any question about it by just working with the draws. Probabilities, means, the distribution of some messy function of the parameter, intervals. This note collects that habit and the one place where it gets subtle: which interval to report. (The lab was done in a pair.)

## Draws are the posterior

Say the posterior of a parameter $\theta$ is something we can sample from, like $\theta \mid y \sim \text{Beta}(\alpha, \beta)$. With $N$ draws $\theta^{(1)}, \dots, \theta^{(N)}$:

- a posterior probability is a fraction: $\Pr(\theta > c \mid y) \approx \frac{1}{N}\sum_{i} \mathbb{1}[\theta^{(i)} > c]$,
- a posterior mean is an average: $E[\theta \mid y] \approx \frac{1}{N}\sum_i \theta^{(i)}$,
- the Monte Carlo error of these shrinks like $1/\sqrt{N}$, so going from 100 to 10,000 draws buys one extra decimal.

In the lab, 10,000 draws gave $\Pr(\theta > 0.3 \mid y) = 0.4392$ against an exact value of $0.43995$ from `pbeta`. Close, and as close as $1/\sqrt{N}$ says it should be.

The nice part is transformations. If you want the posterior of $\phi = g(\theta)$, you don't need a change-of-variables formula. You just compute $\phi^{(i)} = g(\theta^{(i)})$ for every draw. The log-odds $\log\frac{\theta}{1-\theta}$ is one line of code.

## A less trivial example: the Gini coefficient

Incomes are often modelled as log-normal: $\log y_i \sim N(\mu, \sigma^2)$. With $\mu$ known and the prior $p(\sigma^2) \propto 1/\sigma^2$, the posterior is a scaled inverse chi-squared,

$$
\sigma^2 \mid y \sim \text{Inv-}\chi^2(n, \tau^2), \qquad \tau^2 = \frac{1}{n}\sum_{i=1}^n (\log y_i - \mu)^2,
$$

where $n$ is the number of observations and $\tau^2$ is the average squared deviation of the log-incomes from $\mu$. You sample it as $\sigma^2 = n\tau^2 / X$ with $X \sim \chi^2_n$.

**A mistake I made here.** In my draft I checked my draws against `rinvchisq(N, df = n)` without a scale argument. That's the *unscaled* inverse chi-squared, so it's a different distribution, and the comparison didn't mean anything. The theoretical density has to include $\tau^2$:

$$
p(\sigma^2) = \frac{(\tau^2 n/2)^{n/2}}{\Gamma(n/2)}\,(\sigma^2)^{-(n/2+1)} \exp\!\left(-\frac{n\tau^2}{2\sigma^2}\right).
$$

For log-normal incomes the Gini coefficient (0 = everyone earns the same, 1 = one person earns everything) is

$$
G = 2\,\Phi\!\left(\frac{\sigma}{\sqrt 2}\right) - 1,
$$

where $\Phi$ is the standard normal CDF. So the posterior of $G$ is just `2 * pnorm(sqrt(sigma2) / sqrt(2)) - 1` applied to each draw of $\sigma^2$. No algebra needed.

## Two kinds of 90% interval

A 90% credible interval is any interval holding 90% of the posterior mass. There are many of those, and two common choices:

- **Equal-tail**: cut 5% off each end. It's just the 5% and 95% quantiles of the draws.
- **Highest posterior density (HPD)**: the region where the density is above some level $k$, with $k$ chosen so the region holds 90%. Every point inside is more plausible than every point outside.

For a symmetric, single-peaked posterior they're the same. For a skewed one they're not. The posterior of $G$ is right-skewed, and in the lab the equal-tail interval was $(0.160, 0.334)$ while the HPD was $(0.147, 0.312)$: shifted towards the peak and a bit shorter. The HPD is always the shortest interval for a given level, which is its selling point.

The way I think about the trade-off:

- Equal-tail intervals behave well under transformation. If $(a, b)$ is a 90% equal-tail interval for $\sigma^2$, then $(g(a), g(b))$ is one for $G = g(\sigma^2)$, because $g$ is increasing and quantiles map through it. HPD intervals don't have this property.
- HPD intervals can be a union of pieces if the posterior has two peaks. That's honest, but harder to report.

## Computing an HPD interval yourself

In the lab I first used `HDInterval::hdi`, but computing it by hand from a kernel density estimate is short and shows what it is. Sort the density grid from highest to lowest and keep adding points until they hold 90% of the mass:

```r
set.seed(1)
draws <- rgamma(10000, shape = 3, rate = 20)   # any skewed posterior sample

equal_tail <- quantile(draws, c(0.05, 0.95))

d <- density(draws)                           # kernel density on an even grid
ord <- order(d$y, decreasing = TRUE)          # highest density first
mass <- cumsum(d$y[ord]) / sum(d$y)           # even spacing cancels out
inside <- ord[mass <= 0.90]
hpd <- range(d$x[inside])                     # OK for a single-peaked posterior

round(equal_tail, 3)   # 0.042 0.317
round(hpd, 3)          # 0.023 0.279
```

The HPD is pulled towards zero where the gamma density is highest, and it's narrower. If the posterior had two peaks, `range()` would be wrong and you'd need to find the separate runs in `inside`.

## Key takeaways

- With posterior draws, probabilities are fractions, expectations are averages, and the posterior of $g(\theta)$ is $g$ applied to each draw.
- Monte Carlo error falls like $1/\sqrt{N}$.
- A scaled inverse chi-squared needs its scale $\tau^2$; the unscaled version is a different distribution.
- Equal-tail and HPD intervals agree only for symmetric posteriors. HPD is shortest; equal-tail is invariant under monotone transformations.

## Resources

- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Chapter 2 (single-parameter models, posterior intervals) and Chapter 3 (normal model with unknown variance): http://www.stat.columbia.edu/~gelman/book/
- Rob J. Hyndman, "Computing and graphing highest density regions", *The American Statistician* 50(2) (1996): https://doi.org/10.1080/00031305.1996.10474359
- Wikipedia, "Gini coefficient", log-normal case: https://en.wikipedia.org/wiki/Gini_coefficient
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
