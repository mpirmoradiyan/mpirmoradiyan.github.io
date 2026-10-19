---
title: "Floating-point traps: equality, absorption and catastrophic cancellation"
slug: floating-point-rounding-and-cancellation
date: 2020-01-26
topic: Programming & Data
tags: [university-lab, floating-point, numerical-stability, rounding-error, variance, r]
summary: "Why 1/3 - 1/4 == 1/12 is FALSE in R, why a finite-difference derivative with a tiny step can return 0, and why the one-pass variance formula breaks down while Welford-style updates don't."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "What every computer scientist should know about floating-point arithmetic"
    author: "David Goldberg"
    url: "https://doi.org/10.1145/103162.103163"
    type: paper
  - title: "Accuracy and Stability of Numerical Algorithms (2nd ed.), Chapters 1-2"
    author: "Nicholas J. Higham"
    type: book
  - title: "Note on a method for calculating corrected sums of squares and products"
    author: "B. P. Welford"
    url: "https://doi.org/10.1080/00401706.1962.10490022"
    type: paper
  - title: "Computational Statistics, Chapter 2 (Numerical computations)"
    author: "James E. Gentle"
    type: book


status: draft
---

A computer stores a real number as a double: 53 binary digits of mantissa and an exponent. That's plenty for most things, but it means almost every number you type is rounded a little, and some innocent-looking formulas turn that tiny rounding into a big error. The first lab of the computational statistics course was all about this, and it's worth keeping three traps in mind.

## Trap 1: comparing doubles with `==`

A double can only hold numbers of the form (integer) $\times 2^{e}$. So a fraction is stored exactly only if its denominator is a power of two. $1/2$, $1/4$ and $3/8$ are exact. $1/3$, $1/12$ and even $0.1$ are not: their binary expansions repeat forever, so they get rounded to the nearest double.

That's why `1/3 - 1/4 == 1/12` is `FALSE`, while `1 - 1/2 == 1/2` is `TRUE`. Each side of the first comparison is rounded separately, and the two rounded results don't land on the same double.

In our group's lab report we explained this as "1/2 and 1/4 are finite numbers, while 1/3 recurs". That's half right. What matters is a finite expansion in *base 2*, not base 10. $0.1$ is a finite decimal, but `0.1 + 0.2 == 0.3` is still `FALSE` because $1/10$ has no finite binary expansion.

We "fixed" the comparison by rounding both sides to 2 decimals. That works for this one example but is fragile: two numbers that differ by $10^{-17}$ can still round to different 2-decimal values if they sit right next to a rounding boundary. The usual fix is to compare with a tolerance:

$$
|a - b| \le \varepsilon_{\text{abs}} \quad\text{or}\quad |a - b| \le \varepsilon_{\text{rel}} \max(|a|, |b|),
$$

where $a$ and $b$ are the two computed values, $\varepsilon_{\text{abs}}$ is an absolute tolerance, and $\varepsilon_{\text{rel}}$ is a relative one. In R, `isTRUE(all.equal(a, b))` does the relative version with $\varepsilon_{\text{rel}} \approx 1.5 \times 10^{-8}$.

## Trap 2: absorption in $(x + h) - x$

The forward-difference derivative is

$$
f'(x) \approx \frac{f(x + h) - f(x)}{h},
$$

where $h$ is a small step. For $f(x) = x$ the true answer is always 1. With $h = 10^{-15}$, our lab code returned $1.110223$ at $x = 1$ and $0$ at $x = 100000$.

The key number is **machine epsilon**, $\varepsilon_{\text{mach}} = 2^{-52} \approx 2.22 \times 10^{-16}$. It's the gap between 1 and the next double. Near a number $x$ the gap is roughly $\varepsilon_{\text{mach}} |x|$.

- At $x = 1$, $1 + 10^{-15}$ gets rounded to the nearest multiple of $2.22 \times 10^{-16}$, which is $5 \times 2.22 \times 10^{-16} = 1.110223 \times 10^{-15}$. Divide by $h$ and you get $1.110223$.
- At $x = 10^5$, the gap is about $1.5 \times 10^{-11}$, which is way bigger than $10^{-15}$. So $10^5 + 10^{-15}$ rounds straight back to $10^5$. The step is *absorbed*, the numerator is exactly 0, and so is the "derivative".

In the lab report we called the second case "underflow", and the group wasn't sure whether to call it cancellation. Neither word is quite right. **Underflow** means a result is smaller than the smallest representable double (around $10^{-308}$), which isn't happening here. What happens is that the addition $x + h$ loses $h$ to rounding (absorption), and then the subtraction of two nearly equal numbers exposes that loss. The practical lesson: a step must be well above $\varepsilon_{\text{mach}} |x|$. A common choice for forward differences is $h \approx \sqrt{\varepsilon_{\text{mach}}}\, \max(|x|, 1)$, which balances rounding error against the truncation error of the formula.

## Trap 3: catastrophic cancellation in the variance formula

The sample variance can be written as

$$
s^2 = \frac{1}{n-1}\left(\sum_{i=1}^n x_i^2 - \frac{1}{n}\Big(\sum_{i=1}^n x_i\Big)^2\right),
$$

where $x_1, \dots, x_n$ are the data and $n$ is the sample size. It's correct on paper and needs only one pass over the data. But if the data are like $10^8 \pm 1$, both big sums are about $n \cdot 10^{16}$, and the difference we want is about $n$. Each sum has a rounding error of roughly $\varepsilon_{\text{mach}} \cdot n \cdot 10^{16} \approx 2n$, which is the same size as the answer. Subtracting two big, nearly equal, slightly wrong numbers leaves mostly error. That's **catastrophic cancellation**.

The fix is to never form the big sums. Welford's method updates a running mean and a running sum of squared deviations:

$$
\delta_k = x_k - m_{k-1}, \qquad m_k = m_{k-1} + \frac{\delta_k}{k}, \qquad S_k = S_{k-1} + \delta_k (x_k - m_k),
$$

where $m_k$ is the mean of the first $k$ values and $S_k$ is their sum of squared deviations, so $s^2 = S_n / (n-1)$. Everything stays on the scale of the deviations, not of $x^2$. The Youngs and Cramer algorithm we used in the lab (code provided by the course) does the same thing with running totals $T_j$: it adds $(j x_j - T_j)^2 / (j(j-1))$ at each step.

```r
naive_var <- function(x) { n <- length(x); (sum(x^2) - sum(x)^2 / n) / (n - 1) }
welford_var <- function(x) {
  m <- 0; s <- 0
  for (k in seq_along(x)) {
    d <- x[k] - m
    m <- m + d / k                 # running mean
    s <- s + d * (x[k] - m)        # running sum of squared deviations
  }
  s / (length(x) - 1)
}
set.seed(1)
x <- rnorm(1e4, mean = 1e8, sd = 1)
c(naive = naive_var(x), welford = welford_var(x), R = var(x))
#    naive  welford        R
# 1.638564 1.024866 1.024866
```

The naive version is off by more than 60%. Welford matches R's `var()`, which centres the data first (two passes). Another cheap trick: subtract any value close to the mean (say $x_1$) from all data before using the one-pass formula. Variance doesn't change under a shift, but the cancellation mostly goes away.

The same rounding errors are what make badly conditioned linear systems fail; see [condition numbers](/notes/condition-number-and-ill-conditioned-regression/).

## Key takeaways

- Only fractions with a power-of-two denominator are exact in binary. $0.1$ isn't.
- Compare doubles with a tolerance (`all.equal`, or $|a-b| \le \varepsilon$), not with `==` and not by rounding to a few decimals.
- Adding a tiny number to a big one can lose it completely (absorption). Finite-difference steps should be around $\sqrt{\varepsilon_{\text{mach}}}$ times the scale of $x$, not $10^{-15}$.
- Subtracting two large, nearly equal numbers amplifies their rounding errors (catastrophic cancellation). Reformulate: centre the data, or use Welford / Youngs-Cramer updates.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 1 (2020).
- David Goldberg, "What every computer scientist should know about floating-point arithmetic", *ACM Computing Surveys* 23(1), 1991: https://doi.org/10.1145/103162.103163
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002, Chapters 1-2.
- B. P. Welford, "Note on a method for calculating corrected sums of squares and products", *Technometrics* 4(3), 1962: https://doi.org/10.1080/00401706.1962.10490022
- E. A. Youngs and E. M. Cramer, "Some results relevant to choice of sum and sum-of-product algorithms", *Technometrics* 13(3), 1971.
- James E. Gentle, *Computational Statistics*, Springer, 2009, Chapter 2.
