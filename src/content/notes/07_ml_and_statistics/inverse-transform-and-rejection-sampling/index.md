---
title: "Making random numbers: the inverse-CDF method and rejection sampling"
slug: inverse-transform-and-rejection-sampling
date: 2020-02-08
topic: ML & Statistics
tags: [university-lab, random-number-generation, monte-carlo, rejection-sampling, inverse-transform, laplace-distribution]
summary: "How to turn uniform random numbers into draws from any distribution with a known inverse CDF, and how to sample a normal distribution by proposing from a Laplace and rejecting some proposals."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "Computational Statistics (2nd ed.), Chapter 6: Simulation and Monte Carlo integration"
    author: "Geof H. Givens, Jennifer A. Hoeting"
    type: book
  - title: "Non-Uniform Random Variate Generation"
    author: "Luc Devroye"
    url: "http://luc.devroye.org/rnbookindex.html"
    type: book
  - title: "Introducing Monte Carlo Methods with R, Chapter 2"
    author: "Christian P. Robert, George Casella"
    type: book


status: draft
---

Every simulation starts from a generator of uniform numbers on $(0, 1)$. Everything else, from normal draws to MCMC proposals, is built on top of that. The third computational statistics lab had us build two of the classic tools by hand: the inverse-CDF method (for the Laplace distribution) and rejection sampling (for the normal, using the Laplace as a helper).

## The inverse-CDF method

If $U \sim \text{Uniform}(0, 1)$ and $F$ is a continuous CDF, then

$$
X = F^{-1}(U)
$$

has CDF $F$. Here $F(x) = P(X \le x)$ and $F^{-1}$ is its inverse. The one-line proof: $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$, because $F$ is increasing and $P(U \le u) = u$.

The way I picture it: the CDF maps the real line onto $(0,1)$, squeezing regions where the density is low and stretching regions where it's high. Running it backwards on evenly spread uniforms puts more points where the density is high.

**Example: the Laplace (double exponential) distribution** with location 0 and scale 1 has density $f(x) = \tfrac12 e^{-|x|}$. Integrating each half:

$$F(x) = \begin{cases} \tfrac12 e^{x}, & x < 0 \\ 1 - \tfrac12 e^{-x}, & x \ge 0 \end{cases}
\qquad\Rightarrow\qquad
F^{-1}(u) = \begin{cases} \log(2u), & 0 < u < \tfrac12 \\ -\log\big(2(1-u)\big), & \tfrac12 \le u < 1. \end{cases}$$

The split at $u = 1/2$ comes from $F(0) = 1/2$.

The same idea works for discrete distributions: line up the outcomes, make an interval of length $p_k$ for each, draw $U$ and see which interval it falls in. In the same lab we used this to pick cities with probability proportional to population (cumulative sums of populations as interval ends). Sampling *without* replacement then means removing the chosen city and renormalising before the next draw.

## Rejection sampling

The inverse CDF needs $F^{-1}$ in closed form. The normal distribution doesn't have one. Rejection sampling only needs the target density $f$ (even just up to a constant) and an **envelope**: a density $g$ that's easy to sample and a constant $c$ with

$$
f(x) \le c\, g(x) \quad \text{for all } x.
$$

Then:

1. Draw $Y \sim g$ and $U \sim \text{Uniform}(0,1)$.
2. If $U \le \dfrac{f(Y)}{c\,g(Y)}$, accept $Y$. Otherwise throw it away and go back to step 1.

Accepted values have density exactly $f$. Geometrically: you're throwing points uniformly under the curve $c\,g(x)$ and keeping only those that also fall under $f(x)$.

**Acceptance rate.** The probability that one proposal is accepted is

$$
P(\text{accept}) = \int \frac{f(y)}{c\,g(y)}\, g(y)\,dy = \frac{1}{c}\int f(y)\,dy = \frac{1}{c}.
$$

So $c$ is the expected number of proposals per accepted draw, and the number of tries until a success is geometric. You want $c$ as close to 1 as possible, which means choosing $g$ shaped like $f$.

## Normal from Laplace

Target $f(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$, envelope $g(x) = \frac12 e^{-|x|}$. The Laplace has heavier tails than the normal, which is what you need: an envelope with lighter tails than the target would fail far out, where no constant $c$ can cover $f$.

The ratio is

$$
\frac{f(x)}{g(x)} = \frac{2}{\sqrt{2\pi}}\, e^{|x| - x^2/2}.
$$

The exponent $|x| - x^2/2$ is largest at $|x| = 1$, where it equals $1/2$. So the smallest valid constant is

$$
c = \frac{2}{\sqrt{2\pi}} e^{1/2} = \sqrt{\frac{2e}{\pi}} \approx 1.3155,
$$

and the acceptance test simplifies nicely: $f(y)/(c\,g(y)) = e^{|y| - y^2/2 - 1/2}$. Expected acceptance rate $1/c \approx 0.760$, so about 24% of proposals are rejected. In the lab we generated 2000 normal draws and needed 629 rejections, an observed rejection rate of $629/2629 \approx 0.239$, right on the theoretical $0.240$.

```r
rlaplace01 <- function(n) {                     # inverse CDF, vectorised
  u <- runif(n)
  ifelse(u < 0.5, log(2 * u), -log(2 * (1 - u)))
}
c_env <- sqrt(2 * exp(1) / pi)                  # about 1.3155
rnorm_reject <- function(n) {
  out <- numeric(0); tries <- 0
  while (length(out) < n) {
    y <- rlaplace01(n); u <- runif(n)           # propose a batch
    keep <- u <= exp(abs(y) - y^2 / 2 - 0.5)    # f / (c g), always <= 1
    out <- c(out, y[keep]); tries <- tries + n
  }
  list(x = out[1:n], accept_rate = length(out) / tries)
}
set.seed(3)
s <- rnorm_reject(1e5)
c(theory = 1 / c_env, observed = s$accept_rate)  # 0.760 vs 0.759
c(mean = mean(s$x), sd = sd(s$x))                # about 0 and 1
```

Proposing in batches instead of one at a time (our lab code used a `repeat` loop per draw) gives the same distribution and is much faster in R.

When no good envelope exists, MCMC is the next step; see [Metropolis-Hastings](/notes/metropolis-hastings-and-gelman-rubin/).

## Key takeaways

- $F^{-1}(U)$ with $U$ uniform has CDF $F$. Use it whenever the inverse CDF has a closed form; for discrete outcomes, it's "which cumulative interval did $U$ land in".
- Rejection sampling needs $f \le c\,g$ everywhere. Accept with probability $f/(cg)$; the acceptance rate is $1/c$.
- The envelope must have tails at least as heavy as the target.
- For normal from Laplace, $c = \sqrt{2e/\pi} \approx 1.32$, so about 76% of proposals are kept.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 3 (2020).
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 6.
- Luc Devroye, *Non-Uniform Random Variate Generation*, Springer, 1986, Chapter II (inversion and rejection): http://luc.devroye.org/rnbookindex.html
- Christian P. Robert and George Casella, *Introducing Monte Carlo Methods with R*, Springer, 2010, Chapter 2.
