---
title: "The EM algorithm for missing data: a worked exponential example"
slug: em-algorithm-for-missing-data
date: 2020-03-06
topic: Statistical Inference
tags: [university-lab, em-algorithm, missing-data, maximum-likelihood, exponential-distribution]
summary: "How EM handles missing values by alternating between filling in the expected complete-data log-likelihood and maximising it, worked through for a model with exponential responses, and why its fixed point equals the observed-data MLE."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "Maximum likelihood from incomplete data via the EM algorithm"
    author: "A. P. Dempster, N. M. Laird, D. B. Rubin"
    url: "https://doi.org/10.1111/j.2517-6161.1977.tb01600.x"
    type: paper
  - title: "Computational Statistics (2nd ed.), Chapter 4: EM optimization methods"
    author: "Geof H. Givens, Jennifer A. Hoeting"
    type: book
  - title: "The EM Algorithm and Extensions (2nd ed.)"
    author: "Geoffrey J. McLachlan, Thriyambakam Krishnan"
    type: book


status: draft
---

Here's the situation that EM (expectation-maximisation) solves. You want a maximum likelihood estimate, but some of your data are missing. If you had the missing values, the likelihood would be easy to maximise. You don't have them. So EM goes in circles on purpose: guess the parameter, work out what the log-likelihood would look like *on average* given that guess and the data you do have, maximise that, and repeat. For a long time EM felt abstract to me. What finally made it click was a small example from the last computational statistics lab, where every step can be done by hand.

## The model

For each $i = 1, \dots, n$ there's a known positive input $X_i$ and two responses:

$$
Y_i \sim \text{Exp}\!\left(\text{rate} = \frac{X_i}{\lambda}\right), \qquad Z_i \sim \text{Exp}\!\left(\text{rate} = \frac{X_i}{2\lambda}\right),
$$

all independent, where $\lambda > 0$ is the one unknown parameter. So $E[Y_i] = \lambda / X_i$ and $E[Z_i] = 2\lambda / X_i$. Some $Z_i$ are missing. Call the set of observed ones $O$ and the $M$ missing ones $\mathcal M$.

## Complete-data log-likelihood

If all $Z_i$ were known, the density $\theta e^{-\theta y}$ of an exponential with rate $\theta$ gives

$$\ell_c(\lambda) = \sum_{i=1}^n \left[\log\frac{X_i}{\lambda} - \frac{X_i Y_i}{\lambda} + \log\frac{X_i}{2\lambda} - \frac{X_i Z_i}{2\lambda}\right]
= -2n\log\lambda - \frac{1}{\lambda}\sum_{i=1}^n X_i Y_i - \frac{1}{2\lambda}\sum_{i=1}^n X_i Z_i + \text{const},$$

where "const" collects $2\sum\log X_i - n\log 2$, which doesn't involve $\lambda$.

(In our lab report the constant was written as $-\log 2$ instead of $-n\log 2$. It has no effect on the estimate, since it disappears when you differentiate, but it's the kind of slip that matters if you ever compare log-likelihood values.)

## E-step

Take the expectation of $\ell_c$ over the missing $Z_j$, given the observed data and the current estimate $\lambda_t$. This function is usually called $Q(\lambda \mid \lambda_t)$. Since $\ell_c$ is *linear* in each $Z_j$, we only need $E[Z_j \mid \lambda_t] = 2\lambda_t / X_j$, so each missing term becomes $X_j \cdot 2\lambda_t / X_j \,/\, (2\lambda) = \lambda_t/\lambda$:

$$
Q(\lambda \mid \lambda_t) = -2n\log\lambda - \frac{1}{\lambda}\sum_{i=1}^n X_i Y_i - \frac{1}{2\lambda}\sum_{i \in O} X_i Z_i - \frac{M\lambda_t}{\lambda} + \text{const}.
$$

This is the easy part to get wrong: in general the E-step takes the expectation of the *log-likelihood*, not "plug in the expected value of the missing data". Here the two happen to coincide, because the log-likelihood is linear in $Z_j$. In a model where $Z_j$ enters non-linearly (say through $\log Z_j$ or $Z_j^2$), plugging in $E[Z_j]$ would be wrong.

## M-step

Set $\partial Q / \partial\lambda = 0$:

$$-\frac{2n}{\lambda} + \frac{1}{\lambda^2}\left(\sum_{i=1}^n X_i Y_i + \frac12\sum_{i\in O} X_i Z_i + M\lambda_t\right) = 0
\;\;\Rightarrow\;\;
\lambda_{t+1} = \frac{\sum_{i=1}^n X_i Y_i + \frac12\sum_{i\in O} X_i Z_i + M\lambda_t}{2n}.$$

Each iteration is one line. In the lab, starting from $\lambda_0 = 100$ and stopping once successive values differed by less than 0.001, it took 5 iterations to reach $\lambda \approx 10.696$. Plotting $\lambda/X$ and $2\lambda/X$ over the data showed curves that followed both responses well.

## The fixed point is the observed-data MLE

Call $A = \sum_i X_iY_i + \frac12\sum_{i\in O}X_iZ_i$. The update is $\lambda_{t+1} = (A + M\lambda_t)/(2n)$. At a fixed point $\lambda^* = (A + M\lambda^*)/(2n)$, so

$$
\lambda^* = \frac{A}{2n - M} = \frac{\sum_{i} X_i Y_i + \frac12 \sum_{i \in O} X_i Z_i}{n + |O|}.
$$

That's exactly what you get by maximising the likelihood of only the data you actually have ($n$ values of $Y$, $|O|$ values of $Z$). So EM didn't add information; it found the same answer by a different route. That's the general guarantee: EM never decreases the observed-data likelihood, and its fixed points are stationary points of it.

The update also shows how EM converges: the error shrinks by a factor $M/(2n)$ per step, the share of "missing information". With few missing values that's fast; with lots of missingness EM slows to a crawl.

```r
set.seed(8)
n <- 100; lambda_true <- 10
X <- runif(n, 1, 10)
Y <- rexp(n, rate = X / lambda_true)
Z <- rexp(n, rate = X / (2 * lambda_true))
Z[sample(n, 20)] <- NA                         # 20 values go missing
obs <- !is.na(Z); M <- sum(!obs)

lambda <- 100
repeat {
  new <- (sum(X * Y) + sum(X[obs] * Z[obs]) / 2 + M * lambda) / (2 * n)
  if (abs(new - lambda) < 1e-3) break
  lambda <- new
}
c(em = new, direct = (sum(X * Y) + sum(X[obs] * Z[obs]) / 2) / (n + sum(obs)))
#      em   direct
# 10.39284 10.39275
```

For the more everyday way to deal with missing values, see [missing data and multiple imputation](/notes/missing-data-and-multiple-imputation/). EM gets more interesting when the hidden part is bigger than a few missing values: see [EM for a mixture of Bernoullis](/notes/em-for-bernoulli-mixtures/) and [EM for state space models](/notes/em-for-state-space-models/).

## Key takeaways

- EM alternates: E-step builds $Q(\lambda\mid\lambda_t)$, the expected complete-data log-likelihood; M-step maximises it.
- Plugging in $E[\text{missing}]$ is only valid when the log-likelihood is linear in the missing values.
- Each EM step never decreases the observed-data likelihood; the fixed point is the observed-data MLE.
- Convergence speed depends on the fraction of missing information (here $M/2n$ per step).

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 6 (2020).
- A. P. Dempster, N. M. Laird and D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm", *Journal of the Royal Statistical Society: Series B* 39(1), 1977: https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- Geof H. Givens and Jennifer A. Hoeting, *Computational Statistics*, 2nd ed., Wiley, 2013, Chapter 4.
- Geoffrey J. McLachlan and Thriyambakam Krishnan, *The EM Algorithm and Extensions*, 2nd ed., Wiley, 2008.
