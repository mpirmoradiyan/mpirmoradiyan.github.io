---
title: "Conjugate Bayesian linear regression: prior checks, posterior draws, and credible bands"
slug: conjugate-bayesian-linear-regression
date: 2020-04-27
topic: Bayesian Methods
tags: [university-lab, bayesian-inference, linear-regression, conjugate-prior, prior-predictive-check, shrinkage]
summary: "How the normal / scaled-inverse-chi-squared prior makes Bayesian linear regression exact, how to check a prior by drawing regression curves from it, and why a credible band for the regression line shouldn't contain most of the data."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Data Analysis (3rd ed.), Chapter 14 (introduction to regression models)"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book


status: draft
---

Linear regression is the one model where Bayesian inference is exact and cheap: with the right prior, the posterior has a closed form and you can sample from it directly, no MCMC. In the second Bayesian Learning lab (pair work) we fitted a quadratic curve to a year of daily temperatures this way. The useful lessons weren't the formulas so much as how to pick a prior by looking at it, and how to read the bands you get at the end.

## The model and the conjugate prior

The model is ordinary regression:

$$
y = X\beta + \varepsilon, \qquad \varepsilon \sim N(0, \sigma^2 I_n),
$$

where $y$ is the $n$-vector of responses, $X$ the $n \times p$ design matrix (for a quadratic in time $t$ the columns are $1, t, t^2$), $\beta$ the $p$ coefficients and $\sigma^2$ the noise variance.

The conjugate prior comes in two layers:

$$
\sigma^2 \sim \text{Inv-}\chi^2(\nu_0, \sigma_0^2), \qquad \beta \mid \sigma^2 \sim N\!\left(\mu_0,\; \sigma^2 \Omega_0^{-1}\right).
$$

Here $\mu_0$ is your prior guess for the coefficients, $\Omega_0$ is a *precision* matrix (bigger = more confident), $\sigma_0^2$ is a prior guess for the noise variance and $\nu_0$ says how many "observations" that guess is worth.

The posterior has the same form, with updated hyperparameters:

$$
\Omega_n = X^\top X + \Omega_0, \qquad \mu_n = \Omega_n^{-1}\left(X^\top X \hat\beta + \Omega_0 \mu_0\right),
$$

$$
\nu_n = \nu_0 + n, \qquad \nu_n \sigma_n^2 = \nu_0 \sigma_0^2 + y^\top y + \mu_0^\top \Omega_0 \mu_0 - \mu_n^\top \Omega_n \mu_n,
$$

where $\hat\beta = (X^\top X)^{-1}X^\top y$ is the least-squares estimate. I like reading $\mu_n$ as a precision-weighted average of the data's answer $\hat\beta$ and the prior's answer $\mu_0$.

**A mistake in my draft.** When sampling $\sigma^2$ from the posterior, I used the sample variance of $y$ as the scale instead of $\sigma_n^2$. That ignores the regression completely: the variance of the raw temperatures includes the whole seasonal swing, which the curve is supposed to explain. The scale has to be $\sigma_n^2$ from the formula above, which is (roughly) the residual variance around the fitted curve, blended with the prior.

## Checking a prior by looking at it

Numbers like "$\Omega_0 = 0.01 I$" mean nothing to me on their own. What helped was drawing $(\sigma^2, \beta)$ from the prior many times and plotting the regression curve for each draw. If the curves include summers at $-40$°C, the prior is too vague; if they're all the same line, it's too confident. You adjust $\mu_0$, $\Omega_0$, $\nu_0$, $\sigma_0^2$ until the cloud of curves looks like something you'd actually believe before seeing the data. This is a *prior predictive check*, and it's a habit worth keeping for any Bayesian model.

## Sampling the posterior

```r
set.seed(2)
n <- 100; t <- (1:n) / n
X <- cbind(1, t, t^2)
y <- drop(X %*% c(-5, 90, -85)) + rnorm(n, 0, 4)        # synthetic seasonal data
mu0 <- c(0, 0, 0); Omega0 <- diag(0.01, 3); nu0 <- 4; s20 <- 10

XtX    <- crossprod(X)
bhat   <- solve(XtX, crossprod(X, y))
Omegan <- XtX + Omega0
mun    <- solve(Omegan, XtX %*% bhat + Omega0 %*% mu0)
nun    <- nu0 + n
s2n    <- drop(nu0 * s20 + crossprod(y) + t(mu0) %*% Omega0 %*% mu0 -
               t(mun) %*% Omegan %*% mun) / nun

draws <- t(replicate(5000, {
  sigma2 <- nun * s2n / rchisq(1, nun)                     # scaled Inv-chi2
  beta   <- mun + t(chol(sigma2 * solve(Omegan))) %*% rnorm(3)
  c(beta, sigma2)
}))
t_peak <- -draws[, 2] / (2 * draws[, 3])                    # where the curve peaks
quantile(t_peak, c(0.025, 0.5, 0.975))                      # 0.497 0.517 0.537
```

The last two lines show the same trick as in any simulation-based posterior: the time of the maximum is $\tilde t = -\beta_1 / (2\beta_2)$ (set the derivative $\beta_1 + 2\beta_2 t$ to zero), so its posterior is that formula applied to every draw.

**Another slip.** In my draft I also wrote the maximum *value* as $\beta_0 - \frac{\beta_1^2}{2\beta_2} + \frac{\beta_1^2}{4\beta_2^3}$. Plugging $\tilde t$ back in actually gives

$$
f(\tilde t) = \beta_0 + \beta_1\tilde t + \beta_2 \tilde t^2 = \beta_0 - \frac{\beta_1^2}{4\beta_2}.
$$

## Credible band vs prediction band

For each time point you can compute $f(t) = \beta_0 + \beta_1 t + \beta_2 t^2$ for every posterior draw, and take the 2.5% and 97.5% quantiles. Connecting these gives a 95% band for the *regression function*. It's narrow, and most data points fall outside it. That's correct, not a failure: the band describes uncertainty about the mean curve, and with a year of data the mean is pinned down well. Individual days scatter around the mean with variance $\sigma^2$.

If you want a band that should contain about 95% of observations, you need the *posterior predictive*: for each draw, add noise, $\tilde y = f(t) + \varepsilon$ with $\varepsilon \sim N(0, \sigma^2)$ using that draw's $\sigma^2$, then take quantiles. That band is much wider. (Our report described the narrow band as "the interval the median falls into", which isn't quite it: it's a pointwise interval for $f(t)$, the expected temperature.)

## A prior that guards against overfitting

Suppose you fit a degree-7 polynomial but suspect the high-order terms aren't needed. The conjugate prior can express that directly: set the prior mean of the high-order coefficients to **zero** and give them a **large precision** in $\Omega_0$ (small prior variance), while keeping the low-order terms loose. The posterior then shrinks those coefficients towards zero unless the data really demand them. This is ridge regression in disguise: with $\mu_0 = 0$ and $\Omega_0 = \lambda I$, the posterior mean $\mu_n = (X^\top X + \lambda I)^{-1}X^\top y$ is exactly the ridge estimate.

In my draft I suggested non-zero prior means for the extra terms, and our report talked about putting the prior mean "far from the data". Both miss the point. Shrinkage means pulling towards zero (no effect), not towards some other value.

## Key takeaways

- With the normal / Inv-$\chi^2$ prior, the posterior of $(\beta, \sigma^2)$ is exact: draw $\sigma^2$ first, then $\beta \mid \sigma^2$.
- The posterior scale for $\sigma^2$ is $\sigma_n^2$, not the raw variance of $y$.
- Check a prior by plotting curves drawn from it.
- A credible band for $f(t)$ is not a prediction band; add $\sigma^2$ noise for that.
- Zero prior means plus high prior precision on extra terms = shrinkage = ridge regression.

## Resources

- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Section 14.2 (Bayesian analysis of classical regression) and Section 14.8 (informative prior distributions): http://www.stat.columbia.edu/~gelman/book/
- Kevin P. Murphy, *Machine Learning: A Probabilistic Perspective*, Section 7.6 (Bayesian linear regression)
- Jonah Gabry et al., "Visualization in Bayesian workflow", *JRSS A* 182(2) (2019), on prior predictive checks: https://doi.org/10.1111/rssa.12378
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
