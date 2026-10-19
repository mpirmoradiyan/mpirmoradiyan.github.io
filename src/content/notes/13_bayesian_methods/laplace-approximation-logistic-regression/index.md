---
title: "The Laplace approximation: a normal posterior from optim and the Hessian"
slug: laplace-approximation-logistic-regression
date: 2020-04-27
topic: Bayesian Methods
tags: [university-lab, bayesian-inference, laplace-approximation, logistic-regression, optimization]
summary: "How to approximate a non-conjugate posterior, like Bayesian logistic regression, with a normal distribution centred at the posterior mode, using nothing but a numerical optimiser and its Hessian."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Data Analysis (3rd ed.), Section 4.1 (normal approximations to the posterior)"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book
  - title: "Pattern Recognition and Machine Learning, Section 4.4 (the Laplace approximation)"
    author: "Christopher M. Bishop"
    url: "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/"
    type: book


status: draft
---

Logistic regression with a normal prior has no closed-form posterior. But with a decent amount of data the posterior is usually close to a normal bump, so we can approximate it with one: put the centre at the posterior mode and get the width from the curvature there. In the second Bayesian Learning lab (pair work) we did exactly this, and the whole thing turned out to be one call to `optim`.

## The idea

Take the log posterior $\ell(\beta) = \log p(\beta \mid y)$ and expand it around its maximum $\tilde\beta$ (the posterior mode). The first derivative is zero there, so a second-order Taylor expansion gives

$$
\ell(\beta) \approx \ell(\tilde\beta) - \tfrac12 (\beta - \tilde\beta)^\top J(\tilde\beta)\, (\beta - \tilde\beta),
$$

where $J(\tilde\beta) = -\left.\frac{\partial^2 \ell}{\partial\beta\,\partial\beta^\top}\right|_{\beta = \tilde\beta}$ is the negative Hessian (the observed information). A log density that's quadratic in $\beta$ is a normal, so

$$
\beta \mid y \;\approx\; N\!\left(\tilde\beta,\; J(\tilde\beta)^{-1}\right).
$$

Sharp peak (big curvature) means small posterior variance; flat peak means large variance. That's the whole approximation. It's the Bayesian cousin of the usual "MLE plus inverse-Hessian standard errors" story, with the prior included in $\ell$.

## Logistic regression

The model is $\Pr(y_i = 1 \mid x_i) = \frac{\exp(x_i^\top\beta)}{1 + \exp(x_i^\top\beta)}$, where $x_i$ is the feature vector of observation $i$ (with a leading 1 for the intercept). The log-likelihood is

$$
\log p(y \mid X, \beta) = \sum_{i=1}^n \left[ y_i\, x_i^\top\beta - \log\!\left(1 + e^{x_i^\top\beta}\right) \right].
$$

With the prior $\beta \sim N(0, \tau^2 I)$ the log posterior is this plus $-\frac{1}{2\tau^2}\|\beta\|^2$ (plus a constant).

**A bug worth remembering.** In my draft I wrote the first term as $(x_i^\top\beta)^{y_i}$, and coded it as `linFun^y`. The correct term is $y_i \cdot x_i^\top\beta$: it comes from $\log\left[(e^{x_i^\top\beta})^{y_i}\right] = y_i\, x_i^\top\beta$. The power belongs to the exponential, not to the linear predictor. With $y_i \in \{0, 1\}$ the wrong version gives $1$ instead of $0$ for $y_i = 0$, which shifts things around quietly without crashing. Our final report had the right form, and its posterior mode matched `glm` closely, which is a great sanity check: with a vague prior ($\tau = 10$), the mode should be almost the MLE.

## Code

```r
set.seed(3)
n <- 200
X <- cbind(1, rnorm(n), rbinom(n, 2, 0.4))
y <- rbinom(n, 1, plogis(X %*% c(0.5, 1, -1.2)))
tau <- 10

log_post <- function(beta, X, y, tau) {
  eta <- drop(X %*% beta)
  sum(y * eta - log1p(exp(eta))) +            # log-likelihood
    sum(dnorm(beta, 0, tau, log = TRUE))      # N(0, tau^2 I) prior
}
fit <- optim(rep(0, ncol(X)), log_post, X = X, y = y, tau = tau,
             method = "BFGS", control = list(fnscale = -1), hessian = TRUE)
beta_mode <- fit$par
beta_cov  <- solve(-fit$hessian)             # J(mode)^-1

cbind(mode = beta_mode, sd = sqrt(diag(beta_cov)),
      glm  = coef(glm(y ~ X - 1, family = binomial)))
```

`fnscale = -1` makes `optim` maximise instead of minimise. `hessian = TRUE` returns the Hessian of the log posterior at the optimum, which is negative definite at a maximum, hence the minus sign.

## Reading the result

From the approximate posterior you get a credible interval for each coefficient: $\tilde\beta_j \pm 1.96\, \text{sd}_j$. In the lab, the coefficient for the number of small children came out with a posterior mode of about $-1.36$ and a 95% interval of roughly $(-2.12, -0.59)$.

Was it "an important determinant"? Our report argued yes because its absolute value was the largest. That's a weak argument: coefficient size depends on the units of each feature (a coefficient on income in thousands and one on a count of children aren't comparable). The better argument is that the whole interval sits well away from zero, so the data are clear about the sign, and on the odds scale $e^{-1.36} \approx 0.26$ per extra small child is a big effect. Comparing magnitudes only makes sense after standardising the features.

## When to trust it

The approximation is good when the posterior really is close to normal: lots of data relative to the number of parameters, and parameters away from boundaries. It gets worse with small samples, with separation in logistic regression (where the MLE runs off to infinity), and for parameters like variances that are bounded below (better to approximate $\log\sigma$ than $\sigma$). In the next lab we checked it against MCMC, see [random-walk Metropolis](/notes/random-walk-metropolis/).

## Key takeaways

- Laplace approximation: posterior $\approx N(\text{mode}, (-\text{Hessian})^{-1})$ of the log posterior.
- One `optim(..., hessian = TRUE)` call gives you both pieces.
- Logistic log-likelihood: $\sum_i [y_i x_i^\top\beta - \log(1+e^{x_i^\top\beta})]$; the $y_i$ multiplies, it isn't a power.
- With a vague prior, the mode should match `glm`; use that as a check.
- Judge a coefficient by its interval and on a standardised scale, not by raw magnitude.

## Resources

- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Section 4.1: http://www.stat.columbia.edu/~gelman/book/
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 4.4 (Laplace approximation) and 4.5 (Bayesian logistic regression): https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
- R documentation for `optim`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optim.html
- 732A91 Bayesian Learning, Linköping University (course lab, spring 2020).
