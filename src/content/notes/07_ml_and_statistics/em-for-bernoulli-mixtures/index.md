---
title: "EM for a mixture of Bernoullis, and which log-likelihood to watch"
slug: em-for-bernoulli-mixtures
date: 2019-11-28
topic: ML & Statistics
tags: [university-lab, em-algorithm, mixture-models, latent-variables, clustering, maximum-likelihood]
summary: "How the EM algorithm fits a mixture of multivariate Bernoulli distributions, and why the quantity you monitor must be the observed-data log-likelihood, not the expected complete-data one."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Pattern Recognition and Machine Learning, Sections 9.3 and 9.4"
    author: "Christopher M. Bishop"
    url: "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/"
    type: book
  - title: "Maximum likelihood from incomplete data via the EM algorithm"
    author: "A. P. Dempster, N. M. Laird, D. B. Rubin"
    url: "https://doi.org/10.1111/j.2517-6161.1977.tb01600.x"
    type: paper


status: draft
---

Mixture models say the data come from a few hidden groups, each with its own simple distribution, and we don't know who belongs where. The EM algorithm fits such a model by guessing soft group memberships, refitting each group, and repeating. I implemented it in a lab for binary vectors (a mixture of multivariate Bernoullis), and the part I got wrong is a good lesson: I was monitoring the wrong log-likelihood.

## The model

Each observation is a binary vector $x_n = (x_{n1}, \dots, x_{nD})$, $x_{nd} \in \{0, 1\}$. There are $K$ components. Component $k$ is chosen with probability $\pi_k$ (the mixing weights, summing to 1), and inside component $k$ each bit is an independent coin flip with probability $\mu_{kd}$ of being 1:

$$
p(x_n \mid \mu_k) = \prod_{d=1}^{D} \mu_{kd}^{\,x_{nd}} (1 - \mu_{kd})^{1 - x_{nd}}, \qquad p(x_n) = \sum_{k=1}^{K} \pi_k \, p(x_n \mid \mu_k).
$$

We want to maximize the **observed-data log-likelihood**

$$
\ell(\pi, \mu) = \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_k \, p(x_n \mid \mu_k).
$$

The log of a sum doesn't split nicely, so there's no closed form. EM gets around it.

## E-step and M-step

**E-step.** With the current parameters, compute the responsibility of component $k$ for point $n$, which is the posterior probability that $x_n$ came from $k$:

$$
\gamma_{nk} = \frac{\pi_k \, p(x_n \mid \mu_k)}{\sum_{j} \pi_j \, p(x_n \mid \mu_j)} .
$$

**M-step.** Refit each component as if the responsibilities were known weights:

$$
\pi_k = \frac{1}{N}\sum_n \gamma_{nk}, \qquad \mu_{kd} = \frac{\sum_n \gamma_{nk}\, x_{nd}}{\sum_n \gamma_{nk}} .
$$

So $\mu_k$ is just a weighted average of the data vectors. Repeat until $\ell$ stops changing. Each iteration is guaranteed not to decrease $\ell$, which makes it a good convergence check and a good bug detector.

## The mistake: monitoring $Q$ instead of $\ell$

In my lab code, the "log-likelihood" I printed was

$$
Q = \sum_{n}\sum_{k} \gamma_{nk} \left[\log \pi_k + \log p(x_n \mid \mu_k)\right].
$$

That's the **expected complete-data log-likelihood**, the thing the M-step maximizes. It isn't $\ell$. The two are linked by

$$
\ell = Q + H(\gamma), \qquad H(\gamma) = -\sum_n \sum_k \gamma_{nk} \log \gamma_{nk} \;\ge 0,
$$

where $H$ is the total entropy of the soft assignments. So $Q$ is always below $\ell$, and the gap is bigger when memberships are more uncertain.

This explained something odd in my results. My final values were about $-6497$ for $K=2$, $-6743$ for $K=3$ and $-6874$ for $K=4$: getting *worse* as $K$ grew. For the true log-likelihood that shouldn't happen. A model with more components can always copy a smaller one, so at a good optimum $\ell$ goes up (or stays the same) with $K$. What I saw was the entropy term: with more components, points are shared more ambiguously, $H(\gamma)$ grows, and $Q$ falls.

Re-running a corrected version on data generated the same way shows exactly that pattern: $\ell$ goes from about $-6353$ ($K=2$) to $-6322$ ($K=3$) to $-6312$ ($K=4$), while $Q$ goes down from $-6490$ to $-6699$ to $-6759$. (These come from my re-run, so they don't match the report's numbers exactly.)

Note also that a higher $\ell$ for larger $K$ doesn't mean larger $K$ is better. To choose $K$, use a penalized criterion like BIC, or held-out log-likelihood.

```r
em_bernoulli <- function(x, K, iters = 200, tol = 1e-6) {
  pi <- rep(1 / K, K)
  mu <- matrix(runif(K * ncol(x), 0.4, 0.6), K)                # K x D, random start
  ll_old <- -Inf
  for (it in 1:iters) {
    logp <- x %*% t(log(mu)) + (1 - x) %*% t(log(1 - mu))       # log p(x_n | mu_k), N x K
    a <- sweep(logp, 2, log(pi), "+")                           # log(pi_k p(x_n | mu_k))
    m <- apply(a, 1, max)
    log_px <- m + log(rowSums(exp(a - m)))                      # log-sum-exp: log p(x_n)
    ll <- sum(log_px)                                           # the observed-data log-likelihood
    gamma <- exp(a - log_px)                                    # E-step
    if (ll - ll_old < tol) break
    ll_old <- ll
    pi <- colMeans(gamma)                                       # M-step
    mu <- pmin(pmax(t(gamma) %*% x / colSums(gamma), 1e-6), 1 - 1e-6)
  }
  list(pi = pi, mu = mu, loglik = ll, Q = sum(gamma * a), iters = it)
}
```

Two practical details in there. Products of many probabilities underflow, so it's safer to work with logs and use the log-sum-exp trick. And clipping $\mu$ away from exactly 0 and 1 avoids $\log 0$ when a component ends up owning only points with the same bit.

## What the lab showed about K

With the true $K = 3$, EM recovered the three true $\mu$ profiles quite well. With $K = 4$, one of the components (the one with all $\mu_{kd} = 0.5$) got split into two. That flat component is hard to pin down: its bit probabilities are exactly the average of the other two components' (they're mirror images around 0.5), so on average it looks like a 50/50 blend of them. Only the correlations between bits tell them apart, and EM converges slowly while it sorts that out. EM also only finds a local optimum, so it's worth running it from several random starts and keeping the best $\ell$.

## Key takeaways

- EM alternates soft assignments (E-step) and weighted refits (M-step); for Bernoulli mixtures, $\mu_k$ is a responsibility-weighted mean.
- Monitor the observed-data log-likelihood $\ell = \sum_n \log \sum_k \pi_k p(x_n \mid \mu_k)$; it must never decrease.
- $Q = \sum \gamma \log(\pi p)$ is not $\ell$: $\ell = Q + $ entropy, so $Q$ can fall as $K$ grows even when $\ell$ rises.
- Choose $K$ with BIC or held-out likelihood, and use several random starts.

## Resources

- 732A99 Machine Learning, Linköping University (course lab, autumn 2019)
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 9.3 (an alternative view of EM) and Section 9.3.3 (mixtures of Bernoulli distributions), Section 9.4 (the decomposition of $\ell$ into a lower bound and a KL term)
- A. P. Dempster, N. M. Laird, D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm", *Journal of the Royal Statistical Society B* 39(1) (1977): https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
