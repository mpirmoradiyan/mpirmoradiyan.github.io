---
title: "Posterior predictive distributions by simulation (and why plugging in a mean is wrong)"
slug: posterior-predictive-simulation
date: 2020-05-10
topic: Bayesian Methods
tags: [university-lab, bayesian-inference, posterior-predictive, simulation, logistic-regression, poisson-regression]
summary: "How to simulate the predictive distribution of a new observation by drawing parameters first and data second, and why a single plug-in probability understates the uncertainty."
sources:
  - title: "732A91 Bayesian Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Data Analysis (3rd ed.), Sections 1.3 and 6.3 (posterior predictive distribution)"
    author: "Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, Donald B. Rubin"
    url: "http://www.stat.columbia.edu/~gelman/book/"
    type: book


status: draft
---

A Bayesian prediction isn't a single number. It's a distribution over the next observation that accounts for two kinds of uncertainty: the randomness of the data, and the fact that we don't know the parameters exactly. The neat thing is that you never have to compute the integral behind it. You simulate it. I got this subtly wrong in one lab and right in the next, which is why it gets its own note. (Both labs were pair work.)

## The integral and the recipe

The posterior predictive distribution of a new observation $\tilde y$ with features $\tilde x$ is

$$
p(\tilde y \mid y) = \int p(\tilde y \mid \theta, \tilde x)\; p(\theta \mid y)\; d\theta,
$$

where $p(\theta \mid y)$ is the posterior of the parameters and $p(\tilde y \mid \theta, \tilde x)$ is the model for one observation given the parameters. In words: average the data model over everything the parameters could plausibly be.

To simulate it, repeat for $i = 1, \dots, N$:

1. Draw $\theta^{(i)}$ from the posterior (exactly, from a normal approximation, or from MCMC output).
2. Draw $\tilde y^{(i)}$ from $p(\tilde y \mid \theta^{(i)}, \tilde x)$.

The $\tilde y^{(i)}$ are draws from the predictive distribution. Each step uses its own parameter draw. That's the part that matters.

## Where I went wrong

In the logistic regression lab, the question was about a group of 10 similar people: how many of them would be in class 1? I computed the predicted probability for each posterior draw, averaged those into one number $\bar p$, and then simulated $\text{Binomial}(10, \bar p)$.

For a *single* binary outcome that happens to give the right answer, because

$$
\Pr(\tilde y = 1 \mid y) = \int p(\theta \mid y)\, \Pr(\tilde y = 1 \mid \theta)\, d\theta = E[p \mid y] = \bar p.
$$

But for a count of 10 it doesn't. The 10 people share the same unknown $\beta$. If $\beta$ is a bit higher than we think, all 10 probabilities go up together. That shared uncertainty makes the count more spread out than a binomial with a fixed probability. The right simulation draws a fresh $p^{(i)}$ for each repetition and then $k^{(i)} \sim \text{Binomial}(10, p^{(i)})$. The result is a mixture of binomials (beta-binomial-like), with fatter tails.

Two smaller mistakes from the same report: plotting the distribution of $p^{(i)}$ and calling it "the predictive distribution" (it's the posterior of the probability, not of the outcome, which only takes the values 0 and 1), and drawing a smooth density over a count variable. A count should be shown as a bar chart over $0, 1, \dots, 10$.

## Code

```r
set.seed(4)
beta_mode <- c(0.5, 1, -1.2); beta_cov <- diag(c(0.04, 0.03, 0.05))  # some posterior
x_new <- c(1, 0.3, 2)
nd <- 10000
beta <- beta_mode + t(chol(beta_cov)) %*% matrix(rnorm(3 * nd), 3)  # 3 x nd draws
p <- plogis(drop(x_new %*% beta))           # one probability per posterior draw

y_new  <- rbinom(nd, 1, p)                  # one new case: Bernoulli per draw
k_full <- rbinom(nd, 10, p)                 # 10 similar cases, uncertainty kept
k_plug <- rbinom(nd, 10, mean(p))           # plug-in: one fixed probability

c(var_full = var(k_full), var_plug = var(k_plug))   # 1.93 vs 1.48
barplot(rbind(tabulate(k_full + 1, 11), tabulate(k_plug + 1, 11)) / nd,
        beside = TRUE, names.arg = 0:10, legend.text = c("full", "plug-in"))
```

Same mean, but the full predictive has about 30% more variance here, and more mass at 0 and at the high counts. With more data the posterior narrows and the gap shrinks; with little data it can be large.

## It works for any model

In the next lab we did Poisson regression with MCMC. The predictive for the number of events in a new case is the same two-step recipe: for each MCMC draw $\beta^{(i)}$, compute $\lambda^{(i)} = \exp(\tilde x^\top \beta^{(i)})$ and draw $\tilde y^{(i)} \sim \text{Poisson}(\lambda^{(i)})$. Questions like "what's the probability of zero events?" are then just the fraction of $\tilde y^{(i)}$ equal to 0. That's the version we got right: one Poisson draw per posterior draw.

## Key takeaways

- Predictive simulation: draw parameters from the posterior, then data given those parameters, once per repetition.
- Averaging the parameter (or probability) first and then simulating throws away parameter uncertainty.
- For one binary outcome it doesn't matter; for counts, sums or anything shared across cases it does.
- Show the predictive distribution of a discrete outcome as bars over its possible values.

## Resources

- Andrew Gelman et al., *Bayesian Data Analysis*, 3rd ed., Section 1.3 (predictive distributions) and Section 6.3 (posterior predictive checking): http://www.stat.columbia.edu/~gelman/book/
- Richard McElreath, *Statistical Rethinking*, 2nd ed., Section 3.3 (sampling to simulate prediction): https://xcelab.net/rm/
- 732A91 Bayesian Learning, Linköping University (course labs, spring 2020).
