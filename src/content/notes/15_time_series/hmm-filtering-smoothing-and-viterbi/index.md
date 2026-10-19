---
title: "Hidden Markov models: filtering, smoothing, Viterbi, and one-step prediction"
slug: hmm-filtering-smoothing-and-viterbi
date: 2020-09-18
topic: Time Series & State Space
tags: [university-lab, hidden-markov-models, forward-backward, viterbi, state-estimation]
summary: "What the forward-backward algorithm computes, why smoothed estimates beat filtered ones, why the most probable path is not the same as the most probable state at each time, and how to predict the next hidden state."
sources:
  - title: "732A96 Advanced Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Pattern Recognition and Machine Learning, Section 13.2 (hidden Markov models)"
    author: "Christopher M. Bishop"
    url: "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/"
    type: book
  - title: "A tutorial on hidden Markov models and selected applications in speech recognition"
    author: "Lawrence R. Rabiner"
    url: "https://doi.org/10.1109/5.18626"
    type: paper


status: draft
---

A hidden Markov model (HMM) is the simplest model for "something moves around, and I only see noisy readings of where it is". The second Advanced Machine Learning lab used a robot walking around a ring of 10 sectors with an inaccurate tracker. The modelling was easy. The questions were about *what kind* of estimate you want, and the three standard answers (filtering, smoothing, Viterbi) are easy to mix up.

## The model

An HMM has hidden states $z_t$ and observations $x_t$, for $t = 1, \dots, T$:

- **Initial distribution** $p(z_1)$.
- **Transition model** $A_{ij} = p(z_t = j \mid z_{t-1} = i)$: the hidden state is a Markov chain.
- **Emission model** $B_{ik} = p(x_t = k \mid z_t = i)$: each observation depends only on the current state.

For the ring robot: it stays or moves to the next sector with probability 0.5 each, and the tracker reports a sector uniformly within $\pm 2$ of the true one. Since we have no idea where the robot starts, $p(z_1)$ should be uniform over the 10 sectors. (In my lab code I used `runif(10)` for the start probabilities, random unnormalised numbers. That happened not to matter much, but a uniform start is what the problem actually says.)

## Filtering and smoothing

Two posterior distributions over the hidden state at time $t$:

- **Filtering**: $p(z_t \mid x_{1:t})$, using observations up to now. This is what you'd have in real time.
- **Smoothing**: $p(z_t \mid x_{1:T})$, using all observations, including later ones. This is what you'd use offline.

The forward-backward algorithm computes both with two passes:

$$
\alpha_t(j) = p(x_{1:t}, z_t = j) = B_{j, x_t} \sum_i \alpha_{t-1}(i)\, A_{ij},
$$

$$
\beta_t(i) = p(x_{t+1:T} \mid z_t = i) = \sum_j A_{ij}\, B_{j, x_{t+1}}\, \beta_{t+1}(j), \qquad \beta_T(i) = 1.
$$

Then filtering is $\alpha_t$ normalised over states, and smoothing is $\alpha_t(i)\,\beta_t(i)$ normalised. Each forward step is "predict with the transition model, then update with the new observation". In practice you normalise $\alpha_t$ at every step, otherwise it underflows for long sequences.

Smoothing is generally more accurate because it uses more evidence: a later reading of sector 7 rules out the robot being at sector 2 a step earlier. In my runs, smoothing got 59%, 71% and 69% of states right, while filtering got 43%, 51% and 65%.

## Viterbi: the most probable path

The Viterbi algorithm finds the single most probable *sequence*:

$$
\hat z_{1:T} = \arg\max_{z_{1:T}} p(z_{1:T} \mid x_{1:T}).
$$

That's a different question from picking the most probable state at each time, $\hat z_t = \arg\max_{z_t} p(z_t \mid x_{1:T})$. The per-time choice maximises the expected number of correct states, so it should score best on a "fraction of states right" metric. Viterbi instead has to return a path that hangs together: in the ring model it can never jump backwards or skip a sector, because those transitions have probability zero. Sometimes that forces it to commit to a whole stretch of positions that's jointly plausible but individually less likely. That's why, in my runs, Viterbi was the least accurate of the three (33%, 52%, 55.5%).

The per-time smoothed choices, on the other hand, don't have to form a valid path at all. If you need a trajectory the robot could actually have taken, use Viterbi; if you want the best guess at each moment, use smoothing.

Our group's written explanation said that Viterbi maximises a joint probability, which is "inherently less probable than marginal probabilities". That mixes up two things. The size of a probability isn't the point; the point is what's being maximised: the whole path (Viterbi) versus each time step separately (smoothing), and only the second one targets per-step accuracy.

## Does more data help?

I compared runs of 100 and 200 steps and saw no clear gain in accuracy, and the entropy of the filtered distribution didn't shrink over time either. My explanation was "the current state only depends on the previous one". The better way to say it: the robot keeps moving, and every step adds new uncertainty through the transition model. Old observations tell you about old positions; their information about the current position fades after a few steps. So the filtered uncertainty settles at a steady level instead of going to zero. More observations would help if the hidden state were fixed (like estimating a constant parameter), not when it keeps changing.

## Predicting the next step

To predict $z_{T+1}$, push the last filtered distribution through the transition model:

$$
p(z_{T+1} = j \mid x_{1:T}) = \sum_i p(z_T = i \mid x_{1:T})\, A_{ij}.
$$

It's the "predict" half of a forward step without the update. Note it uses the *filtered* distribution at $T$ (which equals the smoothed one at the last time step).

## Code

```r
forward_backward <- function(obs, init, A, B) {
  T <- length(obs); K <- length(init)
  alpha <- beta <- matrix(0, K, T)
  a <- init * B[, obs[1]]; alpha[, 1] <- a / sum(a)
  for (t in 2:T) {
    a <- B[, obs[t]] * drop(t(A) %*% alpha[, t - 1])   # predict, then update
    alpha[, t] <- a / sum(a)                             # filtered p(z_t | x_1:t)
  }
  beta[, T] <- 1
  for (t in (T - 1):1) {
    b <- A %*% (B[, obs[t + 1]] * beta[, t + 1]); beta[, t] <- b / sum(b)
  }
  smooth <- alpha * beta
  list(filtered = alpha, smoothed = sweep(smooth, 2, colSums(smooth), "/"))
}
K <- 10; ring <- function(i) ((i - 1) %% K) + 1          # the ring robot
A <- matrix(0, K, K); B <- matrix(0, K, K)
for (i in 1:K) { A[i, c(i, ring(i + 1))] <- 0.5; B[i, ring((i - 2):(i + 2))] <- 0.2 }
set.seed(8); z <- numeric(100); z[1] <- sample(K, 1)
for (t in 2:100) z[t] <- sample(K, 1, prob = A[z[t - 1], ])
x <- sapply(z, function(s) sample(K, 1, prob = B[s, ]))
fb <- forward_backward(x, rep(1 / K, K), A, B)
c(filtered = mean(apply(fb$filtered, 2, which.max) == z),     # 0.46
  smoothed = mean(apply(fb$smoothed, 2, which.max) == z))     # 0.68
pred_next <- drop(t(A) %*% fb$filtered[, 100])                # p(z_101 | x_1:100)
```

## Key takeaways

- Filtering uses the past, smoothing uses everything; smoothing is usually more accurate.
- Viterbi maximises the probability of the whole path, not of each state, so it's usually less accurate per step but always gives a feasible trajectory.
- If the hidden state keeps moving, uncertainty about the current state levels off; more history doesn't drive it to zero.
- One-step prediction = filtered distribution times the transition matrix.
- Normalise the forward messages at each step to avoid underflow.

## Resources

- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 13.2 (forward-backward, Viterbi): https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
- Lawrence R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition", *Proceedings of the IEEE* 77(2) (1989): https://doi.org/10.1109/5.18626
- Kevin P. Murphy, *Machine Learning: A Probabilistic Perspective*, Chapter 17 (Markov and hidden Markov models), MIT Press, 2012.
- 732A96 Advanced Machine Learning, Linköping University (course lab, autumn 2020).
