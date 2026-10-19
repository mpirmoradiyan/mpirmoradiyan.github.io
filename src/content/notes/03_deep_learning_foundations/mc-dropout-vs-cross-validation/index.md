---
title: "Monte Carlo dropout vs. cross-validation: two different kinds of uncertainty"
slug: mc-dropout-vs-cross-validation
date: 2020-05-11
topic: Deep Learning Foundations
tags: [university-lab, dropout, uncertainty, cross-validation, bayesian-deep-learning]
summary: "What Monte Carlo dropout actually estimates (per-prediction model uncertainty), what the spread of cross-validation scores estimates (how much the test score would change with other data splits), and why the two aren't interchangeable."
sources:
  - title: "732A78 Deep Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
    author: "Yarin Gal, Zoubin Ghahramani"
    url: "http://proceedings.mlr.press/v48/gal16.pdf"
    type: paper
  - title: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
    author: "Nitish Srivastava et al."
    url: "http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf"
    type: paper
  - title: "The Elements of Statistical Learning, Section 7.10 (Cross-Validation)"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book


status: draft
---

In the deep learning course we got two "uncertainty" numbers for the same network on the same data. One came from leaving dropout on at test time and running the test set 100 times. The other came from 10-fold cross-validation. The lab asked which is better, and our answer was "dropout, because it's cheaper". That's true, but it skips the main point: the two numbers measure different things, so cost isn't the only difference.

## Dropout, quickly

During training, dropout sets each hidden unit to zero with probability $p$ (we used $p = 0.5$) and scales the survivors by $1/(1-p)$, so each mini-batch trains a slightly different thinned network. Normally dropout is switched off at test time, and you get one deterministic prediction.

## Monte Carlo dropout

Gal and Ghahramani showed that keeping dropout **on** at test time and averaging many forward passes can be read as approximate Bayesian inference over the network weights. For one input $x$, run $T$ passes, each with a fresh random dropout mask $m_t$:

$$
\hat{p}(y \mid x) = \frac{1}{T}\sum_{t=1}^{T} f(x; W \odot m_t)
$$

Here $f(x; W \odot m_t)$ is the network output (for example the predicted class probabilities) with weights $W$ and mask $m_t$ applied, and $\odot$ is element-wise multiplication. The average is the prediction. The spread of the $T$ outputs, for example their standard deviation, is the **model uncertainty for that input**. A sample the network has seen lots of similar examples of gives nearly the same output every pass. An unusual sample gives outputs that jump around.

So the useful output of MC dropout is a number *per test example*. You can use it to flag inputs the model is unsure about.

## What we actually computed

We evaluated the whole test set 100 times with dropout on and took the mean and standard deviation of the 100 **accuracies**: mean 0.9032, standard deviation 0.00047. That tiny spread isn't surprising. Each accuracy averages over 114,621 test rows, so the random masks mostly cancel out. The number answers "how much does test accuracy wobble because of dropout noise?", which isn't very interesting. The interesting information was per-row, and it got averaged away.

## What the cross-validation spread means

With 10-fold cross-validation we trained 10 separate networks, each on 90% of the data, and tested each on the 10% left out. We got mean accuracy 0.9065 and standard deviation 0.0021 over the folds.

This spread measures something else: how much the **score** changes when the training set and test set change. It covers the luck of the split, retraining from a new random start, and the fact that each fold's test set is a different sample. It says how far to trust a single reported test accuracy. It says nothing about which individual predictions are shaky.

| | MC dropout | Cross-validation |
|---|---|---|
| What varies | Dropout masks on one trained model | The training/test split (and the training run) |
| Natural output | Spread of predictions **per input** | Spread of the **score** across folds |
| Answers | "How sure is the model about this sample?" | "How stable is my performance estimate?" |
| Cost | $T$ forward passes | $K$ full training runs |

So the honest answer to "what's the main advantage of dropout?" is: it gives uncertainty for each prediction from one trained model, cheaply. Cross-validation can't do that, and MC dropout can't tell you how stable your test score is.

## Doing it in Keras

The lab redefined the `Dropout` layer to stay active at test time. In current `tf.keras` you don't need that. Calling the model with `training=True` keeps dropout on:

```python
import numpy as np

def mc_dropout_predict(model, x, T=100):
    """Mean and std of predicted probabilities over T dropout passes."""
    probs = np.stack([model(x, training=True).numpy() for _ in range(T)])
    return probs.mean(axis=0), probs.std(axis=0)

# mean_p, std_p = mc_dropout_predict(model, x_test)
# y_hat = (mean_p > 0.5).astype(int)       # binary case
# unsure = np.argsort(std_p.ravel())[-20:]   # the 20 most uncertain test rows
```

One catch: `training=True` also puts batch normalization into training mode, so it uses the statistics of the current batch instead of the running averages. Our network had both layers. If you mix them, either keep BN in inference mode (a custom dropout layer that ignores the `training` flag does that) or predict with big batches.

## Key takeaways

- MC dropout keeps dropout on at test time and averages many passes. The spread across passes is a per-input measure of model uncertainty.
- Averaging that spread into one accuracy standard deviation throws the useful information away.
- The standard deviation across cross-validation folds measures how stable the performance estimate is, not how sure individual predictions are.
- They answer different questions. Cost is a side issue.

## Resources

- 732A78 Deep Learning, Linköping University (DNN lab, done in a pair).
- Gal and Ghahramani, [Dropout as a Bayesian Approximation](http://proceedings.mlr.press/v48/gal16.pdf), ICML 2016.
- Srivastava et al., [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf), JMLR 2014.
- Hastie, Tibshirani and Friedman, [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/), Section 7.10.
