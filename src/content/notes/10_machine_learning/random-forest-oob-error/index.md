---
title: "Random forests, out-of-bag error, and how boosting differs"
slug: random-forest-oob-error
date: 2019-11-28
topic: Machine Learning
tags: [university-lab, random-forest, bagging, boosting, adaboost, out-of-bag-error]
summary: "Why averaging many de-correlated trees reduces variance, how the out-of-bag error gives a free test estimate, a small R trap when reading it, and how boosting builds trees in a completely different way."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Random Forests"
    author: "Leo Breiman"
    url: "https://doi.org/10.1023/A:1010933404324"
    type: paper
  - title: "The Elements of Statistical Learning, Chapters 10 and 15"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book
  - title: "randomForest (R package)"
    author: "Andy Liaw, Matthew Wiener"
    url: "https://cran.r-project.org/package=randomForest"
    type: docs


status: draft
---

In a lab on spam classification I compared random forests and boosting as the number of trees grew. My forest's "out-of-bag error" came out higher than the test error, and I took that as a sign the forest generalized perfectly. It wasn't. I had read the number wrong. Getting to the bottom of it taught me how forests work in the first place. The starting problem is simple: a single deep decision tree is unstable. Change a few training rows and the tree can look completely different. Random forests and boosting both fix this by combining many trees, but in opposite ways.

## Why averaging trees helps

If you average $B$ predictors, each with variance $\sigma^2$ and pairwise correlation $\rho$, the variance of the average is

$$
\rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .
$$

More trees kill the second term, but the first one stays. So the trick is to make the trees *less correlated*. A random forest does two things:

1. **Bagging**: each tree is grown on a bootstrap sample (rows drawn with replacement).
2. **Feature subsampling**: at each split, only a random subset of $m$ features is considered (a common default for classification is $m \approx \sqrt{p}$; with the 57 spam features that's 7, which is what my forest used).

Each tree is deep and low-bias but noisy; averaging (or majority vote) cuts the variance. That's why adding trees doesn't make a forest overfit more. The error just levels off. It doesn't mean a forest can't overfit at all, though: with very noisy labels, deep trees still fit the noise, and the vote inherits some of it.

## Out-of-bag error

A bootstrap sample of size $n$ leaves out each row with probability $(1 - 1/n)^n \approx e^{-1} \approx 0.37$. So every row is "out of bag" for roughly a third of the trees. Predict each row using only those trees, compare with its label, and you get the **OOB error**: an almost-free estimate of test error, without a separate validation set.

### The trap I fell into

In R's `randomForest`, `fit$err.rate` is a matrix with one row per tree. Row $b$ is the OOB error of the forest made of the *first $b$ trees*. The OOB error of the whole forest is the **last** row.

In my lab I took `mean(fit$err.rate[, 1])`, the average over forests of size 1, 2, ..., B. The early rows (forests of 1–10 trees) have high error, so this average is biased upward. That's why my "OOB error" came out above the test error (the report said OOB went from about 10% to 6% while test error stayed between 4% and 5%), and I read that as the forest generalizing "perfectly". Really, I was comparing a pessimistic, wrongly computed number with the test error. Read the last row instead:

```r
library(randomForest)
set.seed(4)
n <- 1500
X <- data.frame(matrix(rnorm(n * 10), ncol = 10))
X$y <- factor(ifelse(X$X1 + X$X2^2 - X$X3 * X$X4 + rnorm(n) > 1, "a", "b"))
tr <- X[1:1000, ]; te <- X[1001:1500, ]

fit <- randomForest(y ~ ., data = tr, ntree = 100)
oob_final <- fit$err.rate[fit$ntree, "OOB"]   # OOB error of the full 100-tree forest
oob_mean  <- mean(fit$err.rate[, "OOB"])      # average over forests of 1..100 trees: not an estimate
test_err  <- mean(predict(fit, te) != te$y)
round(c(oob_final = oob_final, oob_mean = oob_mean, test = test_err), 3)
# about 0.230, 0.256 and 0.238: the final OOB error is close to the test error
```

## Boosting builds trees in sequence

Boosting goes the other way. Instead of many independent deep trees, it adds **small** trees one after another, each one focusing on what the current ensemble gets wrong. AdaBoost reweights the training points after each round: misclassified points get more weight, so the next tree tries harder on them. The final model is a weighted vote,

$$
F(x) = \sum_{m=1}^{M} \alpha_m\, G_m(x), \qquad \hat y = \operatorname{sign} F(x),
$$

where $G_m(x) \in \{-1, +1\}$ is the $m$-th tree and $\alpha_m$ its weight. Friedman, Hastie and Tibshirani showed that this is the same as fitting an additive model step by step under the exponential loss $L(y, F) = e^{-yF}$. That's the view the R package `mboost` takes (`family = AdaExp()`), with a learning rate $\nu$ that shrinks each step.

The practical differences I saw, and that the theory predicts:

- **Few trees**: boosting is worse at the start (each small tree is weak), while a forest of even 10 trees is already decent. In my lab AdaBoost started near 9% error at 10 trees and came down toward the forest's level by 100.
- **Many trees**: a forest just levels off. Boosting keeps reducing bias and *can* start to overfit, so the number of rounds $M$ (together with $\nu$) is a hyperparameter to tune, for example with cross-validation.
- **What they reduce**: forests mainly reduce variance; boosting mainly reduces bias.

## Key takeaways

- Averaging only helps as much as the trees are de-correlated; bootstrap rows plus random feature subsets do that.
- Each row is out of bag for about 37% of trees, which gives a free test-error estimate.
- In `randomForest`, the OOB error of the forest is the last row of `err.rate`, not its mean.
- Boosting adds weak trees in sequence on reweighted data (exponential loss for AdaBoost); its number of rounds needs tuning, while a forest's number of trees mostly doesn't.

## Resources

- 732A99 Machine Learning, Linköping University (course lab, autumn 2019)
- Leo Breiman, "Random Forests", *Machine Learning* 45 (2001): https://doi.org/10.1023/A:1010933404324
- Jerome Friedman, Trevor Hastie, Robert Tibshirani, "Additive logistic regression: a statistical view of boosting", *Annals of Statistics* 28(2) (2000): https://doi.org/10.1214/aos/1016218223
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Chapter 10 (boosting) and Chapter 15 (random forests; the variance formula is eq. 15.1): https://hastie.su.domains/ElemStatLearn/
- R package `randomForest` documentation (`err.rate`): https://cran.r-project.org/package=randomForest
