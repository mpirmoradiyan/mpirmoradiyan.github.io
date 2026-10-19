---
title: "The C parameter in soft-margin SVMs, and how to choose it honestly"
slug: svm-cost-parameter-c
date: 2019-12-14
topic: ML & Statistics
tags: [university-lab, svm, regularization, model-selection, hyperparameters, train-validation-test]
summary: "What the cost parameter C trades off in a soft-margin SVM, and why choosing it on the test set and then reporting training error gives a number you can't trust."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Support-vector networks"
    author: "Corinna Cortes, Vladimir Vapnik"
    url: "https://doi.org/10.1007/BF00994018"
    type: paper
  - title: "The Elements of Statistical Learning, Sections 7.2 and 12.2–12.3"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book
  - title: "Pattern Recognition and Machine Learning, Section 7.1.1 (Overlapping class distributions)"
    author: "Christopher M. Bishop"
    type: book


status: draft
---

A support vector machine looks for the boundary with the widest gap between classes. Real data overlap, so a perfect gap usually doesn't exist, and the SVM needs a knob that says how much to care about points on the wrong side. That knob is $C$. I tuned it in a spam-classification lab, and in hindsight the way I *chose* it taught me more than the parameter itself.

## The soft-margin problem

For training points $x_i$ with labels $y_i \in \{-1, +1\}$, the soft-margin SVM solves

$$
\min_{w,\,b,\,\xi}\; \frac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^{n} \xi_i \quad\text{subject to}\quad y_i\,(w^\top \phi(x_i) + b) \ge 1 - \xi_i,\;\; \xi_i \ge 0 .
$$

- $w, b$ define the boundary $w^\top \phi(x) + b = 0$; $\phi$ is the feature map behind the kernel (an RBF kernel in my lab).
- The margin width is $2/\lVert w \rVert$, so a small $\lVert w \rVert$ means a wide margin.
- $\xi_i$ is the **slack** of point $i$: 0 if it's on the right side outside the margin, between 0 and 1 if it's inside the margin, above 1 if it's misclassified.
- $C > 0$ is the price of each unit of slack.

So $C$ trades a wide margin (first term) against margin violations (second term). Equivalently, it's an inverse regularization strength: think of $\frac{1}{2}\lVert w\rVert^2$ as the penalty and $\sum \xi_i$ as the loss (the hinge loss), and $C$ plays the role of $1/\lambda$.

- **Small $C$**: violations are cheap. Wide margin, many support vectors, smoother boundary, more bias.
- **Large $C$**: violations are expensive. The SVM bends the boundary to classify training points correctly. Narrow margin, fewer support vectors, more variance.

In the dual problem, $C$ shows up as a box constraint on the Lagrange multipliers, $0 \le \alpha_i \le C$. A point with $\alpha_i > 0$ is a support vector, and $C$ caps how much any single point can pull on the boundary.

## What I got wrong in the description

In the report I wrote that increasing $C$ lets us "penalize the weights" to reduce training error. It's the other way round. Increasing $C$ penalizes the **slacks** more, so relative to that the penalty on $\lVert w\rVert$ becomes weaker. The weights are allowed to grow, the margin shrinks, and training error goes down. That's why training error kept falling as $C$ grew while test error didn't follow.

## Choosing $C$ honestly

Here's what I did in the lab: tried $C$ from 0.05 up to about 100, picked the value with the lowest **test** error ($C = 2.55$), refit on all data, and reported the **training** error of that final model (2.87%) as "a good performance". Two problems:

1. **The test set was used for model selection.** Once you pick $C$ because it did best on the test set, that test error is optimistically biased. You've tuned to it.
2. **Training error on the data the model was fit on** is also optimistic, especially for a flexible RBF SVM, which can fit training data very closely.

So 2.87% isn't an estimate of how the model will do on new emails. The fix is the usual three-way discipline: choose $C$ on a **validation** set (or by cross-validation on the training part), refit on train + validation, and look at the **test** set exactly once at the end. (If the rows come in groups, the split has to respect that too; see [grouped cross-validation](/notes/grouped-cross-validation/).)

```r
library(e1071)
set.seed(3)
n <- 600
X <- matrix(rnorm(2 * n), ncol = 2)
y <- factor(ifelse(rowSums(X^2) + rnorm(n, sd = 0.7) > 1.4, "out", "in"))
d <- data.frame(X, y)
split <- sample(rep(c("train", "valid", "test"), c(300, 150, 150)))
tr <- d[split == "train", ]; va <- d[split == "valid", ]; te <- d[split == "test", ]

err <- function(model, data) mean(predict(model, data) != data$y)
Cs  <- c(0.01, 0.1, 1, 10, 100, 1000)
res <- t(sapply(Cs, function(C) {
  m <- svm(y ~ ., data = tr, kernel = "radial", gamma = 2, cost = C)
  c(C = C, train = err(m, tr), valid = err(m, va), n_sv = m$tot.nSV)
}))
round(res, 3)          # train error keeps falling with C; validation error is lowest at C = 1

best  <- res[which.min(res[, "valid"]), "C"]
final <- svm(y ~ ., data = rbind(tr, va), kernel = "radial", gamma = 2, cost = best)
err(final, te)         # about 0.15: the one number to report
```

In this run the training error falls from 0.47 to 0.10 as $C$ goes from 0.01 to 1000, the number of support vectors drops from 285 to 119, and the validation error is lowest at $C = 1$ (0.16) and climbs again after that.

Two more things worth remembering. $C$ and the kernel width (`sigma` in kernlab, `gamma` in e1071) interact, so tune them together on a grid. And scale the features first; the RBF kernel uses Euclidean distance, so a feature measured in large units would dominate.

## Key takeaways

- $C$ is the cost of margin violations; large $C$ means narrow margin and low training error, small $C$ means wide margin and more bias.
- Large $C$ penalizes slack, not weights: it's weak regularization, not strong.
- In the dual, $C$ is the upper bound on each $\alpha_i$.
- Pick $C$ on validation data or by cross-validation, never on the test set, and never report training error as performance.

## Resources

- 732A99 Machine Learning, Linköping University (course lab, autumn 2019)
- Corinna Cortes and Vladimir Vapnik, "Support-vector networks", *Machine Learning* 20 (1995): https://doi.org/10.1007/BF00994018
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 7.2 (train/validation/test) and Sections 12.2–12.3 (support vector classifiers and machines): https://hastie.su.domains/ElemStatLearn/
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 7.1.1
- R package `e1071` documentation (`svm`): https://cran.r-project.org/package=e1071
