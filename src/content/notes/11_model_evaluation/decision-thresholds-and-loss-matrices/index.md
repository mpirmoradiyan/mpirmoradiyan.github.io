---
title: "Decision thresholds and loss matrices: when 0.5 is the wrong cut-off"
slug: decision-thresholds-and-loss-matrices
date: 2019-12-08
topic: Model Evaluation
tags: [university-lab, classification, decision-theory, loss-matrix, roc-curve, thresholds]
summary: "How a classifier's probabilities turn into decisions, why the right threshold comes from the costs of the two kinds of mistakes, and how a loss matrix gives you that threshold directly."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Pattern Recognition and Machine Learning, Section 1.5 (Decision Theory)"
    author: "Christopher M. Bishop"
    url: "https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/"
    type: book
  - title: "The Elements of Statistical Learning, Sections 2.4 and 9.2.5"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book


status: draft
---

Most classifiers don't really output a class. They output a probability, and then *we* pick a rule that turns it into a decision. The default rule, "predict class 1 if $P(y=1 \mid x) > 0.5$", quietly assumes both kinds of mistake cost the same. In spam filtering or credit scoring they clearly don't, and that's where the threshold, or a loss matrix, comes in.

## Two labs, same idea

In two of my first lab assignments I ran into this from two directions.

- With a logistic regression spam filter, I moved the threshold from 0.5 to 0.8. Overall accuracy got worse, but the number of real emails wrongly sent to spam dropped a lot (on the test half, from 143 to 20). Losing a real email is worse than seeing one more spam, so that trade can be the right one.
- With a Naive Bayes credit-scoring model, I used a loss matrix that said "calling a bad customer good costs 10 times more than calling a good customer bad". Again the overall error went up, but the expensive mistake became much rarer.

These are the same idea. A loss matrix is just a principled way to pick the threshold.

## The Bayes decision rule

Write the costs in a matrix $L$, where $L_{jk}$ is the loss when the true class is $j$ and we predict $k$. Correct decisions usually cost 0. For a new input $x$ the model gives probabilities $P(j \mid x)$. The expected loss of predicting class $k$ is

$$
R(k \mid x) = \sum_j L_{jk} \, P(j \mid x),
$$

and the best decision is the class with the smallest expected loss. Nothing more.

For two classes, call them positive and negative, let $c_{FP}$ be the cost of a false positive and $c_{FN}$ the cost of a false negative. Predicting positive costs $c_{FP}\,(1-p)$ on average and predicting negative costs $c_{FN}\,p$, where $p = P(\text{positive} \mid x)$. So we predict positive when

$$
p > \frac{c_{FP}}{c_{FP} + c_{FN}}.
$$

With equal costs this gives 0.5. Choosing 0.8 in the spam case means I implicitly said a false "spam" label costs four times as much as a missed spam ($0.8 = 4/(4+1)$). Seeing the threshold as a cost ratio made it click for me. It's not a magic number, it's a statement about what mistakes hurt.

For the credit case, with "good" as the positive class, a false "good" costs 10 and a false "bad" costs 1. So we predict good only when

$$
P(\text{good} \mid x) > 10 \, P(\text{bad} \mid x) \quad\Longleftrightarrow\quad P(\text{good} \mid x) > \tfrac{10}{11} \approx 0.91.
$$

That's exactly what my code did. In the report I described it as "penalizing the probability of being good by a factor of 10". That's a fuzzy way to say it. The probabilities aren't changed at all. What changes is how they're weighed against the costs, which moves the decision threshold up to $10/11$.

```r
# Loss matrix: rows = true class, columns = predicted class
L <- matrix(c(0, 1,     # true good: predict good costs 0, predict bad costs 1
              10, 0),   # true bad:  predict good costs 10, predict bad costs 0
            nrow = 2, byrow = TRUE,
            dimnames = list(true = c("good", "bad"), pred = c("good", "bad")))

p_good <- c(0.95, 0.92, 0.88, 0.60)          # model's P(good | x) for four applicants
p <- cbind(good = p_good, bad = 1 - p_good)
expected_loss <- p %*% L                     # expected loss of each possible decision
decision <- colnames(L)[apply(expected_loss, 1, which.min)]
data.frame(p_good, round(expected_loss, 2), decision)
# 0.95 and 0.92 -> "good"; 0.88 and 0.60 -> "bad" (the cut-off is 10/11)
```

## Where the ROC curve fits

If you don't know the costs yet, you can look at all thresholds at once. The ROC curve plots the true positive rate against the false positive rate as the threshold moves from 1 down to 0. Each point on the curve is one possible decision rule. Comparing two models by their ROC curves (or the area under them) tells you which one ranks cases better overall. Picking the *point* on the curve is still a cost question.

A small trap I hit while building ROC curves by hand: first decide which class is "positive", then compute TPR and FPR consistently for that class. If the threshold is applied to $P(\text{good})$ but TPR is computed for "bad", the curve still comes out, but it's easy to misread. And at extreme thresholds one predicted class can vanish from the confusion table, so code that indexes the table by position can break or pick the wrong cell.

## Things to keep in mind

- The rule only works if the probabilities are reasonably calibrated. Naive Bayes, for example, often gives probabilities that are too extreme, so a cost-based threshold on top of it is only approximately optimal.
- Accuracy isn't the goal once costs are unequal. Report the confusion matrix and the total (or average) cost, not just the error rate.
- Class imbalance and unequal costs are different problems, even though both push the threshold away from 0.5.

## Key takeaways

- A classifier gives probabilities; the threshold is a separate decision.
- Pick the class with the smallest expected loss, $\sum_j L_{jk} P(j \mid x)$.
- For two classes the threshold is $c_{FP}/(c_{FP}+c_{FN})$, so a threshold always implies a cost ratio.
- Raising the threshold trades overall accuracy for fewer of the expensive mistakes.
- Use ROC curves to compare models across thresholds, and costs to pick the operating point.

## Resources

- 732A99 Machine Learning, Linköping University (course labs, autumn 2019)
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 1.5 (decision theory and the loss matrix)
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 2.4 (statistical decision theory) and Section 9.2.5 (loss matrices for trees): https://hastie.su.domains/ElemStatLearn/
- Tom Fawcett, "An introduction to ROC analysis", *Pattern Recognition Letters* 27 (2006): https://doi.org/10.1016/j.patrec.2005.10.010
