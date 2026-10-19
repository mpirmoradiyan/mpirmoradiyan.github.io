---
title: "Class imbalance: baselines first, then class weights or undersampling"
slug: class-imbalance-baselines-and-weights
date: 2020-05-11
topic: Model Evaluation
tags: [university-lab, class-imbalance, baselines, class-weights, undersampling, evaluation-metrics]
summary: "How to tell when a classifier has only learned the majority class, which baselines to compare against, how balanced class weights and undersampling work, and why accuracy and macro-F1 can move in opposite directions."
sources:
  - title: "732A78 Deep Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "732A92 Text Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "scikit-learn: compute_class_weight"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html"
    type: docs
  - title: "Learning from Imbalanced Data"
    author: "Haibo He, Edwardo A. Garcia"
    url: "https://doi.org/10.1109/TKDE.2008.239"
    type: paper


status: draft
---

When one class is much bigger than the others, a model can get a high accuracy by always predicting the big class. That model is useless, but the number looks fine. I ran into this in two labs: a network-traffic classifier where 84% of the rows were "malicious", and a text classifier for Swedish parliament speeches where one party gave 30% of the speeches. Both times the first step should have been the same: work out what a model that knows nothing would score.

## Step 1: compute the baselines

Two cheap baselines tell you where "nothing learned" sits:

- **Most frequent class**: always predict the biggest class. Its accuracy is that class's share of the test set.
- **Stratified random**: guess at random, using the training class proportions. Its expected accuracy is $\sum_c p_c^2$, where $p_c$ is the share of class $c$.

In the deep learning lab, the training data had 85,221 benign and 449,674 malicious rows, so the most-frequent baseline is about 84%. In the speech lab, the baselines on the test set were 30% (always the biggest party) and 18% (stratified). scikit-learn's `DummyClassifier` gives you both.

## Step 2: recognise a model stuck at the baseline

In the DNN lab, several networks reported a training accuracy of exactly 0.8407 epoch after epoch, and a test accuracy of 0.8403. That's $449{,}674 / 534{,}895$, the majority share of the training set, to four digits. Those networks (4 dense layers with sigmoid activations, plain SGD) had learned nothing. They predicted "malicious" for every row. I didn't spot that in my report. The 2-layer sigmoid network reached 0.91 on the same data, and switching the 2-layer net to ReLU, adding batch normalization, or using Adam gave 0.92 to 0.94, so the data was learnable. The deeper sigmoid nets were just stuck: sigmoid derivatives are at most 0.25, so the gradient shrinks at every layer on the way back, and with plain SGD the early layers barely moved in 20 epochs.

So the rule I keep now: if accuracy sits right at the majority share, check the predictions before you celebrate.

## Step 3: use a metric that notices the small classes

Accuracy weighs every row the same, so the big class dominates it. **Macro-averaged F1** computes F1 for each class and takes the plain mean, so every class counts equally:

$$
F_1^{(c)} = \frac{2\, P_c R_c}{P_c + R_c}, \qquad \text{macro-}F_1 = \frac{1}{K}\sum_{c=1}^{K} F_1^{(c)}
$$

Here $P_c$ is the precision for class $c$ (of everything predicted as $c$, the share that really is $c$), $R_c$ is its recall (of everything that really is $c$, the share that was found), and $K$ is the number of classes.

## Fix A: class weights

Class weights make mistakes on rare classes cost more in the loss. scikit-learn's "balanced" weights are

$$
w_c = \frac{n}{K \, n_c}
$$

where $n$ is the number of training rows and $n_c$ the number in class $c$. With the lab's counts that's $534{,}895 / (2 \cdot 85{,}221) = 3.14$ for benign and $0.59$ for malicious, the same numbers the notebook printed.

One trap here. Keras wants `class_weight` as a dictionary `{class_index: weight}`. Our notebook passed the NumPy array straight from `compute_class_weight`. In the standalone Keras version we used, a `class_weight` that isn't a dict seems to be skipped without any warning, and our results fit that: test accuracy 0.9093 without weights and 0.9075 "with" them, with almost the same training curves. Newer TensorFlow raises an error instead. So build the dict:

```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

y_train = np.array([0] * 85221 + [1] * 449674)              # class counts from the lab
classes = np.unique(y_train)
w = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes.tolist(), w.tolist()))      # {0: 3.138..., 1: 0.594...}
# model.fit(X, y, class_weight=class_weight, ...)

# Always predicting the majority class: accuracy looks fine, macro-F1 doesn't
y_true = np.array([0] * 160 + [1] * 840)
y_pred = np.ones_like(y_true)
print(accuracy_score(y_true, y_pred))                    # 0.84
print(f1_score(y_true, y_pred, average="macro"))         # 0.46
```

## Fix B: undersampling

Undersampling throws away random rows from the big classes until every class has as many as the smallest one. In the speech lab that left 5,752 of the training speeches (8 parties of 719). A Multinomial Naive Bayes classifier trained on the full data and on the undersampled data gave:

| Training data | Accuracy | Macro-F1 |
|---|---|---|
| All speeches | 0.43 | 0.24 |
| Undersampled | 0.40 | 0.39 |

Accuracy went *down* and macro-F1 went *up*. The full-data model predicted the two biggest parties most of the time (recall 0.84 for the largest party, S, and only 0.02 to 0.04 for C, KD and L). The balanced model spread its guesses, so the small parties got found and the big one got missed more often.

Neither is "right". It depends on what you need. If every party matters equally, undersampling helps. If you care about overall hit rate on data with the same imbalance, it hurts. Undersampling also throws data away. With 8 classes and a smallest class of 719, most of the training set disappeared. Class weights keep every row, so they're usually the first thing to try.

## Key takeaways

- Always compute the most-frequent and stratified baselines first. An accuracy at the majority share means the model may have learned nothing.
- Look at per-class recall or macro-F1, not only accuracy.
- Balanced class weights are $n / (K n_c)$. In Keras, pass them as a dict.
- Undersampling trades overall accuracy for better balance across classes, and costs data. Pick based on what errors matter.

## Resources

- 732A78 Deep Learning, Linköping University (DNN lab, done in a pair).
- 732A92 Text Mining, Linköping University (text classification lab).
- [scikit-learn: compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) and [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).
- He and Garcia, [Learning from Imbalanced Data](https://doi.org/10.1109/TKDE.2008.239), IEEE TKDE, 2009.
