---
title: "MSE, cross-entropy and focal loss: choosing a loss for regression and classification"
slug: classification-and-regression-losses
date: 2024-11-28
topic: Deep Learning Foundations
tags: [loss-functions, cross-entropy, mse, focal-loss, class-imbalance]
summary: "What mean squared error, binary and categorical cross-entropy, and focal loss each measure, when to use which, and how focal loss down-weights easy examples to cope with class imbalance."
sources:
  - title: "Focal Loss for Dense Object Detection"
    author: "Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár"
    url: "https://arxiv.org/abs/1708.02002"
    type: paper
  - title: "Deep Learning, Chapter 6.2: Gradient-Based Learning (cost functions and output units)"
    author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville"
    url: "https://www.deeplearningbook.org/contents/mlp.html"
    type: book
  - title: "torch.nn loss functions"
    author: "PyTorch"
    url: "https://pytorch.org/docs/stable/nn.html#loss-functions"
    type: docs


status: draft
---

A loss function turns "how wrong is the model?" into one number that gradient descent can push down. The choice is not cosmetic. It decides which mistakes the model cares about, how strong the gradients are, and how the model treats rare classes. This note covers the three losses I use most often: mean squared error for regression, cross-entropy for classification, and focal loss for heavily imbalanced classification.

## Mean squared error (regression)

For $n$ targets $y_i$ and predictions $\hat y_i$,

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2 .
$$

Squaring makes all errors positive and punishes large errors much more than small ones. Its gradient with respect to a prediction is $\frac{2}{n}(\hat y_i - y_i)$, so the push is proportional to the error. MSE is also what you get from maximum likelihood if you assume the target is the prediction plus Gaussian noise.

The flip side is sensitivity to outliers: one point that is 10 units off counts as much as a hundred points that are 1 unit off. Mean absolute error or the Huber loss (MSE near zero, absolute error further out) are the usual fixes.

MSE is a poor choice for classification with a sigmoid or softmax output. When the output saturates near the wrong answer, the sigmoid's derivative is close to zero, and the MSE gradient is multiplied by it. The model then learns slowest exactly when it is most wrong.

## Cross-entropy (classification)

Cross-entropy compares a predicted probability distribution with the true one. It is the negative log-likelihood of the correct label under the model.

**Binary** (one probability $p$ for the positive class, label $y \in \{0,1\}$):

$$
\text{BCE} = -\big[\,y\log p + (1-y)\log(1-p)\,\big].
$$

**Categorical** ($C$ classes, one-hot label $y_c$, softmax probabilities $p_c$):

$$
\text{CE} = -\sum_{c=1}^{C} y_c \log p_c = -\log p_{\text{true}} .
$$

Only the probability of the true class matters. A confident correct answer ($p_{\text{true}} \to 1$) costs almost nothing. A confident wrong answer ($p_{\text{true}} \to 0$) costs without limit. Combined with softmax, the gradient with respect to the logits is simply $p - y$. That is why cross-entropy trains classifiers well: the gradient does not vanish when the model is badly wrong.

Pick the variant that matches the labels:

- Exactly one class per example: softmax plus categorical cross-entropy.
- Several labels can be on at once (multi-label): one sigmoid per label plus binary cross-entropy for each.

My original notebook applied the categorical formula to multi-label targets such as `[1, 0, 1]`. That silently ignores the zeros, so the model is never penalized for false positives. Binary cross-entropy per label is the correct choice there.

In practice, compute the loss from logits, not probabilities. `torch.nn.CrossEntropyLoss` and `BCEWithLogitsLoss` use the log-sum-exp trick, which is numerically stable. Clipping probabilities with a small epsilon before the log is the manual fallback.

```python
import numpy as np

def cross_entropy_from_logits(y_onehot, logits):
    z = logits - logits.max(axis=1, keepdims=True)          # stability shift
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True)) # log-softmax
    return -np.mean(np.sum(y_onehot * log_p, axis=1))

def binary_cross_entropy(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)                             # avoid log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print(cross_entropy_from_logits(np.array([[0, 1, 0]]), np.array([[1.0, 3.0, 0.5]])))  # 0.197
```

## Focal loss (imbalanced classification)

In dense object detection, a single image yields tens of thousands of candidate boxes, and almost all of them are easy background. Each easy example has a small cross-entropy, but there are so many that together they dominate the total loss and the gradient. The few hard, informative examples get drowned out.

Focal loss (Lin et al., 2017) fixes this by scaling cross-entropy down for examples the model already gets right. Define $p_t$ as the probability the model gives to the true class: $p_t = p$ if $y = 1$ and $p_t = 1 - p$ if $y = 0$. Then

$$
\text{FL}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t .
$$

- $-\log p_t$ is ordinary binary cross-entropy.
- $(1-p_t)^\gamma$ is the *modulating factor*. The focusing parameter $\gamma \ge 0$ controls how fast easy examples fade out. With $\gamma = 0$ it is plain cross-entropy.
- $\alpha_t$ balances the classes: $\alpha_t = \alpha$ for positives and $1-\alpha$ for negatives.

With $\gamma = 2$, how much each example is scaled down:

| $p_t$ (confidence in the true class) | Cross-entropy $-\log p_t$ | Factor $(1-p_t)^2$ |
|---|---|---|
| 0.9 (easy) | 0.105 | 0.01 (100× smaller) |
| 0.5 (unsure) | 0.693 | 0.25 |
| 0.1 (hard, wrong) | 2.303 | 0.81 (almost unchanged) |

So raising $\gamma$ does not mainly "punish mistakes more". It mostly *silences easy examples*, which makes the hard ones a larger share of the gradient.

A common intuition is to set $\alpha$ to the inverse class frequency, so the rare class gets the larger weight. That is right for α-balanced cross-entropy alone. With the focal term, the paper found $\gamma = 2$ and $\alpha = 0.25$ for the rare foreground class worked best. That is a *smaller* weight on the rare class, because the modulating factor already removes most of the easy negatives. Treat $\alpha$ and $\gamma$ as hyperparameters to tune together.

```python
def focal_loss(y, p, alpha=0.25, gamma=2.0, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    p_t = np.where(y == 1, p, 1 - p)              # prob. of the true class
    alpha_t = np.where(y == 1, alpha, 1 - alpha)  # class balance weight
    return np.mean(-alpha_t * (1 - p_t) ** gamma * np.log(p_t))

y = np.array([[1, 0, 1], [0, 1, 0]])
p = np.array([[0.8, 0.1, 0.6], [0.2, 0.7, 0.3]])
print(binary_cross_entropy(y, p), focal_loss(y, p))   # 0.296, 0.0104
```

The focal value is much smaller here because most predictions are already on the right side. Its absolute size is not comparable to BCE; what matters is how the gradient is shared between easy and hard examples. In PyTorch, `torchvision.ops.sigmoid_focal_loss` implements it from logits.

## Key takeaways

- MSE suits regression with roughly Gaussian noise; it is sensitive to outliers and a poor fit for sigmoid or softmax outputs.
- Cross-entropy is the negative log-likelihood of the true label. With softmax its gradient is $p - y$, so it keeps learning when the model is badly wrong.
- Use categorical cross-entropy for single-label problems and per-label binary cross-entropy for multi-label ones, computed from logits.
- Focal loss multiplies cross-entropy by $(1-p_t)^\gamma$ to silence easy examples; $\alpha_t$ balances the classes, and the two need joint tuning.

## Resources

- Tsung-Yi Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017): https://arxiv.org/abs/1708.02002
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, Section 6.2 "Gradient-Based Learning" (cost functions and output units): https://www.deeplearningbook.org/contents/mlp.html
- PyTorch docs, loss functions (`CrossEntropyLoss`, `BCEWithLogitsLoss`, `MSELoss`, `HuberLoss`): https://pytorch.org/docs/stable/nn.html#loss-functions
- torchvision docs, `sigmoid_focal_loss`: https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html
