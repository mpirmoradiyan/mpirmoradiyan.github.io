---
title: "MSE, cross-entropy and focal loss: choosing a loss for regression and classification"
slug: classification-and-regression-losses
date: 2024-11-28
topic: Deep Learning Foundations
tags: [loss-functions, cross-entropy, mse, focal-loss, class-imbalance]
summary: "What MSE, cross-entropy and focal loss each measure, when to pick which, and how focal loss quiets easy examples so rare classes still get attention."
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

A loss function turns "how wrong is the model?" into a single number that gradient descent can push down. Which loss you pick really matters. It decides which mistakes the model cares about, how strong the gradients are, and how rare classes get treated. Here I go through the three losses I wanted to get straight: mean squared error for regression, cross-entropy for classification, and focal loss for classification when one class is much rarer than the others.

## Mean squared error (regression)

For $n$ targets $y_i$ and predictions $\hat y_i$,

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2 .
$$

Squaring makes every error positive, and it punishes big errors a lot more than small ones. The gradient for one prediction is $\frac{2}{n}(\hat y_i - y_i)$, so the push is proportional to the error. MSE is also what falls out of maximum likelihood if you assume the target is the prediction plus Gaussian noise.

The downside is outliers. One point that's 10 units off counts as much as a hundred points that are 1 unit off. The usual fixes are mean absolute error, or the Huber loss, which acts like MSE near zero and like absolute error further out.

MSE is a bad match for classification with a sigmoid or softmax output. This one took me a moment: when the sigmoid saturates near the wrong answer, its derivative is close to zero, and the MSE gradient gets multiplied by it. So the model learns slowest exactly when it's most wrong.

## Cross-entropy (classification)

Cross-entropy compares the predicted probabilities with the true ones. Put simply, it's the negative log of the probability the model gave to the correct label (the negative log-likelihood).

**Binary** (one probability $p$ for the positive class, label $y \in \{0,1\}$):

$$
\text{BCE} = -\big[\,y\log p + (1-y)\log(1-p)\,\big].
$$

**Categorical** ($C$ classes, one-hot label $y_c$, softmax probabilities $p_c$):

$$
\text{CE} = -\sum_{c=1}^{C} y_c \log p_c = -\log p_{\text{true}} .
$$

Only the probability of the true class counts. A confident right answer ($p_{\text{true}} \to 1$) costs almost nothing. A confident wrong answer ($p_{\text{true}} \to 0$) costs without limit. With softmax, the gradient with respect to the logits (the raw scores before softmax) is just $p - y$. That's why cross-entropy trains classifiers well: the gradient doesn't die when the model is badly wrong.

Pick the version that fits your labels:

- Exactly one class per example: softmax plus categorical cross-entropy.
- Several labels can be on at once (multi-label): one sigmoid per label plus binary cross-entropy for each.

My original notebook used the categorical formula on multi-label targets like `[1, 0, 1]`. That quietly ignores the zeros, so the model never gets penalized for false positives. Binary cross-entropy per label is the right choice there.

In practice, compute the loss from logits, not probabilities. `torch.nn.CrossEntropyLoss` and `BCEWithLogitsLoss` use the log-sum-exp trick, which avoids overflow and log of zero. If you do it by hand, clip the probabilities with a small epsilon before taking the log.

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

In dense object detection, one image gives tens of thousands of candidate boxes, and nearly all of them are easy background. Each easy one has a small cross-entropy, but there are so many that together they take over the total loss and the gradient. The few hard, useful examples get drowned out.

Focal loss (Lin et al., 2017) fixes this by turning down the loss for examples the model already gets right. Let $p_t$ be the probability the model gives to the true class: $p_t = p$ if $y = 1$ and $p_t = 1 - p$ if $y = 0$. Then

$$
\text{FL}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t .
$$

- $-\log p_t$ is ordinary binary cross-entropy.
- $(1-p_t)^\gamma$ is the *modulating factor*. The focusing parameter $\gamma \ge 0$ sets how fast easy examples fade out. With $\gamma = 0$ you're back to plain cross-entropy.
- $\alpha_t$ balances the classes: $\alpha_t = \alpha$ for positives and $1-\alpha$ for negatives.

With $\gamma = 2$, here's how much each example gets scaled down:

| $p_t$ (confidence in the true class) | Cross-entropy $-\log p_t$ | Factor $(1-p_t)^2$ |
|---|---|---|
| 0.9 (easy) | 0.105 | 0.01 (100× smaller) |
| 0.5 (unsure) | 0.693 | 0.25 |
| 0.1 (hard, wrong) | 2.303 | 0.81 (almost unchanged) |

So raising $\gamma$ doesn't really "punish mistakes more". Mostly it *quiets the easy examples*, which makes the hard ones a bigger share of the gradient.

This part tripped me up. The natural guess is to set $\alpha$ to the inverse class frequency, so the rare class gets the bigger weight. That's right for α-balanced cross-entropy on its own. But with the focal term, the paper found $\gamma = 2$ and $\alpha = 0.25$ for the rare foreground class worked best. That's a *smaller* weight on the rare class, because the modulating factor already gets rid of most of the easy negatives. So treat $\alpha$ and $\gamma$ as hyperparameters you tune together.

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

The focal value is much smaller here because most predictions are already on the right side. You can't compare its size directly with BCE. What matters is how the gradient gets split between easy and hard examples. In PyTorch, `torchvision.ops.sigmoid_focal_loss` does it from logits.

## Key takeaways

- MSE fits regression with roughly Gaussian noise. It's sensitive to outliers and a bad match for sigmoid or softmax outputs.
- Cross-entropy is the negative log-likelihood of the true label. With softmax its gradient is $p - y$, so it keeps learning even when the model is badly wrong.
- Use categorical cross-entropy when each example has one label, and per-label binary cross-entropy for multi-label problems. Compute both from logits.
- Focal loss multiplies cross-entropy by $(1-p_t)^\gamma$ to quiet easy examples. $\alpha_t$ balances the classes, and you need to tune the two together.

## Resources

- Tsung-Yi Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017): https://arxiv.org/abs/1708.02002
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, Section 6.2 "Gradient-Based Learning" (cost functions and output units): https://www.deeplearningbook.org/contents/mlp.html
- PyTorch docs, loss functions (`CrossEntropyLoss`, `BCEWithLogitsLoss`, `MSELoss`, `HuberLoss`): https://pytorch.org/docs/stable/nn.html#loss-functions
- torchvision docs, `sigmoid_focal_loss`: https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html
