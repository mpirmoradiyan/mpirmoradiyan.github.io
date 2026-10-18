---
title: "Batch normalization: what it does, why it helps, and how it behaves at test time"
slug: batch-normalization
date: 2024-12-27
topic: Deep Learning Foundations
tags: [batch-normalization, normalization, training-stability, saturation, pytorch]
summary: "How batch normalization standardizes each feature over a mini-batch, why that keeps activations out of saturation, and why it uses batch statistics in training but running averages at inference."
sources:
  - title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    author: "Sergey Ioffe, Christian Szegedy"
    url: "https://arxiv.org/abs/1502.03167"
    type: paper
  - title: "How Does Batch Normalization Help Optimization?"
    author: "Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry"
    url: "https://arxiv.org/abs/1805.11604"
    type: paper
  - title: "Understanding the difficulty of training deep feedforward neural networks"
    author: "Xavier Glorot, Yoshua Bengio"
    url: "https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
    type: paper
  - title: "Building makemore Part 3: Activations & Gradients, BatchNorm"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=P6sfmUTpUmc"
    type: video
  - title: "torch.nn.BatchNorm1d"
    author: "PyTorch"
    url: "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html"
    type: docs


status: draft
---

Deep networks are hard to train when the values flowing through them drift to extreme ranges. Saturating activations then stop passing gradient, and the learning rate has to be tiny to avoid blowing up. Batch normalization (BatchNorm) fixes the scale of each layer's inputs during training. It became a standard layer because it lets networks train faster, with higher learning rates and less fuss about initialization.

## The problem: activations that drift and saturate

During training, each layer sees inputs produced by all the layers before it. When earlier weights change, the distribution of those inputs changes too, so later layers keep chasing a moving target. Ioffe and Szegedy called this *internal covariate shift*. ("Covariate shift" is the classic name for a model's input distribution changing between training and use.)

A concrete symptom is saturation. Take a layer $z = g(Wu + b)$ with a sigmoid or $\tanh$ activation $g$. When $|Wu + b|$ is large, $g'$ is almost zero, so almost no gradient flows back to $u$ or to the weights below. Large pre-activations come easily: with random weights the variance of a sum grows with the number of inputs (the fan-in).

I checked this with a small experiment: a batch of 10 inputs with 100 features, random normal weights to a 20-unit $\tanh$ layer, then a second 20-unit $\tanh$ layer.

![Histograms for layer 1: inputs, weights and biases are near standard normal, but the pre-activation out1 spreads from about -25 to 25, so tanh pushes almost all outputs to -1 or +1; normalizing those outputs only rescales the two clusters](fig-layer1-histograms.png)

With fan-in 100 and unit-variance weights, the pre-activations have a standard deviation of about 10. Almost every $\tanh$ output is pinned at $\pm 1$. Standardizing the *outputs* afterwards does not help much: it only moves the two clusters to around $\pm 1$ again.

![Histograms for layer 2: whether or not its input was normalized, the second layer's pre-activations still span about -10 to 10 and its tanh outputs are still saturated at -1 and +1](fig-layer2-histograms.png)

The second layer saturates in both cases because its own weights are again too large for its fan-in. The lesson: what needs controlling is the input to the nonlinearity, layer by layer. That is exactly where BatchNorm is placed in the original paper: after the linear layer and before the activation.

## The transform

For one feature, take its values over a mini-batch of size $m$, $x_1, \dots, x_m$. BatchNorm computes

$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2,
$$

$$
\hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad
y_i = \gamma\,\hat x_i + \beta .
$$

- $\mu_B$ and $\sigma_B^2$ are the batch mean and (biased) batch variance of that feature.
- $\epsilon$ is a small constant (PyTorch uses $10^{-5}$) that avoids division by zero when the variance is tiny.
- $\hat x_i$ has zero mean and unit variance over the batch.
- $\gamma$ (scale) and $\beta$ (shift) are learned per feature. They give back the freedom that normalization took away. With $\gamma = \sqrt{\sigma_B^2+\epsilon}$ and $\beta = \mu_B$, the layer could even undo the normalization if that were best.

Every feature (every channel, for convolutions) gets its own $\mu$, $\sigma^2$, $\gamma$ and $\beta$. Because $\beta$ already shifts the output, the bias of the preceding linear or conv layer is redundant; that layer is usually created with `bias=False`.

## Training versus inference

During training the statistics come from the current batch. That has two side effects. Each example's output depends a little on which other examples share its batch, which adds noise and acts as a mild regularizer. And you cannot use it as-is at inference, where you may have a single example and want a deterministic answer.

So BatchNorm keeps running estimates of the mean and variance, updated at every training step:

$$
\hat\mu \leftarrow (1-\alpha)\,\hat\mu + \alpha\,\mu_B, \qquad
\hat\sigma^2 \leftarrow (1-\alpha)\,\hat\sigma^2 + \alpha\,\sigma_B^2,
$$

where $\alpha$ is the momentum (0.1 by default in PyTorch). At inference the layer uses $\hat\mu$ and $\hat\sigma^2$ instead of batch statistics, so it becomes a fixed affine map per feature. (PyTorch normalizes with the biased batch variance but stores the unbiased one, with $m-1$ in the denominator, in the running estimate.)

```python
import numpy as np

def batchnorm(x, gamma, beta, state, training, momentum=0.1, eps=1e-5):
    """x: (batch, features). state holds running 'mean' and 'var'."""
    if training:
        mu, var = x.mean(axis=0), x.var(axis=0)            # biased variance
        m = x.shape[0]
        state['mean'] = (1 - momentum) * state['mean'] + momentum * mu
        state['var'] = (1 - momentum) * state['var'] + momentum * var * m / (m - 1)
    else:
        mu, var = state['mean'], state['var']
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta

x = np.random.randn(32, 20) * 10 + 3            # badly scaled pre-activations
state = {'mean': np.zeros(20), 'var': np.ones(20)}
y = batchnorm(x, np.ones(20), np.zeros(20), state, training=True)
print(y.mean(axis=0)[:3].round(3), y.std(axis=0)[:3].round(3))   # ~0 and ~1
```

In PyTorch this is `nn.BatchNorm1d(num_features)` for (batch, features) data and `nn.BatchNorm2d(channels)` for images. The switch between the two modes is `model.train()` and `model.eval()`. Forgetting `model.eval()` before evaluation is a common bug: predictions then depend on the batch, and the running statistics keep changing on test data.

## Why it helps (and what is still debated)

What is clear in practice:

- Pre-activations stay in a sensible range, so saturating activations keep useful gradients.
- Gradients depend less on the scale of the weights, so higher learning rates are stable and initialization matters less.
- The batch noise gives a small regularization effect, which often reduces the need for dropout.

Why it works is less settled than the original paper suggested. Santurkar et al. (2018) showed that BatchNorm still helps even when noise is deliberately injected after it to bring the distribution shift back. Their explanation is that it makes the loss surface smoother, so gradients are more predictive and larger steps are safe. "Reducing internal covariate shift" is a good intuition for the motivation but probably not the main mechanism.

## Limits and alternatives

BatchNorm needs reasonably large batches; with very small batches $\mu_B$ and $\sigma_B^2$ are noisy and the train and test behaviour drift apart. It is also awkward for sequence models with variable lengths. Common alternatives normalize over other axes and behave the same in training and inference:

- **Layer Normalization**: normalizes over the features of each single example. It is the standard choice in Transformers.
- **Group Normalization**: normalizes over groups of channels per example. It is popular in detection and segmentation with small batches.
- **Batch Renormalization**: keeps the batch axis but corrects towards the running statistics.

## Key takeaways

- BatchNorm standardizes each feature over the mini-batch, then applies a learned scale $\gamma$ and shift $\beta$; $\epsilon$ keeps the division safe.
- Put it on the input to the nonlinearity; normalizing after a saturated activation is too late.
- Training uses batch statistics; inference uses running averages, so always call `model.eval()` before evaluating.
- It allows higher learning rates and reduces sensitivity to initialization; the deeper reason is likely a smoother loss surface.
- For small batches or sequences, prefer LayerNorm or GroupNorm.

## Resources

- Sergey Ioffe, Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (2015): https://arxiv.org/abs/1502.03167
- Shibani Santurkar et al., "How Does Batch Normalization Help Optimization?" (2018): https://arxiv.org/abs/1805.11604
- Johan Bjorck et al., "Understanding Batch Normalization" (2018): https://arxiv.org/abs/1806.02375
- Xavier Glorot, Yoshua Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010): https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
- Yann LeCun et al., "Efficient BackProp" (1998), on why whitened inputs train faster: https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf
- Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey Hinton, "Layer Normalization" (2016): https://arxiv.org/abs/1607.06450
- Yuxin Wu, Kaiming He, "Group Normalization" (2018): https://arxiv.org/abs/1803.08494
- Sergey Ioffe, "Batch Renormalization" (2017): https://arxiv.org/abs/1702.03275
- Yuxin Wu, Justin Johnson, "Rethinking 'Batch' in BatchNorm" (2021): https://arxiv.org/abs/2105.07576
- Lei Huang et al., "Normalization Techniques in Training DNNs: Methodology, Analysis and Application" (2020): https://arxiv.org/abs/2009.12836
- Andrej Karpathy, "Building makemore Part 3: Activations & Gradients, BatchNorm" (video): https://www.youtube.com/watch?v=P6sfmUTpUmc
- PyTorch docs, `torch.nn.BatchNorm1d`: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
