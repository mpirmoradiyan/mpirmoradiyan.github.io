---
title: "Getting initialization right: confident softmax, saturated tanh and Kaiming scaling"
slug: activations-at-initialization
date: 2024-12-18
topic: NLP & LLMs
tags: [initialization, kaiming, tanh, softmax, activations, gradients, makemore, pytorch]
summary: "Why a randomly initialized MLP language model starts with a huge loss and half-dead tanh units, how to fix both with simple scaling rules, and which statistics to plot to check that a deeper network is healthy."
sources:
  - title: "Building makemore Part 3: Activations & Gradients, BatchNorm"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=P6sfmUTpUmc"
    type: video
  - title: "Understanding the difficulty of training deep feedforward neural networks"
    author: "Xavier Glorot, Yoshua Bengio (AISTATS 2010)"
    url: "https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"
    type: paper
  - title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    author: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)"
    url: "https://arxiv.org/abs/1502.01852"
    type: paper
  - title: "torch.nn.init"
    author: "PyTorch"
    url: "https://pytorch.org/docs/stable/nn.init.html"
    type: docs


status: draft
---

The MLP language model from the previous note trains, but its first loss is around 34 when it should be about 3.3, and many of its hidden units barely learn. Both problems come from how the weights are initialized. They are easy to miss in a two-layer network and fatal in a deep one. This note diagnoses them with a few histograms and fixes them with simple scaling rules.

## Know the loss you should expect at step 0

Before training, the network has no reason to prefer any of the 27 characters. The best it can honestly do is predict the uniform distribution, which gives a loss of

$$
-\log\frac{1}{27} = \log 27 \approx 3.30 .
$$

My first run started at a loss of about 34. That means the network was not unsure; it was **confidently wrong**. Some logits were large, so softmax put almost all probability on a few characters, and the true character usually got almost none.

A tiny example with 8 classes shows why. If all logits are 0, every class gets probability 1/8 and the loss for any target is $\log 8 \approx 2.08$. If one wrong logit is 5, that class takes 95% of the mass and every other class gets 0.6%, so the loss on a typical target jumps to about 5. With logits like 10 or 9, the loss for most targets goes above 10.

The logits are `h @ W2 + b2`. To start near uniform, make them small: set `b2` to zero and scale `W2` down.

```python
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.zeros(vocab_size)
```

With that change the first loss became **3.28**, right where it should be. The loss curve lost its "hockey stick" shape: the network no longer spends its first thousand steps just shrinking the logits, so the training budget goes into learning real structure. In my 10,000-step runs, the final training-split loss improved from 2.81 to 2.44.

Do not set `W2` exactly to zero. With zero weights, every hidden unit receives the same gradient and they stay identical. A small random scale keeps the symmetry broken.

## The second problem: saturated tanh units

The hidden layer is `h = tanh(hpreact)`. The derivative of tanh is

$$
\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x).
$$

When $|\tanh(x)|$ is close to 1, this derivative is close to 0. In backpropagation the gradient arriving at the unit is multiplied by this local derivative, so a saturated unit passes almost no gradient back to its weights or to the layers below. If a unit is saturated for every example in a batch, it does not learn at all; it is a **dead neuron**.

The pre-activations were the problem. They had a standard deviation of about 5, so most of them landed deep in the flat tails of tanh.

![Three histograms for the hidden layer at initialization. The pre-activations spread from about -15 to 15. The tanh outputs pile up at -1 and +1. The local derivative 1 - h squared is mostly near 0.](fig-tanh-saturated.png)

Scaling `W1` by 0.2 and `b1` by 0.01 shrinks the pre-activations into a range where tanh is still roughly linear:

![The same three histograms after scaling W1 and b1 down. Pre-activations sit between about -2 and 2, tanh outputs are spread smoothly across -1 to 1, and most derivatives are close to 1.](fig-tanh-healthy.png)

Now almost every unit passes a healthy gradient. In this shallow network the effect on the final loss was small and noisy across my short runs. A two-layer net is forgiving; the same mistake in a ten-layer net compounds layer after layer.

## A principled scale instead of magic numbers

Guessing 0.2 does not scale to many layers of different sizes. Look at what a linear layer does to the variance. If $x$ has $n$ independent inputs with variance 1 and the weights also have variance 1, then each output $y = \sum_{i=1}^{n} x_i w_i$ is a sum of $n$ terms with variance 1, so

$$
\operatorname{Var}(y) = n \quad\Rightarrow\quad \operatorname{std}(y) = \sqrt{n}.
$$

Here $n$ is the **fan-in**, the number of inputs to each unit. A quick check with 10 inputs gives an output standard deviation of about 3.1, close to $\sqrt{10}$. Divide the weights by $\sqrt{n}$ and the output standard deviation goes back to about 1.

```python
x = torch.randn(1000, 10)
w = torch.randn(10, 200) / 10**0.5   # divide by sqrt(fan_in)
y = x @ w
print(x.std(), y.std())              # both close to 1
```

Nonlinearities shrink the signal, so each one needs a **gain** to compensate. This is the idea behind Kaiming (He) initialization:

$$
W \sim \mathcal{N}\!\left(0,\; \frac{\text{gain}^2}{\text{fan\_in}}\right),
$$

with gain $\sqrt{2}$ for ReLU and $5/3$ for tanh. For the MLP with 10-dimensional embeddings and a context of 3, the fan-in is 30:

```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) \
     * (5/3) / (n_embd * block_size) ** 0.5
```

PyTorch implements the same rule as `torch.nn.init.kaiming_normal_(w, nonlinearity="tanh")`.

## Diagnosing a deeper network

To see whether the rule holds up, I stacked five `Linear` + `Tanh` layers of width 100, with every linear weight initialized as `randn / sqrt(fan_in) * 5/3` and the last layer scaled by 0.1. Then I tracked four statistics.

1. **Activation histograms per layer.** The tanh outputs had a standard deviation of about 0.7 in every layer and 8-9% of units were saturated ($|h| > 0.97$), except the first layer at about 20%. Without the 5/3 gain the activations shrink towards 0 layer by layer; with a much larger gain they saturate.
2. **Gradient histograms per layer.** The gradients at the tanh outputs had similar spread in all layers (standard deviation about $1$ to $2 \times 10^{-3}$). Gradients that grow or shrink sharply with depth signal trouble.
3. **Gradient-to-data ratio** for each weight matrix: `p.grad.std() / p.std()`. It was around 0.02-0.03 for the hidden layers. The last layer was much higher (0.65), which is expected right after scaling its weights down by 0.1, and it settles during training.
4. **Update-to-data ratio** over time: the standard deviation of `lr * p.grad` divided by that of `p`, plotted on a log10 scale. A rough rule of thumb is that it should hover around $10^{-3}$. Much lower means the learning rate is too small; much higher means updates are too violent.

```python
with torch.no_grad():
    for p in parameters:
        if p.ndim == 2:
            ratio = ((lr * p.grad).std() / p.data.std()).log10().item()
            print(tuple(p.shape), f"{ratio:.2f}")   # aim for about -3
```

These plots are cheap and turn "training feels slow" into something you can see and fix.

## The less fragile route

Careful initialization works, but it is brittle: change the architecture and you have to redo the arithmetic. Normalization layers make networks far less sensitive to the initial scale. They standardize the pre-activations directly, during training. Batch normalization was the first widely used one and is the subject of its own note in Deep Learning Foundations ([Batch normalization](/notes/batch-normalization/)). Transformers use layer normalization instead, which appears in the GPT note. Even with normalization, a sensible initial scale and an output layer that starts near uniform still help.

## Key takeaways

- Compute the expected initial loss ($\log$ of the vocabulary size for a uniform guess). A much higher first loss means the softmax is confidently wrong.
- Start the output layer small (zero bias, small weights) so the first predictions are near uniform.
- Large pre-activations saturate tanh, kill its local gradient and can leave units dead. Check the histogram of activations.
- Scale weights by $\text{gain}/\sqrt{\text{fan\_in}}$ (Kaiming), with gain 5/3 for tanh and $\sqrt{2}$ for ReLU, so activations keep unit scale through depth.
- Monitor activation and gradient statistics per layer and the update-to-data ratio (around $10^{-3}$) to catch problems early.

## Resources

- Andrej Karpathy, "Building makemore Part 3: Activations & Gradients, BatchNorm", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=P6sfmUTpUmc. The diagnostics and the deep-MLP experiment follow this lecture.
- Xavier Glorot and Yoshua Bengio, "Understanding the difficulty of training deep feedforward neural networks", AISTATS 2010: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
- Kaiming He et al., "Delving Deep into Rectifiers", 2015: https://arxiv.org/abs/1502.01852
- PyTorch documentation, `torch.nn.init` (including `calculate_gain` and `kaiming_normal_`): https://pytorch.org/docs/stable/nn.init.html
