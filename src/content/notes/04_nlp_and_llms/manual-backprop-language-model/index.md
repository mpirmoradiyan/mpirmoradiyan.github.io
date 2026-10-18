---
title: "Backprop by hand through an MLP language model"
slug: manual-backprop-language-model
date: 2024-12-25
topic: NLP & LLMs
tags: [backpropagation, cross-entropy, gradients, embeddings, tensors, makemore, pytorch]
summary: "How to write the backward pass of a character-level MLP by hand at the tensor level, including the compact gradient of softmax cross-entropy and the scatter-add for the embedding table, and verify every step against PyTorch autograd."
sources:
  - title: "Building makemore Part 4: Becoming a Backprop Ninja"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=q8SA3rM6ckI"
    type: video
  - title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    author: "Sergey Ioffe, Christian Szegedy (2015)"
    url: "https://arxiv.org/abs/1502.03167"
    type: paper
  - title: "Yes you should understand backprop"
    author: "Andrej Karpathy"
    url: "https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b"
    type: docs


status: draft
---

`loss.backward()` hides a lot. When training goes wrong (dead units, exploding gradients, a silent shape bug) it pays to know what flows backwards through each line of the forward pass. In this exercise I replaced autograd with a hand-written backward pass for the MLP language model, one tensor at a time, and checked each gradient against PyTorch. The scalar chain rule is covered in the [micrograd note](/notes/backpropagation-micrograd/); here the focus is on tensors, shapes and broadcasting.

## The forward pass, broken into small steps

The model embeds 3 characters with 10-dimensional vectors, applies a linear layer, batch normalization and tanh, then an output layer. The cross-entropy loss is written out explicitly so every intermediate has a name.

```python
emb = C[Xb]                                  # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)          # (32, 30)
hprebn = embcat @ W1 + b1                    # (32, 64)
# ... batch norm produces hpreact ...
h = torch.tanh(hpreact)                      # (32, 64)
logits = h @ W2 + b2                         # (32, 27)
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes           # for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

Each intermediate gets `retain_grad()` so the hand-computed gradient `dX` can be compared with `X.grad`.

## Two rules that handle most of the shape bookkeeping

**1. A gradient has the shape of the thing it belongs to.** `dW2` must be (64, 27) because `W2` is. This alone often tells you where a transpose goes.

**2. Broadcasting and summing are mirror images.**
- If a tensor was **broadcast** (copied) in the forward pass, **sum** its gradient over the copied dimension in the backward pass. Example: `b2` of shape (27,) is added to every row of the (32, 27) logits, so `db2 = dlogits.sum(0)`.
- If a tensor was **summed** in the forward pass, **broadcast** the gradient back. Example: `counts_sum = counts.sum(1, keepdim=True)`, so every element of a row of `counts` receives the gradient of that row's sum.

A variable used in two places collects gradient from both, so its gradients are added. `counts` feeds both `probs` and `counts_sum`, so `dcounts` has two contributions.

## Walking backwards

Below, `n` is the batch size (32). The loss averages $-\log p$ at the target positions only. So `dlogprobs` is $-1/n$ at the 32 target entries and 0 everywhere else.

```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n
dprobs = (1.0 / probs) * dlogprobs                       # d log(x) = 1/x
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True) # was broadcast
dcounts = counts_sum_inv * dprobs
dcounts_sum = -counts_sum**-2 * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum          # second use
dnorm_logits = counts * dcounts                          # d exp(x) = exp(x)
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, 27) * dlogit_maxes
```

`dlogit_maxes` comes out as essentially zero. That makes sense: subtracting a constant from every logit in a row does not change the softmax, so the loss cannot depend on it. The max subtraction is there only to stop `exp` from overflowing.

For a linear layer $Y = XW + b$ the rules are

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top, \qquad
\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}, \qquad
\frac{\partial L}{\partial b} = \sum_{\text{rows}} \frac{\partial L}{\partial Y}.
$$

You do not need to memorize them: write the matrix product whose shapes match the target.

```python
dh = dlogits @ W2.T          # (32,27) @ (27,64) -> (32,64)
dW2 = h.T @ dlogits          # (64,32) @ (32,27) -> (64,27)
db2 = dlogits.sum(0)
dhpreact = (1.0 - h**2) * dh # tanh'(x) = 1 - tanh(x)^2
```

## The gradient of cross-entropy in one line

The eight steps from `logits` to `loss` collapse into a short formula. For one example with logits $z$, softmax probabilities $p = \text{softmax}(z)$ and target $y$, the loss is $-\log p_y$ and

$$
\frac{\partial L}{\partial z_j} = p_j - \mathbb{1}[j = y].
$$

Averaging over a batch of $n$ examples divides by $n$:

```python
dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```

This matched PyTorch to about $10^{-9}$. The formula is also easy to read: every wrong class is pushed down in proportion to the probability the model gave it, and the correct class is pulled up by $1 - p_y$. Each row of `dlogits` sums to zero, so the push and the pull balance.

![Left: the 32 by 27 logits of a batch, which look like random noise at initialization. Right: the gradient with respect to the logits. Every row has one dark cell at the target character (negative gradient) and small light values elsewhere.](fig-logits-and-dlogits.png)

## Through batch norm, and into the embedding table

Batch normalization couples all examples in a batch through the mean and variance, so its backward pass has several paths. Doing it step by step and then simplifying on paper gives a single expression. With $\hat{x}$ the normalized pre-activations, $\gamma$ the gain, $\sigma^{-1} = (\text{var} + \epsilon)^{-1/2}$, and the unbiased variance (divide by $n-1$):

$$
\frac{\partial L}{\partial x} = \frac{\gamma\,\sigma^{-1}}{n}\left(n\,\frac{\partial L}{\partial y} - \sum_{i}\frac{\partial L}{\partial y_i} - \frac{n}{n-1}\,\hat{x}\sum_{i}\frac{\partial L}{\partial y_i}\,\hat{x}_i\right)
$$

The sums run over the batch, separately for each of the 64 features. I only note it here; the layer itself is explained in the [batch normalization note](/notes/batch-normalization/).

The last step is the embedding lookup `emb = C[Xb]`. Each row of `C` was copied into `emb` once for every time its character appeared in the batch. So the gradient of a row of `C` is the **sum** of the gradients at all positions that used it. That is a scatter-add:

```python
demb = dembcat.view(emb.shape)                 # undo the flatten
dC = torch.zeros_like(C)
dC.index_add_(0, Xb.view(-1), demb.view(-1, C.shape[1]))
```

My first version used two nested Python loops over batch and position. It gives the same result; `index_add_` does it in one vectorized call.

## Training with only my own gradients

With every gradient in place I removed `loss.backward()`, wrapped the loop in `torch.no_grad()`, and trained the network for 200,000 steps using only the hand-written backward pass. The loss went from 3.77 to around 1.9-2.0 on the minibatches, the same behaviour as with autograd. A final check of all parameter gradients against PyTorch showed differences below $2 \times 10^{-8}$. The small mismatches come from floating-point rounding when the algebra is done in a different order, not from errors.

## Key takeaways

- A tensor gradient always has the same shape as its tensor; matching shapes tells you where transposes and sums go.
- Broadcast in the forward pass means sum in the backward pass, and sum in the forward pass means broadcast in the backward pass.
- The gradient of softmax cross-entropy with respect to the logits is $(\text{softmax} - \text{one\_hot})/n$. It is cheap, stable and easy to interpret.
- Subtracting the max logit has zero gradient; it exists only for numerical stability.
- The gradient of an embedding table is a scatter-add of the gradients at every position that used each row.

## Resources

- Andrej Karpathy, "Building makemore Part 4: Becoming a Backprop Ninja", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=q8SA3rM6ckI. The exercise structure and the step-by-step forward pass come from this lecture.
- Andrej Karpathy, "Yes you should understand backprop" (blog post): https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b
- Sergey Ioffe and Christian Szegedy, "Batch Normalization", 2015: https://arxiv.org/abs/1502.03167
- PyTorch documentation, `Tensor.index_add_`: https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html
