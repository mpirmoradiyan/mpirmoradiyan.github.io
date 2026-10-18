---
title: "A WaveNet-style language model: fusing context in a tree instead of all at once"
slug: wavenet-hierarchical-model
date: 2024-12-27
topic: NLP & LLMs
tags: [language-model, wavenet, hierarchical, batchnorm, tensor-shapes, makemore, pytorch]
summary: "How to turn a flat MLP language model into a hierarchical one that merges neighbouring characters two at a time, why that works better for the same number of parameters, and the batch-norm shape bug that comes with 3D activations."
sources:
  - title: "Building makemore Part 5: Building a WaveNet"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=t3YJ5hKiMQ0"
    type: video
  - title: "WaveNet: A Generative Model for Raw Audio"
    author: "Aäron van den Oord et al. (2016)"
    url: "https://arxiv.org/abs/1609.03499"
    type: paper


status: draft
---

The MLP language model concatenates all context characters and squeezes them through one hidden layer in a single step. With a longer context that is wasteful: eight characters of information get crushed together immediately. WaveNet, a 2016 model for raw audio, combines inputs gradually in a tree: neighbours first, then pairs of pairs, and so on. This note builds that idea for character-level names and tidies the code into small PyTorch-like modules along the way.

## From a flat MLP to a tree

With a context of 8 characters and 10-dimensional embeddings, the flat model does

```
(B, 8, 10) --flatten--> (B, 80) --Linear--> (B, 200) --> ... --> (B, 27)
```

The hierarchical model instead merges two neighbours at a time:

```
(B, 8, 10) -> (B, 4, 20) -> Linear -> (B, 4, 68)    # pairs of characters
           -> (B, 2, 136) -> Linear -> (B, 2, 68)   # pairs of pairs
           -> (B, 136)    -> Linear -> (B, 68)      # whole context
           -> Linear -> (B, 27)
```

Each level sees a slightly larger piece of the context and builds a richer summary of it. The final layer gets a summary of all eight characters that was built step by step, not in one jump.

## Two tensor tricks that make it easy

**Matrix multiplication broadcasts over leading dimensions.** A linear layer only acts on the last dimension:

```python
(torch.randn(4, 5, 6, 80) @ torch.randn(80, 200)).shape   # (4, 5, 6, 200)
```

So a `Linear` layer applied to a (B, 4, 20) tensor processes the 4 pairs in parallel, with the same weights for every pair.

**`view` can group neighbours.** For a (B, 8, 10) tensor in contiguous memory, `x.view(B, 4, 20)` places characters 0 and 1 side by side, then 2 and 3, and so on. It is exactly the same as concatenating the even and odd positions:

```python
e = torch.randn(4, 8, 10)
explicit = torch.cat([e[:, ::2, :], e[:, 1::2, :]], dim=2)
print((e.view(4, 4, 20) == explicit).all())   # tensor(True)
```

That gives a tiny layer:

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n                       # how many neighbours to merge
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:              # last level: drop the time axis
            x = x.squeeze(1)
        self.out = x
        return x
    def parameters(self):
        return []
```

## Organizing the code like `torch.nn`

Up to this point the MLP was a pile of loose tensors. I rewrote it as small classes with the same interface as PyTorch modules: `Embedding`, `FlattenConsecutive`, `Linear`, `BatchNorm1d`, `Tanh` and a `Sequential` container, each with a `__call__` for the forward pass and a `parameters()` method. The whole model then reads like its diagram:

```python
n_embd, n_hidden = 10, 68
model = Sequential([
    Embedding(27, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False),
    BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, 27),
])
```

The linear layers before batch norm have no bias, because batch norm subtracts the mean and has its own learned shift. The hidden width of 68 was chosen so the model has about 22,400 parameters, roughly the same as the flat model with 200 hidden units (about 22,100). That keeps the comparison fair.

## The batch-norm bug that 3D inputs create

My `BatchNorm1d` computed statistics with `x.mean(0)` and `x.var(0)`. That is correct for 2D input of shape (B, C). With the hierarchical model the input to the first batch-norm layer is (B, 4, 68). Reducing only over dimension 0 gives statistics of shape (1, 4, 68): a separate mean and variance for each of the 4 pair positions. Nothing crashes, because broadcasting makes the shapes work. But each statistic is estimated from fewer numbers, and the running averages end up with the wrong shape.

The 4 positions are just more examples of "a pair of characters", so the statistics should be pooled over both the batch and the position dimension:

```python
dim = 0 if x.ndim == 2 else (0, 1)
xmean = x.mean(dim, keepdim=True)    # (1, 1, 68)
xvar = x.var(dim, keepdim=True)
```

This is what `torch.nn.BatchNorm1d` does for 3D input, except that PyTorch expects the channel dimension in the middle, (B, C, L), not last. Silent shape bugs like this are a good reason to print the output shape of every layer once:

```python
for layer in model.layers:
    print(layer.__class__.__name__, tuple(layer.out.shape))
```

## Results

Trained for 200,000 steps with batch size 32 and a learning rate of 0.1, dropped to 0.01 after 150,000 steps:

| model | context | train loss | val loss |
|---|---|---|---|
| hierarchical (this note) | 8 | 1.91 | 2.03 |

This is clearly better than the 3-character MLP from the earlier notes, whose validation loss was around 2.1 to 2.25 depending on size and initialization. Part of the gain comes from the longer context and part from the hierarchical structure. Before evaluating, every layer must be switched to evaluation mode (`layer.training = False`) so batch norm uses its running statistics instead of the statistics of the current batch.

![Training loss (log10 of the cross-entropy, averaged over windows of 1,000 steps) over 200,000 steps. It falls steadily from about 0.39 to 0.30, then drops sharply to about 0.285 when the learning rate is reduced at step 150,000.](fig-wavenet-loss.png)

Averaging the noisy per-batch loss over windows of 1,000 steps makes the curve readable. The step down at 150,000 is the learning-rate decay.

Samples now look like real names: `lyncia`, `audlyn`, `kimber`, `erisha`, `ella`, `roni`.

## How this relates to the real WaveNet

The real WaveNet uses **dilated causal convolutions**. A convolution with kernel size 2 is the same small linear layer slid along the sequence, and dilation makes each level skip further, so the receptive field doubles per layer. My model computes the same tree for a single output position. A convolutional version would compute the outputs for every position in the sequence at once and reuse shared intermediate results, which is much more efficient for training. WaveNet also adds gated activations and residual and skip connections, which are left out here.

## Key takeaways

- A hierarchical model fuses context gradually (pairs, then pairs of pairs) instead of flattening everything into one layer.
- `Linear` layers act on the last dimension and broadcast over all others, and `view` can group neighbouring time steps without copying.
- With 3D activations, batch norm must reduce over both batch and time dimensions. Broadcasting will hide the mistake if it does not.
- Wrapping layers in small module classes makes the model readable and makes shape bugs easier to find.
- At about the same parameter count, the 8-character hierarchical model reached a validation loss of about 2.03.

## Resources

- Andrej Karpathy, "Building makemore Part 5: Building a WaveNet", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=t3YJ5hKiMQ0. The model and module classes are adapted from this lecture.
- Aäron van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016: https://arxiv.org/abs/1609.03499
- PyTorch documentation, `torch.nn.BatchNorm1d` (input shapes): https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
