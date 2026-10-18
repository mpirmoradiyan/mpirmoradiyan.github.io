---
title: "Causal attention: how a GPT is stopped from reading the future"
slug: causal-attention
date: 2025-01-09
topic: NLP & LLMs
tags: [attention, causal-mask, masked-self-attention, gpt, dropout, pytorch]
summary: "How a lower-triangular mask turns self-attention into causal attention, why setting scores to minus infinity before the softmax is the clean way to do it, and how the trick doubles as a fast weighted running average."
sources:
  - title: "Let's build GPT: from scratch, in code, spelled out"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=kCc8FmEb1nY"
    type: video
  - title: "Build a Large Language Model (From Scratch), Chapter 3: Coding attention mechanisms"
    author: "Sebastian Raschka (Manning, 2024)"
    url: "https://www.manning.com/books/build-a-large-language-model-from-scratch"
    type: book
  - title: "Attention Is All You Need"
    author: "Ashish Vaswani et al. (2017)"
    url: "https://arxiv.org/abs/1706.03762"
    type: paper


status: draft
---

A GPT is trained to predict token $t+1$ from tokens $1$ to $t$, at every position of a sequence at once. If position $t$ could attend to position $t+1$, it would simply copy the answer, the training loss would collapse, and the model would be useless at generation time, when the future does not exist yet. **Causal** (or masked) self-attention blocks every token from attending to anything after it. This note shows the mask, the clean way to apply it, and a neat matrix trick behind it.

## The trick: a running average as one matrix product

Start with a simpler goal. For a sequence $x_1, \dots, x_T$, replace each $x_t$ by the average of $x_1, \dots, x_t$, the token itself and everything before it. A double loop does it, but slowly. A lower-triangular matrix does it in one multiplication:

```python
import torch
torch.manual_seed(42)
T = 3
a = torch.tril(torch.ones(T, T))          # 1s on and below the diagonal
a = a / a.sum(1, keepdim=True)            # each row sums to 1
b = torch.randint(0, 10, (T, 2)).float()
print(a)        # [[1, 0, 0], [0.5, 0.5, 0], [0.33, 0.33, 0.33]]
print(a @ b)    # row t = mean of rows 0..t of b
```

Row $t$ of `a` has equal weights on positions $0, \dots, t$ and zeros after, so row $t$ of `a @ b` is the average of the first $t+1$ rows of `b`. With a batch, a (T, T) weight matrix times a (B, T, C) tensor broadcasts to (B, T, C), handling every sequence at once.

This is already causal attention with fixed, uniform weights. Self-attention makes the weights data-dependent, and the mask keeps them causal.

## Masking with minus infinity

The same uniform weights can be produced with a softmax:

```python
tril = torch.tril(torch.ones(T, T))
scores = torch.zeros(T, T)                               # stand-in for q @ k.T
scores = scores.masked_fill(tril == 0, float("-inf"))    # hide the future
weights = torch.softmax(scores, dim=-1)                  # same matrix as a above
```

Since $e^{-\infty} = 0$, masked positions get exactly zero weight, and the softmax spreads the remaining probability over the allowed positions. Replace the zeros by real query-key scores and you have causal self-attention:

$$
\operatorname{CausalAttention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V,
\qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

$M$ is the mask: 0 where token $i$ may look at token $j$, $-\infty$ where $j$ lies in the future.

## Mask before the softmax, not after

Another way that seems natural is to compute the full softmax, zero out the upper triangle, and renormalize each row so it sums to 1 again. It gives exactly the same numbers. In my notebook, both versions produced the row `[0.5517, 0.4483, 0, 0, 0, 0]` for the second token. Future tokens do not leak through the first softmax's denominator, because renormalizing removes their contribution completely.

Still, masking the scores with $-\infty$ before the softmax is the standard choice. It is one operation instead of three, and it is how the fused attention kernels in PyTorch work. For the mask itself, `torch.triu(torch.ones(T, T), diagonal=1)` marks the positions to hide (strictly above the diagonal), and `torch.tril` marks the ones to keep; use whichever reads better.

## Dropout on the attention weights

GPT models also apply dropout to the attention weights during training. A random subset of weights is set to zero, and the survivors are scaled by $1/(1-p)$ so the expected total stays the same. With $p = 0.5$ the kept weights double, which is why a matrix of ones turns into a pattern of 0s and 2s. Dropout makes the model rely less on any single connection. It is switched off by `model.eval()` at inference time.

## A batched causal attention module

```python
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):                                  # x: (B, T, d_in)
        B, T, _ = x.shape
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = q @ k.transpose(1, 2) / k.shape[-1] ** 0.5   # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return weights @ v                                 # (B, T, d_out)
```

Three details matter here:

- `k.transpose(1, 2)` swaps only the last two dimensions, so the batch dimension is left alone. Plain `k.T` would also reverse the batch axis.
- `register_buffer` stores the mask with the module, so it moves to the GPU with `model.to(device)` and is saved in the state dict, but the optimizer does not treat it as a parameter.
- `self.mask[:T, :T]` slices the mask to the actual sequence length, so the same module handles sequences shorter than `context_length`, which happens all the time during generation.

PyTorch also offers `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`, which applies the same mask inside a fused, memory-efficient kernel.

## Where causal attention is used

- **Decoder-only models** (GPT, Llama and most chat LLMs) use causal attention in every layer.
- **Encoder models** (BERT) let every token see the whole input and use no causal mask; they are built for understanding text, not generating it.
- **Encoder-decoder models** (the original transformer, T5) use full attention in the encoder, causal attention in the decoder, and cross-attention from decoder to encoder.

Because each position only sees the past, one forward pass over a training sequence of length $T$ yields $T$ valid next-token predictions, one per position. That is what makes GPT training so efficient.

## Key takeaways

- Causal attention lets token $t$ attend only to positions $1, \dots, t$, so next-token training cannot cheat.
- A normalized lower-triangular matrix computes running averages; adding learned scores turns this into causal self-attention.
- Set the masked scores to $-\infty$ before the softmax; they become exact zeros and the rest renormalizes automatically.
- Store the mask with `register_buffer` and slice it to the current length.
- Dropout on attention weights is a standard regularizer during training and is disabled at inference.

## Resources

- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY. The lower-triangular averaging trick follows this lecture.
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 3, sections on causal attention and dropout: https://www.manning.com/books/build-a-large-language-model-from-scratch; code: https://github.com/rasbt/LLMs-from-scratch
- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- PyTorch documentation, `torch.nn.functional.scaled_dot_product_attention`: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
