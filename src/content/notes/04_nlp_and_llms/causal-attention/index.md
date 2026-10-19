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

Plain [self-attention](/notes/self-attention/) has a catch when you use it in a GPT. A GPT is trained to predict token $t+1$ from tokens $1$ to $t$, at every position of a sequence at the same time. But if position $t$ can look at position $t+1$, it just copies the answer. The training loss drops to almost nothing, and the model is useless when it has to write text, because then the future doesn't exist yet. **Causal** (or masked) self-attention fixes this: no token gets to look at anything after it. Below is the mask, the clean way to apply it, and a neat matrix trick that sits behind it.

## The trick: a running average as one matrix product

Before any attention, start with something simpler. For a sequence $x_1, \dots, x_T$, replace each $x_t$ by the average of $x_1, \dots, x_t$: the token itself and everything before it. A double loop works, but it's slow. A lower-triangular matrix (ones on and below the diagonal, zeros above) does it in one multiplication:

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

Row $t$ of `a` has equal weights on positions $0, \dots, t$ and zeros after that, so row $t$ of `a @ b` is the average of the first $t+1$ rows of `b`. With a batch, a (T, T) weight matrix times a (B, T, C) tensor broadcasts to (B, T, C), so every sequence is handled at once.

When I saw this, it clicked: this is already causal attention, just with fixed, equal weights. Self-attention makes the weights depend on the data, and the mask keeps them from looking ahead.

## Masking with minus infinity

You can get the same equal weights out of a softmax:

```python
tril = torch.tril(torch.ones(T, T))
scores = torch.zeros(T, T)                               # stand-in for q @ k.T
scores = scores.masked_fill(tril == 0, float("-inf"))    # hide the future
weights = torch.softmax(scores, dim=-1)                  # same matrix as a above
```

Since $e^{-\infty} = 0$, the hidden positions get exactly zero weight, and the softmax spreads the rest of the probability over the allowed positions. Swap the zeros for real query-key scores and you have causal self-attention:

$$
\operatorname{CausalAttention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V,
\qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

$M$ is the mask: 0 where token $i$ is allowed to look at token $j$, and $-\infty$ where $j$ is in the future.

## Mask before the softmax, not after

Another way that feels natural is to do the full softmax, zero out the upper triangle, and then rescale each row so it adds up to 1 again. I wondered whether the future tokens would sneak in through the first softmax's denominator. They don't. It gives exactly the same numbers, because the rescaling removes their part completely. In my notebook, both versions gave the row `[0.5517, 0.4483, 0, 0, 0, 0]` for the second token.

Still, masking the scores with $-\infty$ before the softmax is the usual choice, simply because it's one step instead of three. For the mask itself, `torch.triu(torch.ones(T, T), diagonal=1)` marks the positions to hide (strictly above the diagonal), and `torch.tril` marks the ones to keep. Use whichever reads better to you.

## Dropout on the attention weights

GPT models also use dropout on the attention weights during training. Some weights are set to zero at random, and the ones that survive are scaled by $1/(1-p)$ so the expected total stays the same. With $p = 0.5$ the kept weights double, which is why a matrix of ones turns into a pattern of 0s and 2s. This confused me the first time I saw it. Dropout makes the model depend less on any single connection. `model.eval()` turns it off when you actually use the model.

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

Three small things in there are easy to get wrong:

- `k.transpose(1, 2)` swaps only the last two dimensions and leaves the batch dimension alone. Plain `k.T` would flip the batch axis too.
- `register_buffer` keeps the mask inside the module, so it moves to the GPU with `model.to(device)` and gets saved in the state dict. But the optimizer doesn't treat it as something to train.
- `self.mask[:T, :T]` cuts the mask down to the real sequence length. That way the same module works on sequences shorter than `context_length`, which happens all the time during generation.

PyTorch also has `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)`, which applies the same mask in one call and, on a GPU, can use a fused kernel that saves memory. With dropout off, it gives the same output as the module above.

Real models run several of these heads side by side, which is the next step: [multi-head attention](/notes/multi-head-attention/).

## Where causal attention shows up

- **Decoder-only models** (GPT, Llama and most chat LLMs) use causal attention in every layer.
- **Encoder models** (BERT) let every token see the whole input, with no causal mask. They're built for understanding text, not generating it.
- **Encoder-decoder models** (the original transformer, T5) use full attention in the encoder, causal attention in the decoder, and cross-attention from the decoder to the encoder.

Because each position only sees the past, one forward pass over a training sequence of length $T$ gives $T$ honest next-token predictions, one per position. That's a big part of why GPT training is so efficient.

## Key takeaways

- Causal attention lets token $t$ look only at positions $1, \dots, t$, so next-token training can't cheat.
- A lower-triangular matrix with rows that sum to 1 computes running averages. Add learned scores and it becomes causal self-attention.
- Set the hidden scores to $-\infty$ before the softmax. They turn into exact zeros and the rest rescales by itself.
- Keep the mask in a `register_buffer` and cut it to the current length.
- Dropout on the attention weights is a common way to fight overfitting during training, and it's turned off when using the model.

## Resources

- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY. The lower-triangular averaging trick follows this lecture.
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 3, sections on causal attention and dropout: https://www.manning.com/books/build-a-large-language-model-from-scratch; code: https://github.com/rasbt/LLMs-from-scratch
- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- PyTorch documentation, `torch.nn.functional.scaled_dot_product_attention`: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
