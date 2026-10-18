---
title: "Multi-head attention: several small attentions in parallel"
slug: multi-head-attention
date: 2025-01-09
topic: NLP & LLMs
tags: [attention, multi-head-attention, transformer, gpt, tensor-reshaping, pytorch]
summary: "Why transformers split attention into several heads, how the simple 'list of heads' version maps onto the efficient single-matrix version with view and transpose, and what the output projection is for."
sources:
  - title: "Attention Is All You Need"
    author: "Ashish Vaswani et al. (2017)"
    url: "https://arxiv.org/abs/1706.03762"
    type: paper
  - title: "Let's build GPT: from scratch, in code, spelled out"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=kCc8FmEb1nY"
    type: video
  - title: "Build a Large Language Model (From Scratch), Chapter 3: Coding attention mechanisms"
    author: "Sebastian Raschka (Manning, 2024)"
    url: "https://www.manning.com/books/build-a-large-language-model-from-scratch"
    type: book


status: draft
---

A single attention head computes one set of weights per token: one opinion about which other tokens matter. Language has many kinds of relationships at once. A token may need its grammatical subject, the previous word, a matching opening bracket and the topic of the paragraph, all at the same time. **Multi-head attention** runs several independent attention computations side by side, each with its own query, key and value projections, and concatenates the results. This note builds it twice: the readable way and the way real models implement it.

## The idea

Each head $i$ has its own projection matrices and performs ordinary scaled dot-product attention (see the [self-attention note](/notes/self-attention/)):

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{(XW_q^{i})(XW_k^{i})^\top}{\sqrt{d_h}} + M\right) XW_v^{i}
$$

Here $X$ is the (T, d_model) input, $W_q^i, W_k^i, W_v^i$ have shape (d_model, $d_h$), $d_h$ is the head size and $M$ is the causal mask from the [causal attention note](/notes/causal-attention/). The heads are concatenated and mixed by an output projection $W_o$:

$$
\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_o .
$$

Usually $d_h = d_\text{model} / h$. With $h$ heads of size $d_\text{model}/h$, the total amount of computation is about the same as one big head of size $d_\text{model}$, but the model gets $h$ separate attention patterns instead of one. GPT-2 small, for example, uses 12 heads of size 64 for a model width of 768.

## Version 1: a list of heads

The most direct implementation creates several single-head modules and concatenates their outputs along the last dimension.

```python
import torch, torch.nn as nn

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_head, context_length, n_heads, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(d_in, d_head, context_length, dropout)
             for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * d_head, n_heads * d_head)

    def forward(self, x):                                   # (B, T, d_in)
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads*d_head)
        return self.proj(out)
```

`CausalSelfAttention` is the single-head module from the causal attention note. `nn.ModuleList` (rather than a plain Python list) makes sure the heads' parameters are registered with the parent module, so the optimizer sees them and `.to(device)` moves them.

In my small Shakespeare GPT, swapping one head of size 32 for four heads of size 8 lowered the validation loss after 5,000 steps from about **2.40** to about **2.29**, with the same embedding size.

## What the output projection is for

My first multi-head version had no projection: it just concatenated the heads. Adding the linear layer `proj` lets the model mix information across heads; without it, each slice of the output vector would only ever contain the result of one head. In a transformer block the output of attention is also added back to the residual stream (`x = x + attn(x)`), and the projection maps the concatenated heads into that stream. In GPT-2 this layer is called `c_proj`.

## Version 2: one big matrix and a reshape

Looping over heads launches many small matrix multiplications. The efficient version computes the queries, keys and values for all heads in one go, then **reshapes** so the heads become an extra batch dimension.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # all heads, all of q,k,v
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):                                   # (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)               # each (B, T, C)
        # split C into h heads of size dh, and move heads next to batch
        q, k, v = [t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        att = q @ k.transpose(-2, -1) / self.dh ** 0.5      # (B, h, T, T)
        att = att.masked_fill(self.mask[:T, :T], float("-inf"))
        att = self.dropout(torch.softmax(att, dim=-1))
        y = att @ v                                         # (B, h, T, dh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # concatenate heads
        return self.proj(y)
```

The shape bookkeeping is the whole trick:

1. `self.qkv(x)` gives (B, T, 3C). Splitting along the last dimension gives three (B, T, C) tensors.
2. `view(B, T, h, dh)` cuts each token's C numbers into $h$ chunks of size $d_h$. Chunk $i$ plays the role of head $i$'s query (or key, or value).
3. `transpose(1, 2)` gives (B, h, T, dh). Matrix multiplication treats all leading dimensions as batch dimensions, so `q @ k.transpose(-2, -1)` computes a separate (T, T) score matrix for every sequence and every head, in one call.
4. After attention, `transpose(1, 2)` brings the heads back next to the channel dimension, and `view(B, T, C)` glues the heads together, which is exactly the concatenation of version 1.

`contiguous()` is needed because `transpose` only changes strides, not the memory layout, and `view` requires memory that is laid out in order. `reshape` would do the copy automatically.

A big (C, C) query matrix whose output is cut into $h$ slices is mathematically the same as $h$ separate (C, $d_h$) query matrices stacked side by side. So the two versions represent exactly the same family of functions; only the speed differs. GPT-2 goes one step further and fuses $W_q$, $W_k$ and $W_v$ into one (768, 2304) layer called `c_attn`, which is what `self.qkv` above does.

## Trying it on a toy batch

In my notebook I used a variant of the class above that takes separate input and output sizes (`d_in` for the query/key/value input, `d_out` for the concatenated heads), so the projections go from 3 to 18 dimensions. With 6 tokens of dimension 3, a batch of 5 identical sequences, `d_out = 18` and 2 heads (so each head has size 9), the output has shape (5, 6, 18): one 18-dimensional context vector per token per sequence. Changing the number of heads changes how the 18 dimensions are split, not the output shape. The model width must be divisible by the number of heads, hence the `assert`.

## Key takeaways

- Multi-head attention runs $h$ independent attention heads, each with its own query, key and value projections, and concatenates their outputs.
- With head size $d_\text{model}/h$, it costs about the same as one large head but can learn several attention patterns at once.
- An output projection mixes the heads and maps the result back into the model's residual stream.
- The efficient implementation uses one projection for all heads and turns heads into a batch dimension with `view` and `transpose`.
- The model width must be divisible by the number of heads; GPT-2 small uses 12 heads of size 64.

## Resources

- Ashish Vaswani et al., "Attention Is All You Need", 2017 (Section 3.2.2, multi-head attention): https://arxiv.org/abs/1706.03762
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY
- Andrej Karpathy, nanoGPT (the batched `CausalSelfAttention` design used by GPT-2 style code): https://github.com/karpathy/nanoGPT
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 3, sections on multi-head attention: https://www.manning.com/books/build-a-large-language-model-from-scratch; code: https://github.com/rasbt/LLMs-from-scratch
