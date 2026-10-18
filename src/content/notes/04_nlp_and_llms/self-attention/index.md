---
title: "Self-attention: queries, keys, values and the square-root scaling"
slug: self-attention
date: 2025-01-09
topic: NLP & LLMs
tags: [attention, self-attention, scaled-dot-product, transformer, query-key-value, pytorch]
summary: "How self-attention lets every token build a new representation as a data-dependent weighted average of all tokens, step by step from plain dot products to trainable query, key and value projections, and why the scores are divided by the square root of the key size."
sources:
  - title: "Attention Is All You Need"
    author: "Ashish Vaswani et al. (2017)"
    url: "https://arxiv.org/abs/1706.03762"
    type: paper
  - title: "Build a Large Language Model (From Scratch), Chapter 3: Coding attention mechanisms"
    author: "Sebastian Raschka (Manning, 2024)"
    url: "https://www.manning.com/books/build-a-large-language-model-from-scratch"
    type: book
  - title: "Let's build GPT: from scratch, in code, spelled out"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=kCc8FmEb1nY"
    type: video


status: draft
---

In a sentence, the meaning of a word depends on the other words around it. "Bank" means something different next to "river" than next to "loan". Self-attention is the mechanism that lets each token look at every other token in the sequence and pull in the information it needs. It is the core operation of the transformer, and it replaced recurrence because it connects any two positions in one step and runs in parallel. This note builds it up from the simplest version to the one used in GPT.

## Step 0: averaging is already a form of communication

Suppose each of $T$ tokens has an embedding vector $x_i$. The crudest way for token $i$ to learn about its context is to replace $x_i$ by the average of all the embeddings. That mixes information, but every token gets the same, uniform blend. Attention keeps the weighted-average idea and makes the weights **depend on the data**: each token decides for itself how much to take from every other token.

## Step 1: weights from dot products

Take one token as the **query**, for example the second token $x_2$. Score every token by its dot product with the query, turn the scores into weights with a softmax, and take the weighted sum. The result $z_2$ is called the **context vector** for token 2.

$$
\omega_{2i} = x_2 \cdot x_i, \qquad
\alpha_{2i} = \frac{\exp(\omega_{2i})}{\sum_{j=1}^{T} \exp(\omega_{2j})}, \qquad
z_2 = \sum_{i=1}^{T} \alpha_{2i}\, x_i .
$$

The dot product is large when two vectors point in a similar direction. The softmax makes all weights positive and makes them sum to 1, and it handles negative scores, which a plain "divide by the sum" normalization does not. For all tokens at once this is three matrix operations:

```python
import torch
x = torch.tensor([[0.43, 0.15, 0.89],   # Your
                  [0.55, 0.87, 0.66],   # journey
                  [0.57, 0.85, 0.64],   # starts
                  [0.22, 0.58, 0.33],   # with
                  [0.77, 0.25, 0.10],   # one
                  [0.05, 0.80, 0.55]])  # step
scores = x @ x.T                        # (6, 6): every pair of tokens
weights = torch.softmax(scores, dim=-1) # each row sums to 1
context = weights @ x                   # (6, 3)
```

There are no trainable parameters yet. A token can only attend to tokens that already look like it, which is far too limited.

## Step 2: separate roles with queries, keys and values

The trainable version gives each token three different vectors, produced by three learned linear maps:

- a **query** $q_i = x_i W_q$: what this token is looking for;
- a **key** $k_i = x_i W_k$: what this token offers, used to decide whether it matches a query;
- a **value** $v_i = x_i W_v$: the information this token actually passes on if it is chosen.

Splitting the roles matters. Whether a token is relevant (key) and what it contributes (value) are separate questions, and the query can look for something different from what the token itself contains. Karpathy's example: a vowel at some position might emit a query like "I am looking for consonants in the previous few positions", and consonants respond through their keys.

In matrix form, with $X$ the (T, d_in) matrix of token embeddings, this gives **scaled dot-product attention**:

$$
Q = XW_q,\quad K = XW_k,\quad V = XW_v,\qquad
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

$QK^\top$ is the (T, T) matrix of scores between every query and every key, the softmax is applied row by row, and $d_k$ is the size of the key vectors.

```python
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=False)
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):                        # x: (T, d_in)
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = q @ k.T                         # (T, T)
        weights = torch.softmax(scores / k.shape[-1] ** 0.5, dim=-1)
        return weights @ v                       # (T, d_out)

torch.manual_seed(123)
print(SelfAttention(3, 2)(x).shape)              # torch.Size([6, 2])
```

`nn.Linear` without bias computes exactly `x @ W`. It is preferable to a raw `nn.Parameter(torch.rand(...))` matrix because it comes with a sensible default initialization. I checked that copying the `nn.Linear` weights (transposed) into a raw-parameter version gives identical outputs.

## Step 3: why divide by $\sqrt{d_k}$

If the components of $q$ and $k$ are independent with mean 0 and variance 1, their dot product $q \cdot k = \sum_{j=1}^{d_k} q_j k_j$ is a sum of $d_k$ terms with variance 1, so its variance is $d_k$. A quick experiment with $d_k = 16$ confirms it:

```python
q, k = torch.randn(4, 8, 16), torch.randn(4, 8, 16)
print((q @ k.transpose(-2, -1)).var())            # about 16
print((q @ k.transpose(-2, -1) * 16**-0.5).var()) # about 1
```

Large scores push the softmax towards a one-hot vector: one token gets almost all the weight, and the gradients for all the others become tiny. Dividing by $\sqrt{d_k}$ brings the variance back to about 1, so the softmax starts diffuse and training stays well-behaved, especially at initialization.

The effect is easy to see on a single row of scores. For the scores $[4, 16, 12]$ with $d_k = 3$, the unscaled softmax gives $[0.000, 0.982, 0.018]$, almost a hard choice. Dividing by $\sqrt{3} \approx 1.73$ first gives $[0.001, 0.909, 0.090]$, still decisive but softer. With realistic head sizes such as 64, the difference is much larger.

One subtle bug from my own GPT notebook: my first `Head` class scaled by `C**-0.5`, where `C` was the embedding size (32), instead of the head size (16). The model still trained, because the scale is just a constant, but it is the wrong constant: the scaling must use the dimension of the vectors in the dot product, $d_k$.

## Properties worth knowing

- **Every token sees every token in one step.** The path between any two positions has length 1, unlike an RNN where information must pass through every intermediate step.
- **It is parallel.** All rows of $QK^\top$ are computed in one matrix multiplication.
- **It costs $O(T^2)$** in time and memory, because of the (T, T) score matrix. Long contexts are expensive.
- **It ignores order.** If you permute the input tokens, the outputs are permuted in the same way and otherwise unchanged. Position information must be added to the inputs, which is the job of positional embeddings.
- **Self vs cross attention.** In self-attention, queries, keys and values all come from the same sequence. In cross-attention (for example in an encoder-decoder translator), the queries come from the decoder and the keys and values from the encoder output.

For a language model that generates text left to right, each token must not see the tokens after it. That is handled by a mask, covered in the [causal attention note](/notes/causal-attention/). Running several attention computations in parallel gives [multi-head attention](/notes/multi-head-attention/).

## Key takeaways

- Self-attention replaces each token's vector by a weighted average of value vectors from all tokens, with weights computed from the data.
- Weights come from query-key dot products passed through a row-wise softmax.
- Separate learned projections for queries, keys and values let a token decide what it looks for, what it advertises and what it passes on.
- Scores are divided by $\sqrt{d_k}$ so their variance stays near 1 and the softmax does not saturate.
- Attention connects all positions in one parallel step but costs $O(T^2)$ and needs positional information added separately.

## Resources

- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 3 "Coding attention mechanisms": https://www.manning.com/books/build-a-large-language-model-from-scratch. The six-token example and the step-by-step build-up follow this chapter; code: https://github.com/rasbt/LLMs-from-scratch
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY. Source of the variance experiment and the query/key intuition.
