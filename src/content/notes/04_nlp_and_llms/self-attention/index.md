---
title: "Self-attention: queries, keys, values and the square-root scaling"
slug: self-attention
date: 2025-01-09
topic: NLP & LLMs
tags: [attention, self-attention, scaled-dot-product, transformer, query-key-value, pytorch]
summary: "How self-attention lets each token rebuild itself as a weighted average of all tokens, going step by step from plain dot products to learned queries, keys and values, and why the scores get divided by the square root of the key size."
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

Start with the problem. What a word means depends on the words around it. "Bank" means one thing next to "river" and another next to "loan". So a token's vector can't just sit there on its own. It needs a way to look at the other tokens and pull in what it needs. Self-attention is that way, and it's the heart of the transformer. It took over from RNNs because it links any two positions in one step, and it runs in parallel. I'll build it up the way it made sense to me: from the dumbest version to the one GPT uses.

## Step 0: plain averaging already shares information

Say each of $T$ tokens has an embedding vector $x_i$. The roughest way for token $i$ to learn about its context is to swap $x_i$ for the average of all the embeddings. That does mix information. But every token ends up with the exact same blend, which is useless. Attention keeps the idea of a weighted average but makes the weights **depend on the data**. So basically each token decides for itself how much to take from every other token.

## Step 1: weights from dot products

Pick one token as the **query**, say the second token $x_2$. Score every token by its dot product with the query, turn the scores into weights with a softmax, and take the weighted sum. The result $z_2$ is called the **context vector** for token 2.

$$
\omega_{2i} = x_2 \cdot x_i, \qquad
\alpha_{2i} = \frac{\exp(\omega_{2i})}{\sum_{j=1}^{T} \exp(\omega_{2j})}, \qquad
z_2 = \sum_{i=1}^{T} \alpha_{2i}\, x_i .
$$

The dot product is big when two vectors point in a similar direction. The softmax makes all the weights positive and makes them add up to 1. It also copes with negative scores, which plain "divide by the sum" doesn't. For all tokens at once, it's just three matrix operations:

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

This works, but there's nothing to train yet. A token can only pay attention to tokens that already look like it. That's way too limited.

## Step 2: give each token three roles (query, key, value)

The trainable version gives each token three different vectors, made by three learned linear maps:

- a **query** $q_i = x_i W_q$: what this token is looking for;
- a **key** $k_i = x_i W_k$: what this token has to offer, used to check if it matches a query;
- a **value** $v_i = x_i W_v$: the information this token actually hands over if it's picked.

This tripped me up at first: why three vectors and not one? The point is that "is this token relevant?" (key) and "what does it contribute?" (value) are different questions. And the query can look for something different from what the token itself contains. Karpathy's example: a vowel might send out a query like "I'm looking for consonants in the last few positions", and consonants answer through their keys.

In matrix form, with $X$ the (T, d_in) matrix of token embeddings, this gives **scaled dot-product attention**:

$$
Q = XW_q,\quad K = XW_k,\quad V = XW_v,\qquad
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

$QK^\top$ is the (T, T) matrix of scores between every query and every key. The softmax is applied row by row, and $d_k$ is the size of the key vectors.

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

`nn.Linear` without bias computes exactly `x @ W`. It's nicer than a raw `nn.Parameter(torch.rand(...))` matrix because it comes with a sensible starting initialization. I checked that copying the `nn.Linear` weights (transposed) into a raw-parameter version gives the same outputs.

## Step 3: why divide by $\sqrt{d_k}$

If the numbers in $q$ and $k$ are independent with mean 0 and variance 1, their dot product $q \cdot k = \sum_{j=1}^{d_k} q_j k_j$ is a sum of $d_k$ terms that each have variance 1. So its variance is $d_k$. A quick check with $d_k = 16$ shows it:

```python
q, k = torch.randn(4, 8, 16), torch.randn(4, 8, 16)
print((q @ k.transpose(-2, -1)).var())            # about 16
print((q @ k.transpose(-2, -1) * 16**-0.5).var()) # about 1
```

Why should we care? Big scores push the softmax towards a one-hot vector. One token grabs nearly all the weight, and the gradients for all the others become tiny. Dividing by $\sqrt{d_k}$ brings the variance back to about 1. Then the softmax starts out spread out, and training behaves, especially early on.

You can see it on one row of scores. For the scores $[4, 16, 12]$ with $d_k = 3$, the plain softmax gives $[0.000, 0.982, 0.018]$, almost a hard pick. Dividing by $\sqrt{3} \approx 1.73$ first gives $[0.001, 0.909, 0.090]$, still clear but softer. With real head sizes like 64, the difference is much bigger.

A sneaky bug from my own GPT notebook: my first `Head` class scaled by `C**-0.5`, where `C` was the embedding size (32), and not the head size (16). The model still trained, because the scale is just a constant. But it's the wrong constant: the scaling has to use the size of the vectors in the dot product, $d_k$.

## Things worth knowing

- **Every token sees every token in one step.** The path between any two positions has length 1. In an RNN, information has to pass through every step in between.
- **It runs in parallel.** All rows of $QK^\top$ come out of one matrix multiplication.
- **It costs $O(T^2)$** in time and memory, because of the (T, T) score matrix. Long contexts get expensive.
- **It doesn't know about order.** If you shuffle the input tokens, the outputs get shuffled the same way and nothing else changes. Position information has to be added to the inputs, which is what positional embeddings are for.
- **Self vs cross attention.** In self-attention, queries, keys and values all come from the same sequence. In cross-attention (for example in an encoder-decoder translator), the queries come from the decoder and the keys and values come from the encoder output.

For a language model that writes text left to right, a token mustn't see the tokens after it. A mask takes care of that, see the [causal attention note](/notes/causal-attention/). Running several attention computations side by side gives [multi-head attention](/notes/multi-head-attention/).

## Key takeaways

- Self-attention replaces each token's vector with a weighted average of the value vectors of all tokens, and the weights come from the data.
- The weights come from query-key dot products put through a softmax, row by row.
- Separate learned projections for queries, keys and values let a token decide what it looks for, what it advertises and what it hands over.
- Scores are divided by $\sqrt{d_k}$ so their variance stays near 1 and the softmax doesn't get stuck near one-hot.
- Attention connects all positions in one parallel step, but it costs $O(T^2)$ and needs position information added separately.

## Resources

- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 3 "Coding attention mechanisms": https://www.manning.com/books/build-a-large-language-model-from-scratch. The six-token example and the step-by-step build-up follow this chapter; code: https://github.com/rasbt/LLMs-from-scratch
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY. Source of the variance experiment and the query/key intuition.
