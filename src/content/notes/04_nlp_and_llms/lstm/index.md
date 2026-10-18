---
title: "LSTMs: gates and a cell state that let gradients survive"
slug: lstm
date: 2025-03-07
topic: NLP & LLMs
tags: [lstm, rnn, gates, vanishing-gradients, sequence-models, pytorch, perplexity]
summary: "How an LSTM cell uses forget, input and output gates around an additive cell state to keep information (and gradients) alive over long sequences, and how to use nn.LSTM correctly for character-level text."
sources:
  - title: "Long Short-Term Memory"
    author: "Sepp Hochreiter, Jürgen Schmidhuber (Neural Computation, 1997)"
    url: "https://www.bioinf.jku.at/publications/older/2604.pdf"
    type: paper
  - title: "Understanding LSTM Networks"
    author: "Christopher Olah"
    url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
    type: docs
  - title: "Learning to Forget: Continual Prediction with LSTM"
    author: "Felix Gers, Jürgen Schmidhuber, Fred Cummins (2000)"
    url: "https://doi.org/10.1162/089976600300015015"
    type: paper
  - title: "Dive into Deep Learning, Section 10.1: Long Short-Term Memory"
    author: "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola"
    url: "https://d2l.ai/chapter_recurrent-modern/lstm.html"
    type: book


status: draft
---

A plain RNN squeezes its whole memory through a tanh and a weight matrix at every step, so gradients fade (or blow up) after a few dozen steps. In a sentence like "The cat, which was very old and grey, sat on the mat", the model has to carry "cat" across the clause to get "sat" right. The Long Short-Term Memory (LSTM) cell, introduced in 1997, solves this with a separate memory track and learned gates that decide what to keep, what to add and what to show.

## Two states instead of one

An LSTM passes two vectors from step to step:

- the **cell state** $c_t$, a long-term memory that is changed only by element-wise multiplication and addition;
- the **hidden state** $h_t$, the short-term output that the next layer (and the next time step) reads.

Three **gates** control the flow. Each gate is a sigmoid layer, so its outputs lie between 0 (block) and 1 (let through), one value per memory unit.

## The equations

Let $[h_{t-1}, x_t]$ be the previous hidden state concatenated with the current input, $\sigma$ the sigmoid and $\odot$ element-wise multiplication.

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{forget gate: what to keep from } c_{t-1}\\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{input gate: how much new content to write}\\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) && \text{candidate content}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t && \text{new cell state}\\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{output gate: what to expose}\\
h_t &= o_t \odot \tanh(c_t) && \text{new hidden state}
\end{aligned}
$$

Each $W$ has shape (hidden, hidden + input). An LSTM therefore has four times the parameters of a plain RNN with the same hidden size.

In the example sentence, the forget gate can stay close to 1 on the units that store "the subject is singular and is a cat" while the clause is read, the input gate can stay closed for the unimportant words, and the output gate can expose that information when the verb arrives. These roles are not programmed; the gates learn them from data.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lstm_step(x, h_prev, c_prev, P):
    z = np.concatenate([h_prev, x])
    f = sigmoid(P["Wf"] @ z + P["bf"])
    i = sigmoid(P["Wi"] @ z + P["bi"])
    c_tilde = np.tanh(P["Wc"] @ z + P["bc"])
    o = sigmoid(P["Wo"] @ z + P["bo"])
    c = f * c_prev + i * c_tilde
    h = o * np.tanh(c)
    return h, c
```

## Why the cell state fixes vanishing gradients

Look at how the gradient flows back along the cell state. Since $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, the direct path gives

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) \quad (\text{plus smaller terms through the gates}).
$$

There is no repeated multiplication by a weight matrix and no squashing tanh on this path, only the forget gate. When the network wants to remember, it learns $f_t \approx 1$, and the gradient passes back almost unchanged over many steps. This additive "highway" is the same idea that later reappears as residual connections in ResNets and transformers.

A common practical trick is to initialize the forget-gate bias to a positive value (for example 1), so the cell starts out remembering by default.

The **GRU** (gated recurrent unit) is a lighter variant with two gates and no separate cell state. It often performs similarly with fewer parameters.

## Using `nn.LSTM` in PyTorch

`nn.LSTM` handles the gates and the loop over time. The main difference from `nn.RNN` is that the state is a **tuple** `(h, c)`, each of shape (num_layers, batch, hidden).

```python
import torch, torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):        # x: (B, T)
        out, state = self.lstm(self.embed(x), state)
        return self.head(self.drop(out)), state   # (B, T, vocab)
```

The `dropout` argument of `nn.LSTM` applies between stacked layers only, which is why it is set to 0 for a single layer. Training follows the usual pattern, with gradient clipping:

```python
for x, y in loader:                          # shuffled windows of 25 characters
    logits, _ = model(x)                     # fresh zero state for every batch
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
```

## Two mistakes I made with the state

**Carrying the state across shuffled batches.** My first training loop passed the `(h, c)` from one batch into the next, after detaching it. That is the right thing for truncated backpropagation through time when consecutive batches continue the same text. With a shuffled `DataLoader` of independent windows, though, batch $k+1$ has nothing to do with batch $k$, so the carried state is just noise from unrelated text. It also crashes when the last batch is smaller than the others, because the state has the wrong batch size. For shuffled windows, start every batch from a zero state (pass `None`). For contiguous chunks of one long text, carry the state and detach both tensors: `state = tuple(s.detach() for s in state)`.

**Validating on training data.** I created a "validation" split with `random_split` from the same windows the model had trained on. Overlapping windows of the same story share almost all their characters, so the validation loss looked excellent but said little. A proper split holds out a separate part of the text before creating any windows.

## How well does it do?

On the one-sentence toy text "The cat sat on the mat and purred." a 16-unit LSTM drove the training loss to almost zero within 200 epochs and, seeded with "c", produced "cat on the mat and purrrrrrrrrr". It memorized the sentence perfectly and then got stuck repeating the last letter, since nothing in training ever followed "purr". On the longer Red Riding Hood story I used the 2-layer, 128-unit model above with dropout 0.3, gradient clipping and temperature sampling at 0.5, 1.0 and 1.5. With a text that small, any LSTM will mostly memorize; a fair comparison with the plain RNN needs a larger corpus and a held-out split.

A common way to report language-model quality is **perplexity**, the exponential of the average cross-entropy:

$$
\text{PPL} = \exp\Big(-\frac{1}{N}\sum_{t=1}^{N} \log p(x_t \mid x_{<t})\Big).
$$

It can be read as the effective number of equally likely choices the model hesitates between at each step. A perplexity of 1 means certainty; a uniform guess over 42 characters gives 42.

## Key takeaways

- An LSTM keeps a cell state $c_t$ that is updated additively, plus a hidden state $h_t$ that is exposed to the rest of the network.
- Forget, input and output gates are sigmoid layers that learn what to erase, what to write and what to show.
- The additive cell update gives gradients a path without repeated matrix products, which avoids vanishing gradients when $f_t \approx 1$.
- In PyTorch the LSTM state is a tuple `(h, c)`. Only carry it between batches that continue the same text, and detach it when you do.
- Perplexity, $\exp(\text{cross-entropy})$, is the standard way to report how well a language model predicts text.

## Resources

- Sepp Hochreiter and Jürgen Schmidhuber, "Long Short-Term Memory", *Neural Computation* 9(8), 1997: https://www.bioinf.jku.at/publications/older/2604.pdf
- Felix Gers, Jürgen Schmidhuber and Fred Cummins, "Learning to Forget: Continual Prediction with LSTM", *Neural Computation*, 2000 (introduced the forget gate): https://doi.org/10.1162/089976600300015015
- Christopher Olah, "Understanding LSTM Networks" (blog post with the standard diagrams): https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Aston Zhang et al., *Dive into Deep Learning*, Section 10.1 "Long Short-Term Memory": https://d2l.ai/chapter_recurrent-modern/lstm.html
- PyTorch documentation, `torch.nn.LSTM`: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
