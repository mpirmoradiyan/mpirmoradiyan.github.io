---
title: "LSTMs: gates and a cell state that let gradients survive"
slug: lstm
date: 2025-03-07
topic: NLP & LLMs
tags: [lstm, rnn, gates, vanishing-gradients, sequence-models, pytorch, perplexity]
summary: "How an LSTM uses forget, input and output gates around a cell state that is updated by addition, so information (and gradients) survive long sequences, and how to use nn.LSTM properly for character-level text."
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

I came to LSTMs straight from the plain [RNN](/notes/recurrent-neural-networks/), and the problem was still fresh: the RNN's gradient fades away (or blows up) after a few dozen steps, so it can't connect things that are far apart. Take "The cat, which was very old and grey, sat on the mat". To get "sat" right, the model has to carry "cat" across that whole middle part. The Long Short-Term Memory (LSTM) cell is the fix. It adds a separate memory track and learned gates that decide what to keep, what to add and what to show.

## Two states instead of one

The first thing that's different: an LSTM passes two vectors from one step to the next, not one.

- the **cell state** $c_t$, a long-term memory that only gets changed by element-wise multiplication and addition;
- the **hidden state** $h_t$, the short-term output that the next layer (and the next time step) reads.

Three **gates** control what flows where. The way I think about a gate: it's a sigmoid layer, so each output is between 0 (block) and 1 (let through), one value per memory unit. Like a row of dimmer switches.

## The equations

Here $[h_{t-1}, x_t]$ is the previous hidden state glued to the current input, $\sigma$ is the sigmoid and $\odot$ is element-wise multiplication.

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

Each $W$ has shape (hidden, hidden + input). So an LSTM has four times the parameters of a plain RNN with the same hidden size. One bit of history: the 1997 LSTM had no forget gate. Gers, Schmidhuber and Cummins added it in 2000, and the version above, with the forget gate, is the one everyone uses now.

Back to the example sentence. While the middle part is read, the forget gate can stay close to 1 on the units that store "the subject is one cat". The input gate can stay closed for the unimportant words. And the output gate can let that information out when the verb comes. Nobody programs these roles, though. The gates learn them from data.

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

This is the part that made LSTMs click for me. In the RNN, every step back multiplies the gradient by the weight matrix and a tanh derivative. Now look at how the gradient flows back along the cell state instead. Since $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, the direct path gives

$$
\frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t) \quad (\text{plus smaller terms through the gates}).
$$

On this path there's no weight matrix multiplied again and again, and no tanh squashing things. Only the forget gate. When the network wants to remember, it learns $f_t \approx 1$, and the gradient flows back almost unchanged over many steps. So basically the cell state is a "highway" built from addition. The same idea shows up later as residual connections in ResNets and transformers.

A common trick is to start the forget-gate bias at a positive value (for example 1), so the cell remembers by default at the start of training.

The **GRU** (gated recurrent unit) is a lighter cousin with two gates and no separate cell state. It often works about as well with fewer parameters.

## Using `nn.LSTM` in PyTorch

`nn.LSTM` takes care of the gates and the loop over time. The main difference from `nn.RNN` is that the state is a **tuple** `(h, c)`, each of shape (num_layers, batch, hidden).

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

The `dropout` argument of `nn.LSTM` only acts between stacked layers, which is why it's set to 0 when there's just one layer. Training is the usual loop, plus gradient clipping (capping how big the gradients can get):

```python
for x, y in loader:                          # shuffled windows of 25 characters
    logits, _ = model(x)                     # fresh zero state for every batch
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
```

Here all windows have the same length. When sentences have different lengths and you pad them, the padding needs masking, or it quietly skews both training and the scores. I wrote that story up in [padding and masking in sequence tagging](/notes/padding-and-masking-in-sequence-tagging/).

## Two mistakes I made with the state

**Carrying the state across shuffled batches.** My first training loop passed the `(h, c)` from one batch into the next, after detaching it. That's the right move for truncated backpropagation through time, when each batch continues the text of the previous one. But with a shuffled `DataLoader` of independent windows, batch $k+1$ has nothing to do with batch $k$. So the carried state is just noise from some unrelated bit of text. It also crashes when the last batch is smaller than the others, because the state has the wrong batch size. For shuffled windows, start every batch from a zero state (pass `None`). For back-to-back chunks of one long text, carry the state and detach both tensors: `state = tuple(s.detach() for s in state)`.

**Validating on training data.** I made a "validation" split with `random_split` from the same windows the model trained on. Overlapping windows from the same story share almost all their characters, so the validation loss looked great but didn't really mean anything. A proper split sets aside a separate part of the text before making any windows.

## How well does it do?

On the one-sentence toy text "The cat sat on the mat and purred." a 16-unit LSTM pushed the training loss to almost zero within 200 epochs. Started with "c", it wrote "cat on the mat and purrrrrrrrrr". It learned the sentence by heart and then got stuck repeating the last letter, since nothing in training ever came after "purr". On the longer Red Riding Hood story I used the 2-layer, 128-unit model above with dropout 0.3, gradient clipping and temperature sampling at 0.5, 1.0 and 1.5. With a text that small, any LSTM will mostly memorize. A fair comparison with the plain RNN needs more text and a held-out split.

A common way to say how good a language model is: **perplexity**, the exponential of the average cross-entropy:

$$
\text{PPL} = \exp\Big(-\frac{1}{N}\sum_{t=1}^{N} \log p(x_t \mid x_{<t})\Big).
$$

The easy way to read it: roughly how many equally likely options the model is torn between at each step. A perplexity of 1 means it always gives the right character probability 1. A model that guesses uniformly over 42 characters gets exactly 42.

## Key takeaways

- An LSTM keeps a cell state $c_t$ that's updated by addition, plus a hidden state $h_t$ that the rest of the network sees.
- The forget, input and output gates are sigmoid layers that learn what to erase, what to write and what to show.
- Because the cell update is additive, gradients get a path with no repeated matrix products. That stops them vanishing when $f_t \approx 1$.
- In PyTorch the LSTM state is a tuple `(h, c)`. Only carry it between batches that continue the same text, and detach it when you do.
- Perplexity, $\exp(\text{cross-entropy})$, is the usual way to report how well a language model predicts text.

## Resources

- Sepp Hochreiter and Jürgen Schmidhuber, "Long Short-Term Memory", *Neural Computation* 9(8), 1997: https://www.bioinf.jku.at/publications/older/2604.pdf
- Felix Gers, Jürgen Schmidhuber and Fred Cummins, "Learning to Forget: Continual Prediction with LSTM", *Neural Computation*, 2000 (introduced the forget gate): https://doi.org/10.1162/089976600300015015
- Christopher Olah, "Understanding LSTM Networks" (blog post with the standard diagrams): https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Aston Zhang et al., *Dive into Deep Learning*, Section 10.1 "Long Short-Term Memory": https://d2l.ai/chapter_recurrent-modern/lstm.html
- PyTorch documentation, `torch.nn.LSTM`: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
