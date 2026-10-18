---
title: "Recurrent neural networks: a hidden state that carries the past"
slug: recurrent-neural-networks
date: 2025-03-07
topic: NLP & LLMs
tags: [rnn, sequence-models, bptt, vanishing-gradients, character-level, temperature, pytorch]
summary: "How a recurrent neural network reads a sequence one step at a time with the same weights, how you train it with backpropagation through time, and why its gradients shrink or blow up over long sequences."
sources:
  - title: "The Unreasonable Effectiveness of Recurrent Neural Networks"
    author: "Andrej Karpathy"
    url: "http://karpathy.github.io/2015/05/21/rnn-effectiveness/"
    type: docs
  - title: "Understanding LSTM Networks"
    author: "Christopher Olah"
    url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
    type: docs
  - title: "On the difficulty of training Recurrent Neural Networks"
    author: "Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (2013)"
    url: "https://arxiv.org/abs/1211.5063"
    type: paper
  - title: "Dive into Deep Learning, Chapter 9: Recurrent Neural Networks"
    author: "Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola"
    url: "https://d2l.ai/chapter_recurrent-neural-networks/index.html"
    type: book


status: draft
---

An MLP always takes the same number of inputs. But text, speech and time series don't come in a fixed size, and order matters: "dog bites man" is not "man bites dog". A recurrent neural network (RNN) deals with this by reading one item at a time and keeping a **hidden state**, a vector that sums up everything it has seen so far. RNNs were the main tool for sequences until attention took over. For me the most useful part was seeing the problems they ran into, because those explain why LSTMs and transformers look the way they do.

## The RNN cell

At each time step $t$ the cell takes two things, the current item $x_t$ (for example a character embedding) and the previous hidden state $h_{t-1}$, and makes a new hidden state:

$$
h_t = \tanh\left(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h\right)
$$

- $W_{xh}$ maps the input into the hidden space (input-to-hidden weights).
- $W_{hh}$ maps the previous hidden state into the new one (hidden-to-hidden weights). This is the "recurrent" part.
- $b_h$ is a bias and $\tanh$ keeps the values between -1 and 1.

To predict something, like the next character, put an output layer on top of the hidden state:

$$
y_t = W_{hy}\, h_t + b_y, \qquad p_t = \operatorname{softmax}(y_t).
$$

The initial state $h_0$ is usually a vector of zeros.

## Unrolling over time

Reading "The cat sat" means using the same cell three times: "The" with $h_0$ gives $h_1$, "cat" with $h_1$ gives $h_2$, "sat" with $h_2$ gives $h_3$. If you draw it out step by step, the RNN looks like a deep feed-forward network with one layer per time step. The difference is that **every layer uses the same weights**. That sharing is what lets one small set of parameters handle sequences of any length.

```python
import numpy as np

hidden, emb = 4, 3
rng = np.random.default_rng(0)
W_xh = rng.normal(0, 0.01, (hidden, emb))
W_hh = rng.normal(0, 0.01, (hidden, hidden))
b_h = np.zeros(hidden)

sequence = [np.array([0.1, 0.2, 0.3]),   # "The"
            np.array([0.4, 0.5, 0.6]),   # "cat"
            np.array([0.7, 0.8, 0.9])]   # "sat"
h = np.zeros(hidden)
for x in sequence:
    h = np.tanh(W_hh @ h + W_xh @ x + b_h)   # same weights at every step
print(h)
```

## Training: backpropagation through time

The loss is added up (or averaged) over all time steps, $L = \sum_t L_t$, where $L_t$ is the cross-entropy of the prediction at step $t$. Doing backprop on the unrolled graph has its own name: **backpropagation through time** (BPTT). Since the weights are shared, the gradient for $W_{hh}$ is just the sum of what each time step contributes.

The trouble is in how the gradient travels back through the hidden states. Each step back from $h_t$ to $h_{t-1}$ multiplies the gradient by the Jacobian (the matrix of partial derivatives)

$$
\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}\!\left(1 - h_t^2\right) W_{hh}.
$$

To reach a step $k$ positions back, the gradient gets multiplied by $k$ of these matrices. So basically it's like multiplying a number by itself over and over. If their size (roughly, the largest singular value of $W_{hh}$ times the tanh derivative) is below 1, the product shrinks very fast: the **vanishing gradient** problem. Then the network can't learn that a word early in the sentence matters for a prediction much later. If it's above 1, the product grows very fast: the **exploding gradient** problem, which makes training unstable.

There are two usual fixes:

- **Gradient clipping**: shrink the gradient whenever its size goes over some limit (`torch.nn.utils.clip_grad_norm_(model.parameters(), 5)`). This fixes explosions but not vanishing.
- **Truncated BPTT**: go through a long text in chunks and call `hidden = hidden.detach()` between chunks. The hidden state still carries information forward, but gradients stop at the chunk border. That keeps each backward pass cheap.

Vanishing gradients need a change to the network itself, and that's what the [LSTM](/notes/lstm/) does.

## A character-level RNN in PyTorch

PyTorch's `nn.RNN` does the cell and the loop over time for you. A small character model just needs an embedding, the RNN and a linear output layer.

```python
import torch, torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):          # x: (B, T) character ids
        out, h = self.rnn(self.embed(x), h)  # out: (B, T, hidden)
        return self.head(out), h             # logits: (B, T, vocab)
```

With `batch_first=True` the tensors are (batch, time, features). The returned `h` has shape (num_layers, batch, hidden), and you can pass it back in to keep going with a sequence.

I trained this on the Little Red Riding Hood story (about 1,200 characters, 42 different symbols), with windows of 25 characters where the target is the input shifted by one. After 10 epochs the samples repeated bits of the story, like "Red Riding Hood" and "As she walked the wolf", mixed with made-up words. That's memorizing, not really learning the language, which is what you'd expect from so little text.

## Sampling with temperature

To generate text: feed in a seed, sample a character from the softmax, feed it back in, and repeat. A **temperature** $\tau$ rescales the logits (the raw scores) before the softmax:

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}.
$$

$\tau < 1$ makes the distribution peakier (safer, more repetitive text). $\tau > 1$ flattens it (more variety, more mistakes). As $\tau \to 0$ it gets closer and closer to always picking the most likely character.

```python
@torch.no_grad()
def generate(model, start_ids, length, temperature=1.0):
    ids = list(start_ids)
    logits, h = model(torch.tensor([ids]))          # read the whole seed once
    for _ in range(length):
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        ids.append(nxt)
        logits, h = model(torch.tensor([[nxt]]), h)  # then one char at a time
    return ids
```

Reading the seed once and then feeding one character at a time, carrying the hidden state along, is how an RNN is meant to be used. My first version re-ran the whole padded window at every step. It worked, but it threw away the whole point of having a hidden state.

## A note on evaluation

My notebook reported a "test" accuracy of 0.86 for next-character prediction. Looking back, that number doesn't mean much: the test text was the last 10% of the same story the model was trained on, so it measured memorizing, not generalizing. A fair test set has to be text the model never saw during training.

## Where RNNs fall short

- **Long-range connections** (a word that matters many steps later) are hard to learn because of vanishing gradients.
- **One step at a time**: step $t$ needs $h_{t-1}$, so you can't run the time steps in parallel on a GPU.
- **A fixed-size bottleneck**: everything the model knows about the past has to fit into one hidden vector.

LSTMs and GRUs fix the first point with gates. Attention, and in the end the transformer, fix all three.

## Key takeaways

- An RNN updates a hidden state $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$ with the same weights at every time step.
- Training unrolls the network over time (BPTT) and adds up the gradient from each step.
- Multiplying by the recurrent Jacobian over and over makes gradients vanish or explode on long sequences.
- Gradient clipping handles explosions. Detaching the hidden state between chunks gives truncated BPTT.
- Temperature controls how adventurous sampling is. And always test on text the model hasn't seen.

## Resources

- Andrej Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" (blog post): http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Christopher Olah, "Understanding LSTM Networks" (blog post, includes the unrolled RNN diagram): https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Razvan Pascanu, Tomas Mikolov and Yoshua Bengio, "On the difficulty of training Recurrent Neural Networks", 2013: https://arxiv.org/abs/1211.5063
- Aston Zhang et al., *Dive into Deep Learning*, Chapter 9 "Recurrent Neural Networks": https://d2l.ai/chapter_recurrent-neural-networks/index.html
- PyTorch documentation, `torch.nn.RNN`: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
