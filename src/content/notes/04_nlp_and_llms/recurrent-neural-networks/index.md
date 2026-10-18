---
title: "Recurrent neural networks: a hidden state that carries the past"
slug: recurrent-neural-networks
date: 2025-03-07
topic: NLP & LLMs
tags: [rnn, sequence-models, bptt, vanishing-gradients, character-level, temperature, pytorch]
summary: "How a recurrent neural network reads a sequence one step at a time with shared weights, how it is trained with backpropagation through time, and why its gradients vanish or explode over long sequences."
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

An MLP sees a fixed number of inputs. Text, speech and time series do not come in fixed sizes, and order matters: "dog bites man" is not "man bites dog". A recurrent neural network (RNN) handles this by reading one element at a time and keeping a **hidden state**, a vector that summarizes everything seen so far. RNNs ruled sequence modelling until attention replaced them, and the problems they ran into explain why LSTMs and transformers look the way they do.

## The RNN cell

At each time step $t$ the cell takes two inputs, the current element $x_t$ (for example a character embedding) and the previous hidden state $h_{t-1}$, and produces a new hidden state:

$$
h_t = \tanh\left(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b_h\right)
$$

- $W_{xh}$ maps the input into the hidden space (input-to-hidden weights).
- $W_{hh}$ maps the previous hidden state into the new one (hidden-to-hidden weights). This is the "recurrent" part.
- $b_h$ is a bias and $\tanh$ keeps the values between -1 and 1.

To make a prediction, for example the next character, add an output layer on top of the hidden state:

$$
y_t = W_{hy}\, h_t + b_y, \qquad p_t = \operatorname{softmax}(y_t).
$$

The initial state $h_0$ is usually a vector of zeros.

## Unrolling over time

Processing "The cat sat" means applying the same cell three times: "The" with $h_0$ gives $h_1$, "cat" with $h_1$ gives $h_2$, "sat" with $h_2$ gives $h_3$. Drawn out step by step, the RNN looks like a deep feed-forward network with one layer per time step, except that **every layer shares the same weights**. Weight sharing is what lets one small set of parameters handle sequences of any length.

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

The loss is summed (or averaged) over all time steps, $L = \sum_t L_t$, where $L_t$ is the cross-entropy of the prediction at step $t$. Backpropagation on the unrolled graph is called **backpropagation through time** (BPTT). Because the weights are shared, the gradient for $W_{hh}$ is the sum of its contributions from every time step.

The trouble shows up in how the gradient travels back through the hidden states. Going from $h_t$ to $h_{t-1}$ multiplies the gradient by the Jacobian

$$
\frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}\!\left(1 - h_t^2\right) W_{hh}.
$$

To reach a step $k$ positions back, the gradient is multiplied by $k$ such matrices. If their size (roughly, the largest singular value of $W_{hh}$ scaled by the tanh derivative) is below 1, the product shrinks exponentially: the **vanishing gradient** problem. The network then cannot learn that a word early in the sentence matters for a prediction much later. If it is above 1, the product grows exponentially: the **exploding gradient** problem, which makes training unstable.

Two practical fixes are standard:

- **Gradient clipping**: rescale the gradient whenever its norm exceeds a threshold (`torch.nn.utils.clip_grad_norm_(model.parameters(), 5)`). This handles explosions but not vanishing.
- **Truncated BPTT**: process a long text in chunks and call `hidden = hidden.detach()` between chunks. The hidden state still carries information forward, but gradients stop at the chunk boundary. This bounds the cost of a backward pass.

Vanishing gradients need an architectural fix, which is what the [LSTM](/notes/lstm/) provides.

## A character-level RNN in PyTorch

PyTorch's `nn.RNN` implements the cell and the loop over time. A small character model needs an embedding, the RNN and a linear output layer.

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

With `batch_first=True` the tensors are (batch, time, features). The returned `h` has shape (num_layers, batch, hidden) and can be passed back in to continue a sequence.

I trained this on the Little Red Riding Hood story (about 1,200 characters, 42 distinct symbols), with windows of 25 characters where the target is the input shifted by one. After 10 epochs the samples reproduced fragments of the story, such as "Red Riding Hood", "As she walked the wolf", mixed with invented words. That is memorization rather than generalization, as expected for a dataset this small.

## Sampling with temperature

To generate text, feed a seed, sample a character from the softmax, feed it back in, and repeat. A **temperature** $\tau$ rescales the logits before the softmax:

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}.
$$

$\tau < 1$ sharpens the distribution (safer, more repetitive text); $\tau > 1$ flattens it (more varied, more mistakes); $\tau \to 0$ approaches always picking the most likely character.

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

Reading the seed once and then feeding one character at a time with the carried hidden state is the natural way to use an RNN. My first version re-ran the whole padded window at every step, which works but throws away the main advantage of recurrence.

## A note on evaluation

My notebook reported a "test" accuracy of 0.86 for next-character prediction. The test text was the last 10% of the same story the model had been trained on, so this measured memorization, not generalization. A fair test set must be text the model never saw during training.

## Where RNNs fall short

- **Long-range dependencies** are hard to learn because of vanishing gradients.
- **Sequential computation**: step $t$ needs $h_{t-1}$, so the time steps cannot be processed in parallel on a GPU.
- **A fixed-size bottleneck**: everything the model knows about the past must fit into one hidden vector.

LSTMs and GRUs address the first point with gates. Attention, and finally the transformer, address all three.

## Key takeaways

- An RNN updates a hidden state $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$ with the same weights at every time step.
- Training unrolls the network over time (BPTT) and sums the gradient contributions of each step.
- Repeated multiplication by the recurrent Jacobian makes gradients vanish or explode over long sequences.
- Gradient clipping tames explosions; detaching the hidden state between chunks gives truncated BPTT.
- Temperature controls how adventurous sampling is; always evaluate on text the model has not seen.

## Resources

- Andrej Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" (blog post): http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Christopher Olah, "Understanding LSTM Networks" (blog post, includes the unrolled RNN diagram): https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Razvan Pascanu, Tomas Mikolov and Yoshua Bengio, "On the difficulty of training Recurrent Neural Networks", 2013: https://arxiv.org/abs/1211.5063
- Aston Zhang et al., *Dive into Deep Learning*, Chapter 9 "Recurrent Neural Networks": https://d2l.ai/chapter_recurrent-neural-networks/index.html
- PyTorch documentation, `torch.nn.RNN`: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
