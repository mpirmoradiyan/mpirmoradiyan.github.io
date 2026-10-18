---
title: "Training a tiny neural network from scratch (and the zero-grad bug)"
slug: training-a-tiny-mlp-from-scratch
date: 2024-12-10
topic: Deep Learning Foundations
tags: [mlp, gradient-descent, training-loop, micrograd, zero-grad]
summary: "How neurons, layers and an MLP are built on top of a scalar autograd engine, how the forward–backward–update loop trains it, and why gradients must be reset every step."
sources:
  - title: "The spelled-out intro to neural networks and backpropagation: building micrograd"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=VMj-3S1tku0"
    type: video
  - title: "micrograd"
    author: "Andrej Karpathy"
    url: "https://github.com/karpathy/micrograd"
    type: repo
  - title: "Deep Learning, Chapter 6: Deep Feedforward Networks"
    author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville"
    url: "https://www.deeplearningbook.org/contents/mlp.html"
    type: book


status: draft
---

Once you have an engine that computes gradients, a neural network is just a convenient way to organise many `Value` objects. This note builds a multi-layer perceptron (MLP) on top of the micrograd-style engine from the previous note, trains it with plain gradient descent on four examples, and shows a mistake that almost everyone makes once: forgetting to reset the gradients.

## A few more operations

The training loss needs subtraction and squaring, and the neuron needs sums that start at the plain number 0. These small additions to the `Value` class from the backpropagation note cover that:

```python
def _pow(self, k):                      # k is a plain int or float
    out = Value(self.data**k, (self,), f'**{k}')
    def _backward():
        self.grad += k * self.data**(k - 1) * out.grad
    out._backward = _backward
    return out

Value.__pow__ = _pow
Value.__neg__ = lambda self: self * -1
Value.__sub__ = lambda self, other: self + (-other)
Value.__radd__ = lambda self, other: self + other    # lets sum() start at 0
Value.__rmul__ = lambda self, other: self * other
```

## Neuron, layer, network

A neuron with $n_{in}$ inputs computes

$$
y = \tanh\Big(\sum_{i=1}^{n_{in}} w_i x_i + b\Big),
$$

where $x_i$ are the inputs, $w_i$ the weights, $b$ the bias and $\tanh$ squashes the result into $(-1, 1)$. A layer is a list of independent neurons that all see the same inputs. An MLP is a chain of layers where each layer's outputs are the next layer's inputs.

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    def __call__(self, x):
        return (sum(wi * xi for wi, xi in zip(self.w, x)) + self.b).tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

`MLP(3, [4, 4, 1])` has 3 inputs, two hidden layers of 4 neurons and one output neuron. That is $4\cdot(3+1) + 4\cdot(4+1) + 1\cdot(4+1) = 41$ parameters. Drawing the graph of a single forward pass already gives hundreds of nodes, which is why we never backpropagate by hand beyond toy examples.

## The training loop

The toy task has four input vectors and a target of $+1$ or $-1$ for each. The loss is the sum of squared errors,

$$
L = \sum_{j=1}^{4} \big(\hat y_j - y_j\big)^2,
$$

where $\hat y_j$ is the network output and $y_j$ the target for example $j$. Gradient descent moves every parameter $p$ a small step against its gradient: $p \leftarrow p - \eta\,\partial L/\partial p$, with learning rate $\eta$.

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
net = MLP(3, [4, 4, 1])
lr = 0.05

for step in range(50):
    ypred = [net(x) for x in xs]                              # 1. forward
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

    for p in net.parameters():                                # 2. reset grads
        p.grad = 0.0
    loss.backward()                                           # 3. backward

    for p in net.parameters():                                # 4. update
        p.data -= lr * p.grad
    if step % 10 == 0:
        print(step, round(loss.data, 4))
```

The loss drops towards zero within a few dozen steps and the predictions approach the targets. In my run I kept training until the loss was about $2\times10^{-15}$ and the outputs were $1.0, -1.0, -1.0, 1.0$ to seven decimals. On four points this is pure memorisation, but it shows the whole learning loop working with nothing but the chain rule.

## The zero-grad bug

Look at step 2. Every `_backward` closure *adds* to `.grad`, which is correct inside one backward pass (a value used twice must collect both contributions). But it also means gradients from the previous step are still there when the next backward pass starts.

I saw this in my first manual steps, before I added the reset. After one update with a step of 0.01, the loss went from 7.34 to 5.92 and one weight's gradient read 0.150. Without resetting, the next `backward()` adds the new gradient on top of the old one. The effective step then becomes a running sum of all past gradients. Training may still seem to work on an easy problem, but steps grow, point in stale directions and can make training unstable.

The fix is one line before every backward pass. In PyTorch it is `optimizer.zero_grad()` (or `model.zero_grad()`). The reason PyTorch also accumulates by default is the same as here, plus a practical bonus: it lets you sum gradients over several small batches before one update.

## Key takeaways

- Neurons, layers and MLPs are just containers that create `Value` objects and expose a `parameters()` list.
- Training is a loop of four steps: forward, reset gradients, backward, update.
- Gradient descent moves each parameter a small step against its gradient, scaled by the learning rate.
- Gradients accumulate by design, so they must be reset to zero before each backward pass.

## Resources

- Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=VMj-3S1tku0. The `Neuron`, `Layer`, `MLP` classes and the toy dataset are adapted from this lecture.
- Andrej Karpathy, micrograd (repository): https://github.com/karpathy/micrograd
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, Chapter 6 "Deep Feedforward Networks": https://www.deeplearningbook.org/contents/mlp.html
- PyTorch docs, `torch.optim.Optimizer.zero_grad`: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
