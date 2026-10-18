---
title: "Training a tiny neural network from scratch (and the zero-grad bug)"
slug: training-a-tiny-mlp-from-scratch
date: 2024-12-10
topic: Deep Learning Foundations
tags: [mlp, gradient-descent, training-loop, micrograd, zero-grad]
summary: "How to build neurons, layers and a small MLP on top of a tiny autograd engine, train it with a simple loop, and why you have to reset the gradients every step."
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

Once you have something that computes gradients, a neural network is really just a tidy way to organise lots of `Value` objects. Here I build a multi-layer perceptron (MLP, a stack of fully connected layers) on top of the micrograd-style engine from the previous note, train it with plain gradient descent on four examples, and look at a mistake I think almost everyone makes once: forgetting to reset the gradients.

## A few more operations

The loss needs subtraction and squaring, and the neuron needs `sum()`, which starts from the plain number 0. These small additions to the `Value` class from the backpropagation note handle that:

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

where $x_i$ are the inputs, $w_i$ the weights, $b$ the bias, and $\tanh$ squashes the result into $(-1, 1)$. A layer is just a list of neurons that all look at the same inputs but work on their own. An MLP is a chain of layers, where each layer's outputs become the next layer's inputs.

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

`MLP(3, [4, 4, 1])` has 3 inputs, two hidden layers of 4 neurons and one output neuron. That's $4\cdot(3+1) + 4\cdot(4+1) + 1\cdot(4+1) = 41$ parameters. If you draw the graph for just one forward pass you already get hundreds of nodes. That's why nobody does backprop by hand past toy examples.

## The training loop

The toy task has four input vectors, each with a target of $+1$ or $-1$. The loss is the sum of squared errors,

$$
L = \sum_{j=1}^{4} \big(\hat y_j - y_j\big)^2,
$$

where $\hat y_j$ is the network output and $y_j$ the target for example $j$. Gradient descent nudges every parameter $p$ a small step in the direction that lowers the loss, which is against its gradient: $p \leftarrow p - \eta\,\partial L/\partial p$. Here $\eta$ is the learning rate (the step size).

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

The loss drops close to zero within a few dozen steps, and the predictions move towards the targets. In my run I kept going until the loss was about $2\times10^{-15}$ and the outputs were $1.0, -1.0, -1.0, 1.0$ to seven decimals. With four points this is just memorising, but it shows the whole learning loop working with nothing but the chain rule.

## The zero-grad bug

Look at step 2 in the loop. Every `_backward` function *adds* to `.grad`. That's correct inside one backward pass, since a value used twice has to collect both parts. But it also means the gradients from the last step are still sitting there when the next backward pass starts.

I ran into this in my first manual steps, before I added the reset. After one update with a step of 0.01, the loss went from 7.34 to 5.92 and one weight's gradient read 0.150. Without a reset, the next `backward()` piles the new gradient on top of the old one, so each step ends up using a running sum of all past gradients. On an easy problem training can still look fine, which is what makes this sneaky. But the steps get bigger, point in old directions, and can make training unstable.

The fix is one line before every backward pass. In PyTorch it's `optimizer.zero_grad()` (or `model.zero_grad()`). PyTorch also adds up gradients by default for the same reason as here, and there's a handy side effect: you can sum gradients over several small batches before doing one update.

## Key takeaways

- Neurons, layers and MLPs are just containers that create `Value` objects and give you a `parameters()` list.
- Training is a loop of four steps: forward, reset gradients, backward, update.
- Gradient descent moves each parameter a small step against its gradient, scaled by the learning rate.
- Gradients add up on purpose, so you have to set them back to zero before each backward pass.

## Resources

- Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=VMj-3S1tku0. The `Neuron`, `Layer`, `MLP` classes and the toy dataset are adapted from this lecture.
- Andrej Karpathy, micrograd (repository): https://github.com/karpathy/micrograd
- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*, Chapter 6 "Deep Feedforward Networks": https://www.deeplearningbook.org/contents/mlp.html
- PyTorch docs, `torch.optim.Optimizer.zero_grad`: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
