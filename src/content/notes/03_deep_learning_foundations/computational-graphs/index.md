---
title: "Computational graphs: how a formula becomes something you can differentiate"
slug: computational-graphs
date: 2024-12-14
topic: Deep Learning Foundations
tags: [computational-graph, derivatives, autograd, micrograd, graphviz]
summary: "How any expression, including a neuron, can be written as a graph of values and operations, and why that graph is the data structure behind automatic differentiation."
sources:
  - title: "The spelled-out intro to neural networks and backpropagation: building micrograd"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=VMj-3S1tku0"
    type: video
  - title: "micrograd"
    author: "Andrej Karpathy"
    url: "https://github.com/karpathy/micrograd"
    type: repo
  - title: "Graphviz Python package documentation"
    author: "Sebastian Bank"
    url: "https://graphviz.readthedocs.io/"
    type: docs


status: draft
---

Every deep learning framework does the same trick: it records your calculation as a graph, then walks that graph backwards to get gradients. Before writing backpropagation, it helps to see that graph clearly. This note covers what a derivative tells us, how a formula becomes a graph of nodes and operations, and how to build and draw such a graph in a few lines of Python.

## What a derivative tells you

The derivative of $f$ at a point $x$ answers one question: if I nudge $x$ by a tiny amount $h$, how much does $f$ change, relative to $h$?

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

Here $h$ is the size of the nudge. With a small but finite $h$ we get a numerical estimate, which is a handy way to check gradients later.

```python
def f(x):
    return 3*x**2 - 4*x + 5

h = 1e-6
x = -3.0
print((f(x + h) - f(x)) / h)   # about -22.0; exact: f'(x) = 6x - 4 = -22
```

![Plot of f(x) = 3x^2 - 4x + 5 for x from -5 to 5: a parabola with its minimum near x = 2/3](fig-parabola.png)

At $x=-3$ the curve goes down steeply as $x$ grows, so the slope is large and negative. At $x = 2/3$ the slope is zero, which is the minimum.

The same idea works with several inputs. For $d = a \cdot b + c$, nudging one input at a time gives the partial derivatives: $\partial d/\partial a = b$, $\partial d/\partial b = a$ and $\partial d/\partial c = 1$. The numbers confirm this: with $a=2$, $b=3$, $c=10$ the estimates come out as about 3, 2 and 1.

## From formula to graph

A computational graph is a directed graph with two kinds of nodes:

- **Value nodes** hold numbers: inputs, parameters, intermediate results and the output.
- **Operation nodes** (`+`, `*`, `tanh`, ...) take values in and produce a new value.

Edges show which values feed which operation. For $z = (x + y)\cdot w$, the inputs $x$ and $y$ flow into `+`, the result and $w$ flow into `*`, and `*` produces $z$. Data flows from left to right. Gradients will later flow from right to left along the same edges.

I first drew these graphs by hand with the `graphviz` package:

```python
from graphviz import Digraph

g = Digraph(graph_attr={'rankdir': 'LR'})          # left-to-right layout
for name in ['x', 'y', 'w', 'z']:
    g.node(name, name)
g.node('add', '+', shape='circle')
g.node('mul', '*', shape='circle')
g.edges([('x', 'add'), ('y', 'add'), ('add', 'mul'), ('w', 'mul'), ('mul', 'z')])
g   # renders in a notebook
```

I also tried to generate the graph automatically by parsing strings such as `"z = ((x + y) - (p + r)) * w"` character by character. That works for tiny examples but breaks quickly with nested brackets and operator precedence. The lesson was useful: you do not need a parser at all. If every arithmetic operation records its own inputs while it runs, the graph builds itself.

## Letting the code build the graph

The approach from micrograd is to wrap each number in a small `Value` object. Overloaded operators compute the result and remember which values produced it.

```python
import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)   # the values this one was computed from
        self._op = _op                # the operation that produced it
        self.label = label
        self.grad = 0.0               # filled in later by backpropagation

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

    def tanh(self):
        return Value(math.tanh(self.data), (self,), 'tanh')
```

Now an ordinary-looking expression produces a graph as a side effect. Here is a single neuron with two inputs, $o = \tanh(x_1 w_1 + x_2 w_2 + b)$:

```python
x1, x2 = Value(2.0, label='x1'), Value(0.0, label='x2')
w1, w2 = Value(-3.0, label='w1'), Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')
n = x1*w1 + x2*w2 + b
o = n.tanh()
print(o.data)   # 0.7071...
```

The odd bias value is chosen so that $n \approx 0.8814$ and $o = \tanh(n) \approx 1/\sqrt{2}$, which gives round numbers later.

## Walking the graph

To draw or process the graph we only need to start at the output and follow `_prev` links recursively, collecting nodes and edges:

```python
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges
```

Feeding these sets to `graphviz` (one record per value showing its label, data and grad, plus a small node for each operation) gives the picture below.

![Computational graph of a single tanh neuron: x1*w1 and x2*w2 are added, the bias b is added to give n, and tanh(n) gives the output 0.7071](fig-neuron-graph.png)

All `grad` fields are still zero. Filling them in is the job of backpropagation, which is the subject of the next note. The key point here is that the graph already contains everything needed: each node knows its inputs and the operation that combined them, and each operation has a simple local derivative.

## Key takeaways

- A derivative measures how sensitive an output is to a small nudge of one input; a finite-difference estimate is a good sanity check.
- A computational graph splits a formula into value nodes and operation nodes connected by edges.
- You do not need to parse formulas. If each operation records its inputs, running the code builds the graph.
- A recursive walk from the output over the stored inputs recovers the whole graph, which is the starting point for automatic differentiation.

## Resources

- Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation: building micrograd", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=VMj-3S1tku0. The `Value` class, `trace` function and the neuron example are adapted from this lecture.
- Andrej Karpathy, micrograd (repository): https://github.com/karpathy/micrograd
- Graphviz Python package documentation: https://graphviz.readthedocs.io/
