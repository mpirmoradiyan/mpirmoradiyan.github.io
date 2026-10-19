---
title: "What a Conv2D layer actually computes: multi-channel cross-correlation"
slug: what-a-conv2d-layer-computes
date: 2020-06-02
topic: Deep Learning Foundations
tags: [university-lab, convolution, cnn, cross-correlation, parameters]
summary: "How a Conv2D layer differs from the textbook 2D convolution in scipy: it skips the kernel flip, each filter spans all input channels, and the outputs of many filters are stacked, which explains its parameter count and why pooling adds none."
sources:
  - title: "732A78 Deep Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Deep Learning, Chapter 9 (Convolutional Networks)"
    author: "Ian Goodfellow, Yoshua Bengio, Aaron Courville"
    url: "https://www.deeplearningbook.org/contents/convnets.html"
    type: book
  - title: "A guide to convolution arithmetic for deep learning"
    author: "Vincent Dumoulin, Francesco Visin"
    url: "https://arxiv.org/abs/1603.07285"
    type: paper
  - title: "SciPy: scipy.signal.convolve2d and correlate2d"
    author: "SciPy developers"
    url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html"
    type: docs


status: draft
---

The CNN lab started with classic filters (Gaussian blur, Sobel edges) run through `scipy.signal.convolve2d`, and then moved on to Keras `Conv2D` layers. One question asked whether `Conv2D` does the same standard 2D convolution. I answered yes. It doesn't, and the differences explain things I got wrong later in the same lab, such as where the parameters come from.

## Three differences from `convolve2d`

**1. No kernel flip.** Mathematical convolution flips the kernel before sliding it:

$$
(I * K)(i, j) = \sum_{u}\sum_{v} I(i - u,\, j - v)\, K(u, v)
$$

Deep learning libraries compute **cross-correlation**, which slides the kernel as it is:

$$
(I \star K)(i, j) = \sum_{u}\sum_{v} I(i + u,\, j + v)\, K(u, v)
$$

$I$ is the image, $K$ the kernel, and $(i, j)$ the output position. Since the kernel is learned, the flip doesn't matter: the network would just learn the flipped kernel. But for a fixed kernel like Sobel, the two give different signs.

**2. Each filter spans all input channels.** A colour image has 3 channels. A $3 \times 3$ filter in the first layer is really $3 \times 3 \times 3$. It looks at all three channels at once and sums everything into **one** output map. That's why the lab's GPU test talked about "7 × 7 × 3" filters. In a later layer with 32 input channels, each $3 \times 3$ filter is $3 \times 3 \times 32$.

**3. Many filters, stacked.** A layer with $C_\text{out}$ filters produces $C_\text{out}$ maps, stacked as channels. Putting it together, output channel $o$ at position $(i, j)$ is

$$
y_o(i, j) = b_o + \sum_{c=1}^{C_\text{in}} \sum_{u=0}^{K-1}\sum_{v=0}^{K-1} x_c(i + u,\, j + v)\, w_{o,c}(u, v)
$$

where $x_c$ is input channel $c$, $w_{o,c}$ is the $K \times K$ slice of filter $o$ that looks at channel $c$, and $b_o$ is one bias per filter.

```python
import numpy as np
from scipy.signal import convolve2d, correlate2d

rng = np.random.default_rng(0)
x = rng.normal(size=(8, 8, 3))          # H x W x C_in
w = rng.normal(size=(3, 3, 3, 16))      # K x K x C_in x C_out
b = np.zeros(16)

def conv2d_layer(x, w, b):
    """What a Conv2D layer computes ('valid' padding, stride 1)."""
    K, _, C_in, C_out = w.shape
    out = np.zeros((x.shape[0] - K + 1, x.shape[1] - K + 1, C_out))
    for o in range(C_out):
        for c in range(C_in):           # sum over ALL input channels
            out[:, :, o] += correlate2d(x[:, :, c], w[:, :, c, o], mode="valid")
        out[:, :, o] += b[o]
    return out

print(conv2d_layer(x, w, b).shape)      # (6, 6, 16)
print(w.size + b.size)                  # 448 = (3*3*3 + 1) * 16

k = w[:, :, 0, 0]                       # correlation = convolution with a flipped kernel
print(np.allclose(correlate2d(x[:, :, 0], k, "valid"),
                  convolve2d(x[:, :, 0], k[::-1, ::-1], "valid")))   # True
```

## Counting parameters

From the formula above, a Conv2D layer has

$$
(K \cdot K \cdot C_\text{in} + 1) \cdot C_\text{out}
$$

parameters: one $K \times K \times C_\text{in}$ block of weights plus one bias for each filter. In our 4-layer network, the first layer has $(3 \cdot 3 \cdot 3 + 1) \cdot 16 = 448$, and the third (32 channels in, 64 out) has $(9 \cdot 32 + 1) \cdot 64 = 18{,}496$, matching `model.summary()`.

One lab question asked why a layer's parameter count isn't just "number of filters × coefficients per filter (plus biases)" when you think of each filter as $K \times K$. I answered "because of pooling". That's wrong. The missing factor is $C_\text{in}$: each filter has a separate $K \times K$ slice for every input channel. Pooling has nothing to do with it.

## Pooling has no parameters

Max pooling takes the largest value in each $2 \times 2$ window. Nothing is learned, so its parameter count is 0, which `model.summary()` shows. Pooling cuts parameters only *indirectly*: it halves the height and width, so when the conv stack ends in `Flatten` and a `Dense` layer, that dense layer gets fewer inputs. In our network the last conv output was $2 \times 2 \times 96 = 384$ numbers, so the dense layer had $384 \cdot 50 + 50 = 19{,}250$ weights. Without the four poolings it would have been $32 \cdot 32 \cdot 96 \cdot 50$, about 4.9 million.

That `Flatten` → `Dense` step is also why the trained network only accepts $32 \times 32$ images. The conv layers would run on any size, but the dense layer expects exactly 384 inputs. In the lab I suggested transfer learning to handle other sizes, which isn't the fix. The fix is to replace `Flatten` with global average pooling (or another fixed-size pooling), see [spatial pyramid pooling](/notes/spatial-pyramid-pooling/).

## "Same" vs. "valid"

With "valid" padding the output shrinks by $K - 1$ per layer ($512 \to 510$ for a $3 \times 3$ Sobel filter in the lab). Stack many layers and the maps keep shrinking, and border pixels take part in fewer outputs. "Same" padding adds $(K-1)/2$ zeros on each side so the size stays put. The general size formula is in [classic CNN architectures](/notes/classic-cnn-architectures/).

## Key takeaways

- Conv2D computes cross-correlation (no flip), summed over all input channels, with one output map per filter.
- Parameters: $(K^2 C_\text{in} + 1) C_\text{out}$. The $C_\text{in}$ factor is the part that's easy to forget.
- Pooling has no parameters. It only shrinks the maps, which shrinks any dense layer after `Flatten`.
- `Flatten` + `Dense` ties a CNN to one input size. Global pooling removes that limit.

## Resources

- 732A78 Deep Learning, Linköping University (CNN lab).
- Goodfellow, Bengio and Courville, [Deep Learning, Chapter 9](https://www.deeplearningbook.org/contents/convnets.html).
- Dumoulin and Visin, [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285), 2016.
- [SciPy convolve2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html) and [correlate2d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html).
