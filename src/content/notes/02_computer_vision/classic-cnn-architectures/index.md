---
title: "Classic CNN architectures: LeNet, VGG, GoogLeNet and ResNet"
slug: classic-cnn-architectures
date: 2024-03-13
topic: Computer Vision
tags: [cnn, lenet, vgg, googlenet, inception, resnet, architectures, pytorch]
summary: "How four landmark networks each solved one design problem (stacking small filters, cutting cost with 1x1 convolutions, and training very deep nets with residual connections), and how to work out output sizes, parameters and compute for any conv layer."
sources:
  - title: "Gradient-Based Learning Applied to Document Recognition (LeNet-5)"
    author: "Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner"
    url: "http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf"
    type: paper
  - title: "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)"
    author: "Karen Simonyan, Andrew Zisserman"
    url: "https://arxiv.org/abs/1409.1556"
    type: paper
  - title: "Going Deeper with Convolutions (GoogLeNet)"
    author: "Christian Szegedy et al."
    url: "https://arxiv.org/abs/1409.4842"
    type: paper
  - title: "Deep Residual Learning for Image Recognition (ResNet)"
    author: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
    url: "https://arxiv.org/abs/1512.03385"
    type: paper
  - title: "Machine-Learning-Collection (PyTorch architecture implementations)"
    author: "Aladdin Persson"
    url: "https://github.com/aladdinpersson/Machine-Learning-Collection"
    type: repo


status: draft
---

Most image backbones in use today are descendants of four networks: LeNet, VGG, GoogLeNet and ResNet. Each one answered a specific question. How do you stack convolutions at all? How deep can you go with simple blocks? How do you get more capacity without paying for it in compute? How do you train a network with a hundred layers? Rebuilding them by hand is the fastest way to see why modern blocks look the way they do. This note covers the architectures; how to pull multi-scale features out of a trained one is in [CNN backbone feature extraction](/notes/cnn-backbone-feature-extraction/).

## The one formula you need: output size

Every conv and pooling layer changes the spatial size by the same rule. For input width $W$, kernel size $K$, padding $P$ and stride $S$:

$$
W_\text{out} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor + 1
$$

The same holds for height. Two cases come up all the time. With stride 1 and $P = (K-1)/2$ the size does not change ("same" padding). With stride 2 the size roughly halves.

```python
def out_size(w, k, s=1, p=0):
    """Output width of a conv or pooling layer: floor((W + 2P - K) / S) + 1."""
    return (w + 2 * p - k) // s + 1

print(out_size(512, k=5, s=1, p=2))   # 512: 'same' padding, p = (k-1)/2
print(out_size(32, k=5))              # 28: LeNet's first conv
print(out_size(224, k=7, s=2, p=3))   # 112: ResNet/GoogLeNet stem
print(out_size(112, k=3, s=2, p=1))   # 56: the max-pool after the stem
```

You need this every time a conv stack ends in a fully connected layer, because the linear layer must know how many numbers arrive. Two more counts are useful. A conv layer has $C_\text{in} \cdot K^2 \cdot C_\text{out}$ weights (plus $C_\text{out}$ biases), and costs about $H_\text{out} W_\text{out} \cdot C_\text{in} K^2 \cdot C_\text{out}$ multiply-adds. The weights do not depend on image size; the compute does.

## LeNet-5: the template

LeNet-5 (1998) read 32×32 handwritten digits. Its pattern is still the basic recipe: conv, nonlinearity, downsample, repeat, then a few fully connected layers.

- 5×5 conv to 6 channels: 32 → 28, pool → 14
- 5×5 conv to 16 channels: 14 → 10, pool → 5
- 5×5 conv to 120 channels: 5 → 1, so the last "conv" is really a fully connected layer
- linear 120 → 84 → 10

The whole network has about 60,000 parameters. My version used ReLU and max-pooling; the original used scaled tanh units and trainable average-pooling ("subsampling"), which were standard at the time. The idea is the same: early layers see small local patterns, and pooling lets later layers see larger areas with fewer pixels.

## VGG: depth from small, uniform blocks

VGG (2014) asked how far a very plain design can go. It uses only 3×3 convs with padding 1 and 2×2 max-pools, in five stages with 64, 128, 256, 512 and 512 channels. VGG-16 and VGG-19 differ only in how many convs each stage has, so the whole network fits in a short config list such as `[64, 64, "M", 128, 128, "M", ...]`, where `"M"` means max-pool.

Why 3×3 only? Two stacked 3×3 convs see a 5×5 area, and three see 7×7. Compared with one 7×7 conv with $C$ channels in and out, three 3×3 convs use $3 \cdot 9C^2 = 27C^2$ weights instead of $49C^2$, and they add two extra nonlinearities. More depth, fewer weights.

VGG's weakness is cost. Its three fully connected layers at the end hold about 124M of its 138M parameters, and the 3×3 convs at full resolution make it the most expensive of the four networks to run (about 15.5 billion multiply-adds per 224×224 image).

## GoogLeNet: width with 1×1 bottlenecks

GoogLeNet (2014) won ImageNet with about 6.6M parameters, roughly 20 times fewer than VGG-16. Its building block, the **Inception module**, runs four branches in parallel on the same input and concatenates their outputs along the channel axis:

1. a 1×1 conv
2. a 1×1 conv, then a 3×3 conv
3. a 1×1 conv, then a 5×5 conv
4. a 3×3 max-pool (stride 1), then a 1×1 conv

The branches look at the image at different scales, and the network learns how much to use each one. (My small `1_cnn.py` experiment did a simple version of this on MNIST: 3×3, 5×5 and 7×7 convs in parallel, concatenated.) Every branch keeps the spatial size, so the outputs can be stacked: `torch.cat([b1, b2, b3, b4], dim=1)`.

The key trick is the 1×1 conv *before* the big filters. A 1×1 conv mixes channels at each pixel without looking at neighbours, so it can squeeze 192 channels down to 16 cheaply before the expensive 5×5 conv:

```python
import torch.nn as nn
n = lambda m: sum(p.numel() for p in m.parameters())
direct = nn.Conv2d(192, 32, 5, padding=2)
reduced = nn.Sequential(nn.Conv2d(192, 16, 1), nn.Conv2d(16, 32, 5, padding=2))
print(n(direct), n(reduced))   # 153632 15920: about 10x fewer weights
```

GoogLeNet also replaced the big dense layers with global average pooling, which removed most of VGG's parameter cost. Finally, it attached two small **auxiliary classifiers** to middle layers during training. Their losses (weighted by 0.3) push useful gradient into the middle of the 22-layer network. They are thrown away at test time, which is why the model returns three outputs in training mode and one in eval mode.

## ResNet: learning a residual

By 2015 a strange problem was clear: past a certain depth, adding layers made *training* error worse, not just test error. This is not overfitting. A deeper network could in principle copy the shallower one and set the extra layers to the identity, but plain stacks of conv layers find that hard to learn.

ResNet changes what each block has to learn. Instead of $y = F(x)$, a block computes

$$
y = \text{ReLU}\big(F(x) + x\big)
$$

where $F$ is a few conv-BN-ReLU layers and $x$ is passed around them unchanged on a "skip" or "shortcut" connection. If the identity is the best thing to do, $F$ only has to output zero, which is easy. In the backward pass the addition copies the gradient straight to earlier layers, so the signal does not have to pass through every layer's weights. This let He et al. train networks with 152 layers (and over 1,000 in experiments).

Two details make it work in practice:

- **Shape changes.** When a block halves the resolution or changes the channel count, $x$ no longer matches $F(x)$. The shortcut then uses a 1×1 conv with stride 2 (plus BN) to project it.
- **The bottleneck block.** ResNet-50 and deeper use 1×1 → 3×3 → 1×1 blocks: reduce channels by 4, do the 3×3 conv in the thin space, expand back. It is the same 1×1 trick as Inception:

```python
basic = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.Conv2d(256, 256, 3, padding=1))
bottleneck = nn.Sequential(nn.Conv2d(256, 64, 1), nn.Conv2d(64, 64, 3, padding=1),
                           nn.Conv2d(64, 256, 1))
print(n(basic), n(bottleneck))   # 1180160 70016
```

The overall layout is a 7×7 stride-2 stem, a max-pool, four stages of residual blocks (each stage after the first halves the resolution and doubles the width), global average pooling, and one linear layer.

## Side by side

Counting parameters and multiply-adds with forward hooks on every conv and linear layer gives:

| Network | Year | Parameters | Multiply-adds (224×224) |
|---|---|---|---|
| LeNet-5 | 1998 | 0.06 M | 0.4 M (32×32 input) |
| VGG-16 | 2014 | 138 M | 15.5 G |
| GoogLeNet | 2014 | 6.6 M | 1.5 G |
| ResNet-50 | 2015 | 25.6 M | 4.1 G |

![Two log-scale bar charts. Left: parameters, from 0.06 million for LeNet-5 to 138 million for VGG-16, with GoogLeNet at 6.6 and ResNet-50 at 25.6. Right: multiply-adds per image, highest for VGG-16 at about 15,500 million.](fig-params-and-compute.png)

Parameters and compute are different things. VGG's weights sit mostly in dense layers that run once, while its compute sits in convs that run at every pixel. GoogLeNet and ResNet win on both by using 1×1 convs and global pooling.

## Key takeaways

- Output size is $\lfloor (W + 2P - K)/S \rfloor + 1$; use it to size the first dense layer and to check every stage.
- VGG showed that stacked 3×3 convs give a large receptive field with fewer weights than one big kernel.
- 1×1 convolutions mix channels cheaply; GoogLeNet and ResNet's bottleneck both use them to make large filters affordable.
- Residual connections let a block learn a change to its input instead of a whole new mapping, which made very deep networks trainable.
- Global average pooling replaced huge fully connected heads and is now standard.

## Resources

- LeCun et al., [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf), 1998 (LeNet-5).
- Simonyan and Zisserman, [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556), 2014 (VGG).
- Szegedy et al., [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842), 2014 (GoogLeNet / Inception).
- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015 (ResNet).
- Aladdin Persson, [Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection): from-scratch PyTorch versions of these networks; my code-alongs followed its style.
- Related note: [CNN backbone feature extraction](/notes/cnn-backbone-feature-extraction/).
