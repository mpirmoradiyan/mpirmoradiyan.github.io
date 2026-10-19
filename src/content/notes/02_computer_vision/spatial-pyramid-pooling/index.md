---
title: "Spatial pyramid pooling: SPP, PSPNet's pyramid pooling module and SPPF"
slug: spatial-pyramid-pooling
date: 2024-03-01
topic: Computer Vision
tags: [pooling, receptive-field, context, yolo, pspnet, segmentation, object-detection, pytorch]
summary: "How pooling one feature map at a few window sizes gives a CNN fixed-length outputs and a view of the whole image, and how SPP-net, PSPNet's PPM and YOLO's SPP/SPPF differ."
sources:
  - title: "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition"
    author: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
    url: "https://arxiv.org/abs/1406.4729"
    type: paper
  - title: "Pyramid Scene Parsing Network (PSPNet)"
    author: "Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia"
    url: "https://arxiv.org/abs/1612.01105"
    type: paper
  - title: "YOLOv4: Optimal Speed and Accuracy of Object Detection"
    author: "Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao"
    url: "https://arxiv.org/abs/2004.10934"
    type: paper
  - title: "Counting and locating high-density objects using convolutional neural network"
    author: "Mauro dos Santos de Arruda et al."
    url: "https://arxiv.org/abs/2102.04366"
    type: paper
  - title: "Ultralytics YOLOv5/YOLOv8 source (SPPF module)"
    author: "Ultralytics"
    url: "https://github.com/ultralytics/ultralytics"
    type: repo


status: draft
---

A convolution only looks at a small window. Even a deep CNN usually "sees" much less of the image than its theoretical receptive field (the patch of input that can affect one output) suggests. But lots of tasks need a wider view. To call a pixel "boat", it helps to know there's water around it. To count a crowd, it helps to know how packed the whole scene is. Spatial pyramid pooling is a cheap way to get that wider view: pool the same feature map at a few scales and stack the results. It shows up in three common forms, and each one solves a slightly different problem.

## The basic idea

Pooling sums up a region with one number per channel, either the max or the average. A small window keeps detail. A big window, up to the whole map, gives context. A **pyramid** of windows gives you both, and if you concatenate them the next layer can pick what it needs. Pooling has no weights, so this extra view costs almost nothing in parameters.

The way I keep the three variants apart is by asking what each one keeps fixed:

| Variant | Fixed | Output | Typical use |
|---|---|---|---|
| SPP-net (2014) | the number of bins | a fixed-length vector | classification or detection on any input size |
| PSPNet PPM (2017) | the number of bins, then upsample | a map at the input resolution | segmentation, counting |
| YOLO SPP / SPPF | the window size, stride 1 | a map at the input resolution | detector backbones |

## SPP-net: a fixed-length vector from any image size

Early CNNs needed a fixed input size, like 224×224. The reason is the fully connected layers at the end: they expect a vector of a fixed length. SPP-net got rid of that limit. It splits the last feature map into fixed grids of bins, for example $1\times1$, $2\times2$ and $4\times4$ as in the paper's figure, no matter how big the map is, and max-pools each bin. With $c$ channels you always get $c(1 + 4 + 16) = 21c$ numbers.

```python
import torch
import torch.nn.functional as F

def spp_vector(x, levels=(1, 2, 4)):
    """SPP-net: pool a (b, c, h, w) map into a fixed-length vector, whatever h and w are."""
    pooled = [F.adaptive_max_pool2d(x, n).flatten(1) for n in levels]   # n x n bins per level
    return torch.cat(pooled, dim=1)

for h, w in [(13, 13), (20, 31), (7, 9)]:
    print((h, w), tuple(spp_vector(torch.randn(1, 256, h, w)).shape))   # always (1, 5376)
```

So the bin *size* changes with the map, but the bin *count* doesn't. For detection this was a big speed-up. The backbone ran once per image, and each region proposal was pooled from that shared map. Fast R-CNN's RoI pooling is basically the single-level version of the same idea.

## PSPNet's pyramid pooling module: context for every pixel

Segmentation needs a map as output, not a vector. The Pyramid Pooling Module (PPM) in PSPNet pools the map into $1\times1$, $2\times2$, $3\times3$ and $6\times6$ bins, usually with average pooling. Each level then goes through a $1\times1$ convolution that cuts its channels to $1/4$, so the four levels together add as many channels as the input had. Each level is upsampled back to the map size with bilinear interpolation. Last step: concatenate all of them with the **original** map.

```python
import torch.nn as nn

class PyramidPooling(nn.Module):
    """PSPNet-style: average-pool into 1x1, 2x2, 3x3 and 6x6 bins, reduce channels, upsample, concat."""
    def __init__(self, in_ch, bins=(1, 2, 3, 6)):
        super().__init__()
        self.bins = bins
        self.reduce = nn.ModuleList(nn.Conv2d(in_ch, in_ch // len(bins), 1) for _ in bins)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x]                                         # keep the original, local features
        for n, conv in zip(self.bins, self.reduce):
            y = conv(F.adaptive_avg_pool2d(x, n))          # context of each n x n region
            outs.append(F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False))
        return torch.cat(outs, dim=1)

print(tuple(PyramidPooling(512)(torch.randn(1, 512, 64, 64)).shape))   # (1, 1024, 64, 64)
```

After this module, every pixel carries four extra summaries: one of the whole image ($1\times1$), one of its half or quarter ($2\times2$), and finer ones of smaller sub-regions ($3\times3$, $6\times6$). Keeping the original map in the concat really matters. Without it, small objects that get washed out in the coarse bins would just be lost.

The same module helps with counting dense objects. In the multi-sigma counting paper I studied, a PPM sits after a VGG feature extractor, so each location "knows" whether it's in a dense or a sparse part of the image. My version of that model used max pooling for the global $1\times1$ level. The difference is simple once you see it: max pooling keeps the strongest response, so it tells you whether something is there at all. Average pooling keeps the mean, so it tells you how much of it there is. PSPNet found average pooling a bit better for scene parsing.

## YOLO's SPP: fixed windows, stride 1

YOLOv3-SPP and YOLOv4 use a different kind of "pyramid", and this one confused me at first because there are no bins. Instead they apply max pooling with windows of 5, 9 and 13, with **stride 1** and "same" padding, so the map size stays the same. The outputs are concatenated with the input. The big windows widen the receptive field of the deepest backbone layer for almost no cost, which helped with large objects.

## SPPF: same result, just faster

Ultralytics' SPPF (YOLOv5 and later) gets exactly the same output by applying one $5\times5$ pool three times in a row. Why does that work? Two stride-1 max pools of size 5 cover a $9\times9$ window, because the max of maxes over overlapping windows is just the max over their union. Three of them cover $13\times13$. Since each step reuses the previous result, you skip most of the comparisons.

```python
import time

def pool(x, k):                      # stride 1, "same" padding: the size does not change
    return F.max_pool2d(x, k, stride=1, padding=k // 2)

def spp(x):                          # YOLOv3/v4 SPP: three pools in parallel
    return torch.cat([x, pool(x, 5), pool(x, 9), pool(x, 13)], dim=1)

def sppf(x):                         # YOLOv5/v8 SPPF: one 5x5 pool applied three times in a row
    y1 = pool(x, 5); y2 = pool(y1, 5); y3 = pool(y2, 5)
    return torch.cat([x, y1, y2, y3], dim=1)

x = torch.randn(8, 256, 40, 40)
print(torch.equal(spp(x), sppf(x)))   # True: identical outputs

for f in (spp, sppf):
    t = time.perf_counter()
    for _ in range(20):
        f(x)
    print(f.__name__, f"{(time.perf_counter() - t) / 20 * 1000:.1f} ms")   # SPPF was ~5x faster on my CPU
```

The outputs match exactly because PyTorch pads max pooling with $-\infty$, so the borders behave the same in both versions. Real SPPF blocks also add a $1\times1$ convolution before the pools, which halves the channels, and another one after the concat, which mixes the four views together.

When I first sketched the YOLOv8 modules, I wrote the parallel 5/9/13 version and called it SPPF. It gives the right numbers but not the speed. The "F" (fast) is exactly the chained trick.

## Which one to use

- You need a **fixed-length vector** from inputs of different sizes: SPP-net bins (or just global average pooling, which is the $1\times1$ level on its own).
- You need **global context at every pixel** for segmentation or density estimation: a PPM after the backbone, keeping the original map in the concat.
- You want a cheap **receptive-field boost** at the end of a detector backbone: SPPF.

## Key takeaways

- Pool one feature map at several scales and concatenate: you get local detail and wide context together, and the pooling itself adds no weights.
- SPP-net fixes the number of bins, so any input size gives a vector of the same length.
- PSPNet's PPM pools into 1, 2, 3 and 6 bins, cuts channels, upsamples and concatenates with the input, so every pixel gets global context.
- YOLO's SPP uses stride-1 max pools of 5, 9 and 13. SPPF gets the identical output from three chained 5×5 pools and runs several times faster.

## Resources

- He, Zhang, Ren, Sun, "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition", ECCV 2014. [arXiv:1406.4729](https://arxiv.org/abs/1406.4729)
- Zhao, Shi, Qi, Wang, Jia, "Pyramid Scene Parsing Network", CVPR 2017. [arXiv:1612.01105](https://arxiv.org/abs/1612.01105)
- Bochkovskiy, Wang, Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection", 2020. [arXiv:2004.10934](https://arxiv.org/abs/2004.10934)
- de Arruda et al., "Counting and locating high-density objects using convolutional neural network", 2021. [arXiv:2102.04366](https://arxiv.org/abs/2102.04366)
- Ultralytics source code, `SPPF` module: [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- Related notes: [dense detection heads](/notes/dense-detection-heads/), [density maps for counting](/notes/density-maps-for-object-counting/), [segmentation decoders](/notes/segmentation-decoders-fcn-and-unet/), [classic CNN architectures](/notes/classic-cnn-architectures/).
