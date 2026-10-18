---
title: "Dense detection heads: how a feature map becomes boxes"
slug: dense-detection-heads
date: 2023-03-28
topic: Computer Vision
tags: [object-detection, detection-head, yolo, anchor-free, decoupled-head, pytorch]
summary: "How one-stage detectors turn each cell of a feature map into box, objectness and class predictions, how the output tensor is laid out and decoded, and why modern heads are decoupled and anchor-free."
sources:
  - title: "YOLOv3: An Incremental Improvement"
    author: "Joseph Redmon, Ali Farhadi"
    url: "https://arxiv.org/abs/1804.02767"
    type: paper
  - title: "YOLOX: Exceeding YOLO Series in 2021"
    author: "Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun"
    url: "https://arxiv.org/abs/2107.08430"
    type: paper
  - title: "FCOS: Fully Convolutional One-Stage Object Detection"
    author: "Zhi Tian, Chunhua Shen, Hao Chen, Tong He"
    url: "https://arxiv.org/abs/1904.01355"
    type: paper
  - title: "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection"
    author: "Xiang Li et al."
    url: "https://arxiv.org/abs/2006.04388"
    type: paper
  - title: "Ultralytics YOLOv8 documentation"
    author: "Ultralytics"
    url: "https://docs.ultralytics.com/models/yolov8/"
    type: docs


status: draft
---

A one-stage detector is usually described as backbone, neck and head. The backbone extracts features, the neck mixes them across scales (for example an [FPN](/notes/feature-pyramid-networks/)), and the head turns them into boxes. The head is the smallest part, often just a few convolutions, but it decides what the network actually predicts and how the loss can talk to it. This note follows the head from the simplest version, one $1\times1$ convolution, to the decoupled, anchor-free heads of current YOLO models.

## One cell, one small prediction vector

The key idea is that a feature map is already a grid. A map of size $13 \times 13$ computed from a $416 \times 416$ image has a **stride** of 32: each cell summarises a 32-pixel step of the input. So instead of a fully connected layer that predicts all boxes at once (as in [YOLOv1](/notes/yolo-v1-detection-as-regression/)), we can apply the same small predictor at every cell. A $1\times1$ convolution does exactly that: it is a linear layer that slides over the grid.

If each cell has $A$ anchors and there are $C$ classes, every anchor needs $4 + 1 + C$ numbers: four box values, one objectness score and $C$ class scores. The head's output therefore has $A(5 + C)$ channels.

```python
import torch
import torch.nn as nn

class CoupledHead(nn.Module):
    """One 1x1 conv predicts, for every cell and anchor: 4 box numbers, 1 objectness, C class logits."""
    def __init__(self, in_ch, num_anchors=3, num_classes=3):
        super().__init__()
        self.A, self.P = num_anchors, 5 + num_classes
        self.conv = nn.Conv2d(in_ch, self.A * self.P, kernel_size=1)

    def forward(self, x):
        b, _, h, w = x.shape
        y = self.conv(x)                                                   # (b, A*P, h, w)
        return y.view(b, self.A, self.P, h, w).permute(0, 1, 3, 4, 2)      # (b, A, h, w, P)
```

The reshape is where bugs hide. The $A \cdot P$ channels are only numbers; *we* decide that channels $0 \dots P-1$ belong to anchor 0, the next $P$ to anchor 1, and so on. The target tensor and the decoding code must use the same convention. Moving the prediction vector to the last axis, giving shape $(b, A, h, w, P)$, makes slicing easy: `out[..., :4]` is every box, `out[..., 4]` every objectness score.

## Several scales, one list of candidates

Since YOLOv3, heads run on three maps with strides 32, 16 and 8. Large objects are predicted from the coarse map and small objects from the fine map, and each level gets anchors sized for its scale.

```python
feats = [torch.randn(1, 512, 13, 13), torch.randn(1, 256, 26, 26), torch.randn(1, 128, 52, 52)]
heads = [CoupledHead(c) for c in (512, 256, 128)]
outs = [hd(f) for hd, f in zip(heads, feats)]           # (1,3,13,13,8) (1,3,26,26,8) (1,3,52,52,8)

# for inference, flatten every level into one list of candidate boxes
flat = torch.cat([o.reshape(o.shape[0], -1, o.shape[-1]) for o in outs], dim=1)
print(tuple(flat.shape))   # (1, 10647, 8) = 3 * (13*13 + 26*26 + 52*52) candidates
```

Two layouts are useful. For the **loss**, keep the grid shape $(b, A, h, w, P)$, because targets are assigned to particular cells and anchors. For **inference**, flatten everything into one list of candidates, filter by score, and run non-maximum suppression.

## Decoding: from logits to boxes

The raw outputs are unbounded numbers. YOLOv2 and YOLOv3 decode them so that each prediction stays close to "its" cell and anchor:

$$
b_x = (\sigma(t_x) + c_x)\,s, \quad b_y = (\sigma(t_y) + c_y)\,s, \quad b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}
$$

Here $(c_x, c_y)$ is the cell index, $s$ the stride, $(p_w, p_h)$ the anchor size, and $\sigma$ the sigmoid. The sigmoid keeps the centre inside its cell, which stabilised early training a lot. The exponential makes the size a positive scale factor of the anchor, so $t_w = 0$ means "exactly the anchor".

```python
def decode_level(raw, anchors, stride):
    """raw: (b, A, h, w, 5+C) logits; anchors: (A, 2) in pixels. Returns boxes in pixels."""
    b, A, h, w, _ = raw.shape
    gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cx = (torch.sigmoid(raw[..., 0]) + gx) * stride
    cy = (torch.sigmoid(raw[..., 1]) + gy) * stride
    bw = anchors[:, 0].view(1, A, 1, 1) * torch.exp(raw[..., 2])
    bh = anchors[:, 1].view(1, A, 1, 1) * torch.exp(raw[..., 3])
    obj = torch.sigmoid(raw[..., 4])
    return torch.stack([cx, cy, bw, bh, obj], dim=-1)

raw = torch.zeros(1, 3, 13, 13, 8)                                  # all-zero logits
anchors = torch.tensor([[116., 90.], [156., 198.], [373., 326.]])  # YOLOv3's stride-32 anchors
print(decode_level(raw, anchors, 32)[0, 2, 0, 0].tolist())        # [16.0, 16.0, 373.0, 326.0, 0.5]
```

All-zero logits decode to "a box of anchor size at the cell centre, 50% sure". That is a useful sanity check for any head. Choosing the anchor sizes themselves is covered in [designing anchors](/notes/designing-anchors-for-your-dataset/).

## Decoupled heads: separate branches for "where" and "what"

In a coupled head, the same features feed box regression and classification. These tasks pull in different directions. Classification wants features that are the same wherever on the object you look. Box regression wants features that change sharply near the object's edges. YOLOX (2021) showed that giving each task its own small convolutional branch speeds up convergence and improves accuracy. Most modern detectors use this **decoupled** design, as RetinaNet already did with its separate classification and box subnets.

My code-along built the first version of this as three separate $1\times1$ heads for box, objectness and class. Real decoupled heads add two $3\times3$ convolutions in each branch before the final $1\times1$, so each task gets its own features.

## Anchor-free heads

Anchors bring many hyperparameters: their sizes, their count, and the IoU rules for assigning them to targets. **Anchor-free** heads (FCOS, YOLOX, YOLOv8) drop them. Each cell predicts at most one object, and the box is described by the distances from the cell centre to its four sides: left, top, right, bottom ($l, t, r, b$), in units of the stride. The objectness branch is also gone in YOLOv8; the class scores themselves act as the confidence.

YOLOv8 adds one more twist from Generalized Focal Loss. Instead of regressing each distance as one number, it predicts a **distribution** over 16 discrete bins, and the distance is the expected value. This lets the network express uncertainty when an edge is blurry or hidden.

```python
def conv(ci, co, k=3):
    return nn.Sequential(nn.Conv2d(ci, co, k, padding=k // 2, bias=False), nn.BatchNorm2d(co), nn.SiLU())

class DecoupledHead(nn.Module):
    """Anchor-free, decoupled: one branch regresses the box, another classifies."""
    def __init__(self, in_ch, num_classes, reg_max=16):
        super().__init__()
        self.box = nn.Sequential(conv(in_ch, 64), conv(64, 64), nn.Conv2d(64, 4 * reg_max, 1))
        self.cls = nn.Sequential(conv(in_ch, 128), conv(128, 128), nn.Conv2d(128, num_classes, 1))

    def forward(self, x):
        return self.box(x), self.cls(x)

box, cls = DecoupledHead(256, num_classes=3)(torch.randn(1, 256, 80, 80))
logits = box.view(1, 4, 16, 80, 80)                              # 4 sides x 16 bins
dist = (logits.softmax(dim=2) * torch.arange(16.).view(1, 1, 16, 1, 1)).sum(dim=2)
print(tuple(dist.shape))   # (1, 4, 80, 80): expected l, t, r, b per cell, in strides
```

The box $= (c_x - l,\; c_y - t,\; c_x + r,\; c_y + b) \times s$. In my TensorFlow sketch of the YOLOv8 head I had written $4 \times$ the number of anchors for the box channels; the correct count is $4 \times$ `reg_max`, because there are no anchors, only bins.

## Key takeaways

- A dense head is a small network applied at every cell; a $1\times1$ convolution with $A(5+C)$ output channels is the simplest version.
- The channel layout is a convention. Reshape to $(b, A, h, w, P)$ for the loss and flatten to one candidate list for inference.
- YOLOv2/v3 decoding keeps centres inside their cell with a sigmoid and scales anchors with an exponential; zero logits should decode to the anchor itself.
- Decoupled heads give box regression and classification separate features, which trains faster and better.
- Anchor-free heads predict distances to the four sides, and YOLOv8 predicts each distance as a distribution over bins.

## Resources

- Redmon, Farhadi, "YOLOv3: An Incremental Improvement", 2018. [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)
- Ge, Liu, Wang, Li, Sun, "YOLOX: Exceeding YOLO Series in 2021". [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)
- Tian, Shen, Chen, He, "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019. [arXiv:1904.01355](https://arxiv.org/abs/1904.01355)
- Li et al., "Generalized Focal Loss", NeurIPS 2020 (the distribution-over-bins box regression). [arXiv:2006.04388](https://arxiv.org/abs/2006.04388)
- Ultralytics YOLOv8 documentation: [docs.ultralytics.com](https://docs.ultralytics.com/models/yolov8/)
- A YouTube code-along series on object-detection components (detection head, better head, FPN neck), and Aladdin Persson's YOLOv3-from-scratch video ([Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection)).
- Related notes: [YOLOv1](/notes/yolo-v1-detection-as-regression/), [anchors and RPNs](/notes/anchors-and-region-proposal-networks/), [FPN](/notes/feature-pyramid-networks/), [mean average precision](/notes/mean-average-precision/).
