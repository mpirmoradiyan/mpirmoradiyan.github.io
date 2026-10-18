---
title: "Dense detection heads: how a feature map becomes boxes"
slug: dense-detection-heads
date: 2023-03-28
topic: Computer Vision
tags: [object-detection, detection-head, yolo, anchor-free, decoupled-head, pytorch]
summary: "How a one-stage detector turns every cell of a feature map into box, objectness and class predictions, how to reshape and decode that output, and why newer heads are split into branches and drop anchors."
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

People usually split a one-stage detector into three parts: backbone, neck and head. The backbone pulls out features. The neck mixes them across scales (for example an [FPN](/notes/feature-pyramid-networks/)). The head turns them into boxes. The head is the smallest part, often just a few convolutions, but it decides what the network actually predicts and how the loss gets to talk to it. Here I follow the head from the simplest version, one $1\times1$ convolution, up to the split, anchor-free heads in current YOLO models.

## Each cell gets its own small prediction

The thing that made it click for me: a feature map is already a grid. A $13 \times 13$ map computed from a $416 \times 416$ image has a **stride** of 32, meaning each cell stands for a 32-pixel step in the input. So instead of one big fully connected layer that predicts all boxes at once (like [YOLOv1](/notes/yolo-v1-detection-as-regression/)), we can run the same small predictor at every cell. A $1\times1$ convolution does exactly that. It's just a linear layer that slides over the grid.

Say each cell has $A$ anchors (preset reference boxes) and there are $C$ classes. Every anchor needs $4 + 1 + C$ numbers: four for the box, one objectness score ("is there an object here?") and $C$ class scores. So the head outputs $A(5 + C)$ channels.

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

The reshape is where the bugs hide. The $A \cdot P$ channels are just numbers. *We* decide that channels $0 \dots P-1$ belong to anchor 0, the next $P$ to anchor 1, and so on. The target tensor and the decoding code have to follow the same rule. Moving the prediction vector to the last axis, so the shape is $(b, A, h, w, P)$, makes slicing easy: `out[..., :4]` is every box and `out[..., 4]` is every objectness score.

## Several scales, one list of candidates

Since YOLOv3, heads run on three maps with strides 32, 16 and 8. Big objects come from the coarse map, small objects from the fine map, and each level gets anchors sized for its scale.

```python
feats = [torch.randn(1, 512, 13, 13), torch.randn(1, 256, 26, 26), torch.randn(1, 128, 52, 52)]
heads = [CoupledHead(c) for c in (512, 256, 128)]
outs = [hd(f) for hd, f in zip(heads, feats)]           # (1,3,13,13,8) (1,3,26,26,8) (1,3,52,52,8)

# for inference, flatten every level into one list of candidate boxes
flat = torch.cat([o.reshape(o.shape[0], -1, o.shape[-1]) for o in outs], dim=1)
print(tuple(flat.shape))   # (1, 10647, 8) = 3 * (13*13 + 26*26 + 52*52) candidates
```

So you need two layouts. For the **loss**, keep the grid shape $(b, A, h, w, P)$, because each target is assigned to a specific cell and anchor. For **inference**, flatten everything into one list of candidates, drop the low scores, and run non-maximum suppression (removing overlapping duplicates).

## Decoding: turning raw outputs into boxes

The raw outputs can be any number. YOLOv2 and YOLOv3 decode them so each prediction stays close to "its" cell and anchor:

$$
b_x = (\sigma(t_x) + c_x)\,s, \quad b_y = (\sigma(t_y) + c_y)\,s, \quad b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}
$$

Here $(c_x, c_y)$ is the cell index, $s$ is the stride, $(p_w, p_h)$ is the anchor size, and $\sigma$ is the sigmoid. The sigmoid keeps the centre inside its cell, which made early training much more stable. The exponential makes the size a positive scale factor on the anchor, so $t_w = 0$ means "exactly the anchor size".

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

All-zero outputs decode to "an anchor-sized box at the cell centre, 50% sure". That's a handy sanity check for any head. How to pick the anchor sizes themselves is in [designing anchors](/notes/designing-anchors-for-your-dataset/).

## Decoupled heads: one branch for "where", one for "what"

In a coupled head, box regression and classification share the same features. But the two jobs want different things. Classification wants features that look the same wherever you are on the object. Box regression wants features that change sharply near the object's edges. YOLOX (2021) showed that giving each job its own small convolutional branch trains faster and gets better accuracy. Most newer detectors use this **decoupled** design. RetinaNet already had something similar, with separate classification and box subnets.

In my code-along, the first version of this was just three separate $1\times1$ heads for box, objectness and class. Real decoupled heads put two $3\times3$ convolutions in each branch before the final $1\times1$, so each job really gets its own features.

## Anchor-free heads

Anchors come with a lot of settings to tune: their sizes, how many, and the IoU rules for matching them to targets. **Anchor-free** heads (FCOS, YOLOX, YOLOv8) just drop them. Each cell predicts at most one object, and the box is described by the distances from the cell centre to its four sides: left, top, right, bottom ($l, t, r, b$), measured in strides. YOLOv8 also drops the objectness branch. The class scores themselves act as the confidence.

YOLOv8 adds one more twist, taken from Generalized Focal Loss. Instead of predicting each distance as a single number, it predicts a **distribution** over 16 bins, and the distance is the expected value. The way I think about it: the network can say "the edge is probably around here, but I'm not sure", which helps when an edge is blurry or hidden.

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

Then the box $= (c_x - l,\; c_y - t,\; c_x + r,\; c_y + b) \times s$. One mistake I made: in my TensorFlow sketch of the YOLOv8 head I wrote $4 \times$ the number of anchors for the box channels. The right count is $4 \times$ `reg_max`, because there are no anchors here, only bins.

## Key takeaways

- A dense head is a small network run at every cell. The simplest version is one $1\times1$ convolution with $A(5+C)$ output channels.
- The channel layout is just a convention you pick. Reshape to $(b, A, h, w, P)$ for the loss and flatten into one candidate list for inference.
- YOLOv2/v3 decoding keeps centres inside their cell with a sigmoid and scales anchors with an exponential. Zero outputs should decode to the anchor itself.
- Decoupled heads give box regression and classification their own features, which trains faster and better.
- Anchor-free heads predict distances to the four sides, and YOLOv8 predicts each distance as a distribution over bins.

## Resources

- Redmon, Farhadi, "YOLOv3: An Incremental Improvement", 2018. [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)
- Ge, Liu, Wang, Li, Sun, "YOLOX: Exceeding YOLO Series in 2021". [arXiv:2107.08430](https://arxiv.org/abs/2107.08430)
- Tian, Shen, Chen, He, "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019. [arXiv:1904.01355](https://arxiv.org/abs/1904.01355)
- Li et al., "Generalized Focal Loss", NeurIPS 2020 (the distribution-over-bins box regression). [arXiv:2006.04388](https://arxiv.org/abs/2006.04388)
- Ultralytics YOLOv8 documentation: [docs.ultralytics.com](https://docs.ultralytics.com/models/yolov8/)
- A YouTube code-along series on object-detection components (detection head, better head, FPN neck), and Aladdin Persson's YOLOv3-from-scratch video ([Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection)).
- Related notes: [YOLOv1](/notes/yolo-v1-detection-as-regression/), [anchors and RPNs](/notes/anchors-and-region-proposal-networks/), [FPN](/notes/feature-pyramid-networks/), [mean average precision](/notes/mean-average-precision/).
