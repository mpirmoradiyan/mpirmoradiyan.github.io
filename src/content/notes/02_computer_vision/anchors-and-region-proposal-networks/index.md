---
title: "Anchors and Region Proposal Networks: from a dense grid to a few good boxes"
slug: anchors-and-region-proposal-networks
date: 2025-12-17
topic: Computer Vision
tags: [anchors, rpn, faster-rcnn, object-detection, nms, roialign, box-regression, numpy, pytorch]
summary: "Why detectors put fixed boxes on a grid, how an RPN scores and nudges them, how they get labelled for training, and how the best ones go on to the second stage."
sources:
  - title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    author: "Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun"
    url: "https://arxiv.org/abs/1506.01497"
    type: paper
  - title: "Feature Pyramid Networks for Object Detection"
    author: "Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie"
    url: "https://arxiv.org/abs/1612.03144"
    type: paper
  - title: "Mask R-CNN"
    author: "Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick"
    url: "https://arxiv.org/abs/1703.06870"
    type: paper
  - title: "Rich feature hierarchies for accurate object detection and semantic segmentation (R-CNN), appendix on box regression"
    author: "Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik"
    url: "https://arxiv.org/abs/1311.2524"
    type: paper
  - title: "Fast R-CNN"
    author: "Ross Girshick"
    url: "https://arxiv.org/abs/1504.08083"
    type: paper
  - title: "torchvision.ops (nms, roi_align, roi_pool)"
    author: "PyTorch"
    url: "https://pytorch.org/vision/stable/ops.html"
    type: docs
  - title: "Detectron2"
    author: "Facebook AI Research"
    url: "https://github.com/facebookresearch/detectron2"
    type: repo


status: draft
---

This one confused me for a while. A CNN gives you a grid of feature vectors, one per cell. But a detector has to give you a short list of boxes, and those boxes can be any size and shape. So how do you get from one to the other? Anchors are the bridge. They're a fixed set of boxes placed on the grid ahead of time, and the network only has to score them and nudge them a bit. The Region Proposal Network (RPN) from Faster R-CNN does exactly that. What comes out is a few hundred to a few thousand candidate boxes, and the second stage of a two-stage detector then classifies and refines those.

## Why we need anchors at all

The obvious first idea is to let each grid cell predict one box from scratch. Two things go wrong. First, objects come in very different sizes and shapes. One box per cell can't cover a tall person and a wide car that happen to be centred at the same spot. Second, predicting raw pixel coordinates is hard to learn. The numbers can be huge, and they depend on the image size.

An **anchor** is a box $(x_0, y_0, x_1, y_1)$ that's fixed in advance, with a known position, size and aspect ratio (height divided by width). For each anchor, the network predicts:

- an **objectness** score: is there an object here (any class)?
- four **deltas**: how to shift and stretch this anchor so it fits that object.

The way I think about it: learning a small fix to a sensible starting box is much easier than drawing a box out of nothing. And with several starting boxes per cell, different shapes at the same location can go to different anchors. One thing that tripped me up: anchors aren't learned, and they don't look at the image at all. They only depend on the feature map size, the stride and the sizes and ratios you pick.

## Putting anchors on the grid

A feature map with stride $s$ has one cell every $s$ input pixels. At each cell we put $A$ anchors: one per aspect ratio $r = h/w$, all with the same area $S^2$:

$$
w = \frac{S}{\sqrt{r}}, \qquad h = S\sqrt{r}
$$

so $wh = S^2$ and $h/w = r$.

```python
import numpy as np

def make_anchors(feat_h, feat_w, stride, size, ratios=(0.5, 1.0, 2.0), offset=0.5):
    r = np.asarray(ratios, dtype=float)              # aspect ratio = height / width
    w, h = size / np.sqrt(r), size * np.sqrt(r)      # every anchor has area size**2
    base = np.stack([-w, -h, w, h], axis=1) / 2      # (A, 4), centred on (0, 0)
    xs = (np.arange(feat_w) + offset) * stride       # cell centres in image pixels
    ys = (np.arange(feat_h) + offset) * stride
    cx, cy = np.meshgrid(xs, ys)                      # (H, W) each
    shifts = np.stack([cx, cy, cx, cy], axis=-1).reshape(-1, 1, 4)
    return (shifts + base).reshape(-1, 4)            # (H*W*A, 4), all anchors of one cell together

img = 512
levels = {"P2": (4, 32), "P3": (8, 64), "P4": (16, 128), "P5": (32, 256), "P6": (64, 512)}
total = 0
for name, (stride, size) in levels.items():
    a = make_anchors(img // stride, img // stride, stride, size)
    total += len(a)
    print(f"{name}: stride {stride:2d}, size {size:3d} -> {len(a):6d} anchors")
print("total:", total)
print("first cell of P4:\n", make_anchors(32, 32, 16, 128)[:3].round(1))
```

```
P2: stride  4, size  32 ->  49152 anchors
P3: stride  8, size  64 ->  12288 anchors
P4: stride 16, size 128 ->   3072 anchors
P5: stride 32, size 256 ->    768 anchors
P6: stride 64, size 512 ->    192 anchors
total: 65472
first cell of P4:
 [[-82.5 -37.3  98.5  53.3]
 [-56.  -56.   72.   72. ]
 [-37.3 -82.5  53.3  98.5]]
```

A few small things worth knowing:

- **Where the centre goes.** Cell $(i, j)$ can be centred at $((j + 0.5)s, (i + 0.5)s)$ or at $(js, is)$. Libraries don't agree on this. It barely changes accuracy, but you have to use the same choice for anchors and deltas at training and test time.
- **One map or a pyramid.** The original Faster R-CNN used one stride-16 map with 3 sizes × 3 ratios = 9 anchors per cell. With an [FPN](/notes/feature-pyramid-networks/), each level gets one size and 3 ratios, because the level already takes care of scale.
- **Anchors can stick out of the image.** Near the edges, big anchors go past the border. That's fine. The final boxes get clipped later.
- That's about 65,000 anchors for a $512\times512$ image, and 75% of them sit on $P_2$. So the finest level is where most of the cost is.

![Anchors on a synthetic 256 by 256 image at stride 16 with size 64: two ground-truth boxes, positive anchors in green, ignored anchors in orange, negative anchor centres as small grey dots](fig-anchor-labels.png)

## The RPN head

The head is surprisingly small. It's a $3\times3$ convolution (a little sliding window over the features), followed by two $1\times1$ convolutions. One gives $A$ objectness logits per cell (raw scores before a sigmoid), and the other gives $4A$ deltas. With an FPN, the same head runs on every level.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class RPNHead(nn.Module):
    def __init__(self, channels=256, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.objectness = nn.Conv2d(channels, num_anchors, 1)     # one logit per anchor
        self.deltas = nn.Conv2d(channels, num_anchors * 4, 1)     # (dx, dy, dw, dh) per anchor

    def forward(self, levels):
        out = []
        for x in levels:
            t = F.relu(self.conv(x))
            n, _, h, w = x.shape                                   # reorder to match make_anchors
            logits = self.objectness(t).permute(0, 2, 3, 1).reshape(n, -1)
            deltas = self.deltas(t).permute(0, 2, 3, 1).reshape(n, -1, 4)
            out.append((logits, deltas))
        return out

head = RPNHead()
for logits, deltas in head([torch.randn(1, 256, 32, 32), torch.randn(1, 256, 16, 16)]):
    print(tuple(logits.shape), tuple(deltas.shape))
```

```
(1, 3072) (1, 3072, 4)
(1, 768) (1, 768, 4)
```

Watch out for the `permute` before `reshape`. It puts all $A$ predictions for one cell next to each other, which is the same order `make_anchors` uses. If the orders don't match, predictions get paired with the wrong anchors and nothing warns you. Also, the RPN is **class-agnostic**: it only says "object" or "not object". The actual classes come in the second stage.

## Writing a box relative to an anchor

The network never predicts coordinates directly. It predicts deltas in a normalised form (this idea comes from R-CNN's box regression). Write each box as a centre $(x, y)$, width $w$ and height $h$. Subscript $a$ means the anchor and $g$ means the ground truth (the real box):

$$
t_x = \frac{x_g - x_a}{w_a},\quad t_y = \frac{y_g - y_a}{h_a},\quad t_w = \log\frac{w_g}{w_a},\quad t_h = \log\frac{h_g}{h_a}
$$

Why these formulas? Dividing the shift by the anchor size makes a 5-pixel shift on a small anchor look the same as a 50-pixel shift on a big one. The log turns "twice as wide" and "half as wide" into $+0.69$ and $-0.69$, nicely balanced around 0, and it also means widths always come out positive when you decode. Decoding just runs the formulas backwards. Real code also clamps $t_w, t_h$ (for example at $\log(1000/16)$) so one bad prediction can't blow up through `exp`.

```python
def to_cxcywh(b):
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return b[:, 0] + w / 2, b[:, 1] + h / 2, w, h

def encode(anchors, gt):
    ax, ay, aw, ah = to_cxcywh(anchors)
    gx, gy, gw, gh = to_cxcywh(gt)
    return np.stack([(gx - ax) / aw, (gy - ay) / ah, np.log(gw / aw), np.log(gh / ah)], 1)

def decode(anchors, d, max_log=np.log(1000 / 16)):
    ax, ay, aw, ah = to_cxcywh(anchors)
    x, y = ax + d[:, 0] * aw, ay + d[:, 1] * ah
    w, h = aw * np.exp(np.minimum(d[:, 2], max_log)), ah * np.exp(np.minimum(d[:, 3], max_log))
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)

anchor = np.array([[100., 100., 164., 164.]])
gt = np.array([[110., 90., 190., 150.]])
t = encode(anchor, gt)
print("deltas:", t.round(3), " round trip:", decode(anchor, t))
```

```
deltas: [[ 0.281 -0.188  0.223 -0.065]]  round trip: [[110.  90. 190. 150.]]
```

## Labelling anchors for training

To train, every anchor needs a target. Faster R-CNN decides it using Intersection over Union (IoU), which is how much two boxes overlap, between each anchor and each ground-truth box (more on IoU in the [overlap-based losses note](/notes/overlap-based-losses/)):

| Best IoU with any ground truth | Label |
|---|---|
| $\ge 0.7$ | positive (1) |
| $< 0.3$ | negative (0) |
| in between | ignored (−1), no loss |

There's one extra rule that's easy to miss: each ground-truth box also makes its best-matching anchor(s) positive, even if the IoU is below 0.7. Without this, an object with a strange size or position might get no positive anchor at all, and the network could never learn it.

```python
def box_iou(a, b):                                    # (N, 4), (M, 4) -> (N, M)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)

def label_anchors(anchors, gt, hi=0.7, lo=0.3):
    iou = box_iou(anchors, gt)
    match, best = iou.argmax(1), iou.max(1)
    labels = np.full(len(anchors), -1)
    labels[best < lo] = 0
    labels[best >= hi] = 1
    for g in range(len(gt)):                          # every object keeps its best anchor(s)
        top = np.flatnonzero(iou[:, g] == iou[:, g].max())
        labels[top], match[top] = 1, g
    return labels, match

anchors = make_anchors(16, 16, stride=16, size=64)   # a 256 x 256 image
gt = np.array([[40., 50., 110., 110.], [150., 120., 190., 230.]])
labels, match = label_anchors(anchors, gt)
print("positive:", (labels == 1).sum(), " negative:", (labels == 0).sum(),
      " ignored:", (labels == -1).sum())
```

```
positive: 4  negative: 706  ignored: 58
```

That's a huge imbalance: a handful of positives against hundreds of negatives (tens of thousands in a real image). So each image only gives a **sample** of 256 anchors, at most half positive, topped up with random negatives. The loss is binary cross-entropy on the objectness of those sampled anchors, plus a smooth-L1 (or similar) loss on the deltas of the **positive** anchors only, with targets `encode(anchors[pos], gt[match[pos]])`. Negatives don't get a box loss, because there's no box for them to move towards.

```python
def sample(labels, batch=256, pos_fraction=0.5, rng=np.random.default_rng(0)):
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n_pos = min(len(pos), int(batch * pos_fraction))
    n_neg = min(len(neg), batch - n_pos)
    return rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)

pos_idx, neg_idx = sample(labels)
targets = encode(anchors[pos_idx], gt[match[pos_idx]])
print(len(pos_idx), "positives,", len(neg_idx), "negatives; targets", targets.shape)
```

```
4 positives, 252 negatives; targets (4, 4)
```

## From scores to proposals: top-k and NMS

At inference, every anchor gets a score and a decoded box, and most of them are background or duplicates. The usual clean-up for each image goes like this:

1. On each level, keep the top-k anchors by objectness (say 1000 to 2000).
2. Decode them into boxes and clip the boxes to the image.
3. Throw away boxes that are nearly empty (width or height close to 0).
4. Run **non-maximum suppression (NMS)**: take the highest-scoring box, delete every other box that overlaps it with IoU above a threshold, and repeat. The RPN uses 0.7, which is loose on purpose, because at this stage not missing objects matters more than having a few extra boxes.
5. Keep the best ~1000 proposals across all levels.

```python
def nms(boxes, scores, iou_thresh=0.7):
    order, keep = np.argsort(-scores), []
    while order.size:
        i, order = order[0], order[1:]
        keep.append(i)
        order = order[box_iou(boxes[i:i + 1], boxes[order])[0] <= iou_thresh]
    return np.array(keep)

rng = np.random.default_rng(1)
jitter = rng.normal(0, 2, (20, 4))                    # 20 noisy copies of two objects
boxes = np.vstack([gt[0] + jitter[:10], gt[1] + jitter[10:]])
scores = rng.random(20)
print("kept", len(nms(boxes, scores)), "of", len(boxes))
```

```
kept 2 of 20
```

This greedy NMS has a weak spot. If two real objects overlap a lot (a crowd, or things packed tightly together), they can wipe each other out. Soft-NMS is one fix: instead of deleting overlapping boxes, it just lowers their scores.

## How proposals go into the second stage

The second stage (the "RoI heads" in Fast R-CNN and Mask R-CNN; RoI means region of interest) takes each proposal, picks a pyramid level based on the box size (see the [FPN note](/notes/feature-pyramid-networks/)), and cuts out a fixed-size feature grid for it. That's usually $7\times7$ for boxes and $14\times14$ for masks. During training, the proposals are labelled again against the ground truth (IoU $\ge 0.5$ means foreground) and sampled again, usually 512 per image with up to 25% foreground. The ground-truth boxes are also added to the proposals, so every object has at least one perfect example. The head then predicts a class and a box fix for that class, and in Mask R-CNN, a mask too.

How you cut out that feature grid matters a lot if you want precise boxes and masks:

- **RoIPool** (Fast R-CNN) rounds the box to whole feature cells, splits it into bins, and rounds again. On a stride-16 map, one rounding step can move a box edge by up to half a cell, which is 8 input pixels, and the bin rounding adds more.
- **RoIAlign** (Mask R-CNN) keeps the fractional coordinates and samples each bin with bilinear interpolation (a weighted average of the four nearest cells). That removes the misalignment, and in the Mask R-CNN paper it clearly improved mask accuracy.

```python
from torchvision.ops import roi_align, roi_pool

feat = torch.arange(16.).repeat(16, 1)[None, None]    # value = x coordinate of the cell
box = torch.tensor([[0, 2.6, 2.6, 9.4, 9.4]])        # (batch index, x0, y0, x1, y1)
print("RoIPool :", roi_pool(feat, box, output_size=4)[0, 0, 0].tolist())
print("RoIAlign:", [round(v, 2) for v in roi_align(feat, box, output_size=4, aligned=True)[0, 0, 0].tolist()])
```

```
RoIPool : [4.0, 6.0, 8.0, 9.0]
RoIAlign: [2.95, 4.65, 6.35, 8.05]
```

This toy example made it click for me. Cell $j$ holds the value $j$ and is centred at $x = j + 0.5$, so the true feature at position $x$ is $x - 0.5$. Four bins from $x = 2.6$ to $9.4$ have centres at 3.45, 5.15, 6.85 and 8.55, so the right answer is 2.95, 4.65, 6.35 and 8.05: evenly spaced, 1.7 apart. RoIAlign gets exactly that. RoIPool snaps the box to cells 3 to 9, takes the max in each bin, and gives uneven steps.

## Key takeaways

- Anchors are fixed boxes on the feature grid. The network just scores them and predicts small, normalised fixes.
- Deltas are $(\Delta x / w_a, \Delta y / h_a, \log(w/w_a), \log(h/h_a))$, so the targets don't depend on object size.
- Training labels come from IoU: positive if $\ge 0.7$, negative if $< 0.3$, plus the best anchor for every object. Then 256 anchors are sampled per image.
- At inference: top-k per level, decode, clip, NMS at 0.7, keep about 1000 proposals.
- The second stage cuts out a fixed grid for each proposal. RoIAlign avoids the rounding errors that RoIPool has.

## Resources

- Ren, He, Girshick, Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017 (anchors per pyramid level). [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- He, Gkioxari, Dollár, Girshick, "Mask R-CNN", ICCV 2017 (RoIAlign). [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Girshick, Donahue, Darrell, Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation", CVPR 2014, appendix C (box regression parameterisation). [arXiv:1311.2524](https://arxiv.org/abs/1311.2524)
- Girshick, "Fast R-CNN", ICCV 2015 (RoIPool, RoI sampling). [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)
- Bodla, Singh, Chellappa, Davis, "Soft-NMS: Improving Object Detection With One Line of Code", ICCV 2017. [arXiv:1704.04503](https://arxiv.org/abs/1704.04503)
- `torchvision.ops` (`nms`, `roi_align`, `roi_pool`, `box_iou`): [PyTorch docs](https://pytorch.org/vision/stable/ops.html)
- One full implementation to read: Detectron2's `modeling/proposal_generator/rpn.py` and `modeling/anchor_generator.py`. [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
