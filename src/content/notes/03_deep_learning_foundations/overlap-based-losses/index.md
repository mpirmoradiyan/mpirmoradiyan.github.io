---
title: "Overlap-based losses: IoU, GIoU, DIoU, CIoU, Dice and Tversky"
slug: overlap-based-losses
date: 2024-11-28
topic: Deep Learning Foundations
tags: [loss-functions, iou, giou, diou, ciou, dice, tversky, object-detection, segmentation]
summary: "How losses built on overlap measure what detection and segmentation are actually judged on, and how each IoU variant fixes a weakness of the one before it."
sources:
  - title: "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
    author: "Hamid Rezatofighi et al."
    url: "https://arxiv.org/abs/1902.09630"
    type: paper
  - title: "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
    author: "Zhaohui Zheng et al."
    url: "https://arxiv.org/abs/1911.08287"
    type: paper
  - title: "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    author: "Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi"
    url: "https://arxiv.org/abs/1606.04797"
    type: paper
  - title: "Tversky loss function for image segmentation using 3D fully convolutional deep networks"
    author: "Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour"
    url: "https://arxiv.org/abs/1706.05721"
    type: paper


status: draft
---

Detection and segmentation models are judged on overlap: does the predicted box or mask cover the same region as the ground truth? Yet classic training losses look at something else, such as per-coordinate L1/L2 errors or per-pixel cross-entropy. Overlap-based losses close that gap by optimizing the overlap itself. This note walks through the IoU family for boxes, then Dice and Tversky for masks.

## IoU and its blind spot

For a predicted region $A$ and a ground-truth region $B$, Intersection over Union is

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|},
$$

where $|\cdot|$ is area (or pixel count). It is 1 for a perfect match and 0 for no overlap. It does not depend on scale: a 5-pixel error matters much more on a small box than on a large one, which is exactly how detection is evaluated. The natural loss is $\mathcal{L}_{\text{IoU}} = 1 - \text{IoU}$.

The blind spot: when the boxes do not overlap, IoU is 0 no matter how far apart they are. The loss is flat, so its gradient is zero and gives no hint which way to move the box. Early in training this is common.

## GIoU: add the empty space of the enclosing box

Generalized IoU (Rezatofighi et al., 2019) uses $C$, the smallest box that encloses both $A$ and $B$:

$$
\text{GIoU} = \text{IoU} - \frac{|C| - |A \cup B|}{|C|}.
$$

The second term is the share of $C$ that neither box covers. When the boxes are far apart, $C$ is mostly empty and GIoU approaches $-1$. As they move closer the penalty shrinks, so there is a useful gradient even with zero overlap. GIoU lies in $(-1, 1]$, and the loss is $1 - \text{GIoU}$.

## DIoU: pull the centres together directly

GIoU tends to first grow the predicted box until it touches the target, then shrink it, which is slow. Distance-IoU (Zheng et al., 2020) penalizes the distance between centres instead:

$$
\text{DIoU} = \text{IoU} - \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2},
$$

where $\rho$ is the Euclidean distance between the two box centres and $c$ is the diagonal length of the enclosing box $C$. Dividing by $c^2$ keeps the term between 0 and 1 and makes it scale-invariant.

## CIoU: also match the shape

Complete IoU (same paper) adds an aspect-ratio term:

$$
\text{CIoU} = \text{DIoU} - \alpha v, \qquad
v = \frac{4}{\pi^2}\Big(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\Big)^2, \qquad
\alpha = \frac{v}{(1 - \text{IoU}) + v}.
$$

Here $w, h$ are the width and height of the predicted box and $w^{gt}, h^{gt}$ those of the ground truth. $v$ is 0 when the aspect ratios match. Note that $\alpha$ is not a free constant: it is computed from IoU and $v$, so the shape term matters more once the boxes already overlap well. My original notebook used a fixed $\alpha = 0.5$, which is not the published definition. CIoU loss is the default box loss in several YOLO versions.

```python
import math

def box_ious(a, b):
    """a, b: boxes as (x1, y1, x2, y2). Returns IoU, GIoU, DIoU, CIoU."""
    area = lambda r: max(r[2] - r[0], 0) * max(r[3] - r[1], 0)
    inter = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - inter
    iou = inter / union
    c = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))  # enclosing box
    giou = iou - (area(c) - union) / area(c)
    rho2 = ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) / 4
    c2 = (c[2] - c[0]) ** 2 + (c[3] - c[1]) ** 2
    diou = iou - rho2 / c2
    v = 4 / math.pi**2 * (math.atan((b[2] - b[0]) / (b[3] - b[1]))
                          - math.atan((a[2] - a[0]) / (a[3] - a[1]))) ** 2
    alpha = v / ((1 - iou) + v) if v > 0 else 0.0
    return iou, giou, diou, diou - alpha * v

print(box_ious((100, 100, 200, 200), (150, 150, 250, 250)))  # 0.143, -0.079, 0.032, 0.032
print(box_ious((0, 0, 1, 1), (1.5, 1.5, 2, 2)))              # 0.0, -0.688, -0.391, -0.391
```

![Two 100 by 100 squares offset by 50 pixels in x and y: IoU 0.1429, GIoU -0.0794, DIoU 0.0317, CIoU 0.0317](fig-overlapping-boxes.png)

For the two offset squares, the union is 17,500 and the enclosing box is 22,500, so GIoU subtracts $5{,}000/22{,}500 \approx 0.22$ from IoU. Both are squares, so $v = 0$ and CIoU equals DIoU.

![A unit square at the origin and a small square near (1.75, 1.75) that do not overlap: IoU 0, GIoU -0.6875, DIoU -0.3906, CIoU -0.3906](fig-disjoint-boxes.png)

For the disjoint pair, IoU is stuck at 0, while GIoU and DIoU are negative and would grow as the boxes approach. That is exactly the signal plain IoU loss lacks.

## Dice and Tversky for segmentation masks

For masks, the same overlap idea is computed over pixels. With a ground-truth mask $y_i \in \{0,1\}$ and predicted probabilities $p_i \in [0,1]$, the "soft" counts are

$$
TP = \sum_i p_i y_i, \qquad FP = \sum_i p_i (1 - y_i), \qquad FN = \sum_i (1 - p_i) y_i .
$$

Using probabilities instead of thresholded masks keeps everything differentiable.

**Soft Jaccard (IoU)**: $J = TP / (TP + FP + FN)$, loss $1 - J$.

**Dice** (the F1 score of the mask):

$$
\text{Dice} = \frac{2\,TP}{2\,TP + FP + FN} = \frac{2|A\cap B|}{|A| + |B|}.
$$

Dice and Jaccard rank predictions the same way ($\text{Dice} = 2J/(1+J)$), but Dice gives a larger value for the same partial overlap. Both are popular for medical and aerial segmentation, where the foreground may be a tiny fraction of the image. Pixel-wise cross-entropy would be dominated by the background there, while Dice only looks at the foreground.

**Tversky** generalizes Dice by weighting the two error types separately:

$$
\text{TI} = \frac{TP}{TP + \alpha\,FP + \beta\,FN}.
$$

With $\alpha = \beta = 0.5$ it is Dice; with $\alpha = \beta = 1$ it is Jaccard. Setting $\beta > \alpha$ (for example $\alpha = 0.3$, $\beta = 0.7$) penalizes missed foreground more than false alarms, which improves recall on small structures.

```python
import numpy as np

def overlap_losses(y, p, alpha=0.3, beta=0.7, smooth=1e-7):
    y, p = y.ravel(), p.ravel()
    tp, fp, fn = np.sum(p * y), np.sum(p * (1 - y)), np.sum((1 - p) * y)
    jaccard = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - jaccard, 1 - dice, 1 - tversky

y = np.array([[1, 0, 1], [0, 1, 0]])
p = np.array([[0.8, 0.1, 0.6], [0.2, 0.7, 0.3]])
print(overlap_losses(y, p))   # about (0.417, 0.263, 0.278)
```

The small `smooth` term avoids $0/0$ on empty masks. In practice these losses are often added to cross-entropy or focal loss. The pixel-wise term gives stable gradients early in training, and the overlap term matches the evaluation metric.

## Key takeaways

- Overlap losses optimize the same quantity that detection and segmentation are evaluated on, and they are scale-invariant.
- Plain IoU loss has zero gradient for non-overlapping boxes; GIoU fixes this with the enclosing box.
- DIoU adds a normalized centre-distance penalty for faster convergence; CIoU adds an aspect-ratio term whose weight $\alpha$ is computed, not fixed.
- Dice and Jaccard work on soft pixel counts and handle tiny foregrounds well; Tversky weights false positives and false negatives separately.

## Resources

- Hamid Rezatofighi et al., "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression" (CVPR 2019): https://arxiv.org/abs/1902.09630
- Zhaohui Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression" (AAAI 2020), source of DIoU and CIoU: https://arxiv.org/abs/1911.08287
- Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (2016), Dice loss: https://arxiv.org/abs/1606.04797
- Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour, "Tversky loss function for image segmentation using 3D fully convolutional deep networks" (2017): https://arxiv.org/abs/1706.05721
- torchvision docs, box losses (`generalized_box_iou_loss`, `distance_box_iou_loss`, `complete_box_iou_loss`): https://pytorch.org/vision/stable/ops.html
