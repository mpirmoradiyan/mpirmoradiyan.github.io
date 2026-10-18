---
title: "Designing anchors for your own dataset: k-means, best possible recall and evolution"
slug: designing-anchors-for-your-dataset
date: 2025-12-09
topic: Computer Vision
tags: [anchors, k-means, object-detection, yolo, fpn, genetic-algorithm, numpy]
summary: "How to pick anchor sizes and aspect ratios from the boxes in your own data: k-means with a 1 − IoU distance, the best-possible-recall and fitness scores, mutation-based refinement, and how the result has to respect the strides of a feature pyramid."
sources:
  - title: "YOLO9000: Better, Faster, Stronger (YOLOv2), section on dimension clusters"
    author: "Joseph Redmon, Ali Farhadi"
    url: "https://arxiv.org/abs/1612.08242"
    type: paper
  - title: "YOLOv5 documentation: AutoAnchor"
    author: "Ultralytics"
    url: "https://docs.ultralytics.com/yolov5/tutorials/architecture_description/"
    type: docs
  - title: "YOLOv5 repository, utils/autoanchor.py"
    author: "Ultralytics"
    url: "https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py"
    type: repo
  - title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
    author: "Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun"
    url: "https://arxiv.org/abs/1506.01497"
    type: paper
  - title: "Feature Pyramid Networks for Object Detection"
    author: "Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie"
    url: "https://arxiv.org/abs/1612.03144"
    type: paper
  - title: "k-means++: The Advantages of Careful Seeding"
    author: "David Arthur, Sergei Vassilvitskii"
    url: "https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf"
    type: paper


status: draft
---

The default anchors of most detectors were tuned on COCO: everyday photos with people, cars and furniture of every size. If your objects are mostly tiny, or long and thin, or all roughly the same size, those defaults waste capacity and make training harder, because few anchors overlap your objects well. Since anchors are just numbers in a config, it is cheap to fit them to your data. This note covers the standard tools for doing that, and one constraint that the standard tools ignore: the stride of the feature map the anchors sit on. (What anchors are and how they are matched to objects is in the [anchors and RPN note](/notes/anchors-and-region-proposal-networks/).)

## Step 0: measure boxes as the network sees them

Collect the width and height of every ground-truth box in the training set, **after** the resizing, cropping or tiling the training pipeline applies. If images are resized from 2000 to 1000 pixels before entering the network, every box shrinks by half, and anchors fitted to the original sizes will be twice too big. Positions do not matter here, only $(w, h)$ pairs.

For this note we use a synthetic dataset with four families of objects: many small, roughly square objects, wide and tall elongated ones, and a few large ones.

```python
import numpy as np

rng = np.random.default_rng(0)

def family(n, w, h, spread):          # boxes scattered log-normally around a typical (w, h)
    return np.exp(rng.normal(np.log([w, h]), spread, size=(n, 2)))

wh = np.vstack([family(5000, 14, 14, 0.30),    # small, roughly square
                family(1500, 60, 22, 0.25),    # wide
                family(1500, 22, 60, 0.25),    # tall
                family(600, 160, 150, 0.30)])  # few, large
groups = {"small": slice(0, 5000), "wide": slice(5000, 6500),
          "tall": slice(6500, 8000), "large": slice(8000, None)}
print(wh.shape, "median w, h:", np.median(wh, 0).round(1))
```

```
(8600, 2) median w, h: [18.1 17.9]
```

## The right distance: 1 − IoU of shapes

To compare a box with an anchor by shape only, put both at the same centre and compute their IoU. The intersection is then simply $\min(w, w_a) \cdot \min(h, h_a)$:

$$
\text{IoU}_{wh} = \frac{\min(w, w_a)\,\min(h, h_a)}{w h + w_a h_a - \min(w, w_a)\,\min(h, h_a)}
$$

YOLOv2 used $d = 1 - \text{IoU}_{wh}$ as the distance in k-means. The reason is scale. With Euclidean distance on $(w, h)$, a large box that is 20 pixels off counts as much as a small box that is 20 pixels off, even though the first is a near-perfect match and the second a terrible one. Euclidean k-means therefore spends its centres where the absolute spread is largest, which is among the big boxes. IoU is scale-free: being 20% too wide costs the same at every size.

```python
def iou_wh(wh, anchors):                            # (N, 2), (K, 2) -> (N, K)
    inter = (np.minimum(wh[:, None, 0], anchors[None, :, 0]) *
             np.minimum(wh[:, None, 1], anchors[None, :, 1]))
    return inter / (wh.prod(1)[:, None] + anchors.prod(1)[None, :] - inter)

def kmeans(wh, k, distance="iou", iters=100, seed=0):
    r = np.random.default_rng(seed)
    dist = ((lambda a, c: 1 - iou_wh(a, c)) if distance == "iou"
            else (lambda a, c: ((a[:, None] - c[None]) ** 2).sum(-1)))
    centers = wh[[r.integers(len(wh))]]
    while len(centers) < k:                         # k-means++ seeding: spread the start points
        d = dist(wh, centers).min(1)
        centers = np.vstack([centers, wh[r.choice(len(wh), p=d / d.sum())]])
    for _ in range(iters):                          # Lloyd iterations
        assign = dist(wh, centers).argmin(1)
        new = np.array([wh[assign == j].mean(0) for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers[np.argsort(centers.prod(1))]     # sorted by area

def best_of(wh, k, distance, restarts=5):           # k-means finds local optima: keep the best run
    runs = [kmeans(wh, k, distance, seed=s) for s in range(restarts)]
    return max(runs, key=lambda c: iou_wh(wh, c).max(1).mean())

for distance in ["euclidean", "iou"]:
    c = best_of(wh, 9, distance)
    per_group = {g: float(iou_wh(wh[s], c).max(1).mean().round(2)) for g, s in groups.items()}
    print(f"{distance:9s} mean IoU {iou_wh(wh, c).max(1).mean():.3f}", per_group)
    print("          anchors:", c.round().astype(int).tolist())
```

```
euclidean mean IoU 0.691 {'small': 0.65, 'wide': 0.74, 'tall': 0.77, 'large': 0.74}
          anchors: [[15, 15], [23, 47], [52, 23], [22, 65], [77, 23], [24, 87], [143, 134], [156, 218], [243, 148]]
iou       mean IoU 0.750 {'small': 0.77, 'wide': 0.75, 'tall': 0.74, 'large': 0.65}
          anchors: [[12, 11], [12, 18], [19, 12], [19, 19], [52, 22], [22, 52], [75, 24], [24, 77], [170, 156]]
```

The Euclidean version gives three of its nine anchors to the large family, which holds only 7% of the boxes, and a single anchor to the small family, which holds 58%. The IoU version gives the small family four anchors and the large family one. The average box gets a better match, and the per-group numbers are more even.

![Scatter plot of synthetic box widths and heights on log axes, with nine k-means centres from Euclidean distance (squares) crowding the large boxes and nine centres from 1 − IoU distance (crosses) spread across the small, wide and tall families](fig-kmeans-anchors.png)

Two practical points. Plain k-means depends on its starting centres, so seed it with k-means++ and keep the best of several runs. And the mean IoU always rises with $k$, but each extra anchor adds output channels to the head and more anchors to label, so pick $k$ at the elbow of the curve, not at the maximum.

## Scoring a set of anchors: best possible recall and fitness

YOLOv5's AutoAnchor uses a simpler shape match than IoU. For a box and an anchor, take the width ratio and the height ratio, fold each into $(0, 1]$ so that "half as big" and "twice as big" score the same, and keep the worse of the two sides:

$$
m(b, a) = \min\left(\frac{w}{w_a}, \frac{w_a}{w}, \frac{h}{h_a}, \frac{h_a}{h}\right), \qquad m(b) = \max_a m(b, a)
$$

$m(b) = 1$ means some anchor has exactly the box's shape. If $m(b) < 1/4$, no anchor is within a factor of 4 on both sides, and in YOLOv5's matching rule the box would get no anchor at all. Two scores follow:

- **Best possible recall (BPR)**: the fraction of boxes with $m(b) > 1/4$. It is an upper bound on recall from anchor coverage alone. AutoAnchor recomputes anchors when BPR falls below 0.98.
- **Fitness**: the mean of $m(b)$, counting boxes below the threshold as 0. It rewards close matches, not just any match.

```python
def ratio_metric(wh, anchors):
    r = wh[:, None] / anchors[None]                 # (N, K, 2) width and height ratios
    return np.minimum(r, 1 / r).min(2).max(1)       # worse side per anchor, best anchor per box

def best_possible_recall(wh, anchors, thr=0.25):
    return (ratio_metric(wh, anchors) > thr).mean()

def fitness(wh, anchors, thr=0.25):
    m = ratio_metric(wh, anchors)
    return (m * (m > thr)).mean()

def grid(sizes, ratios):                            # FPN-style: every size with every ratio (h / w)
    s, r = np.meshgrid(sizes, ratios)
    return np.stack([s / np.sqrt(r), s * np.sqrt(r)], -1).reshape(-1, 2)

default = grid([32, 64, 128, 256, 512], [0.5, 1.0, 2.0])   # the FPN paper's anchors
print(f"default anchors: BPR {best_possible_recall(wh, default):.3f}, "
      f"fitness {fitness(wh, default):.3f}, mean IoU {iou_wh(wh, default).max(1).mean():.3f}")
```

```
default anchors: BPR 0.992, fitness 0.541, mean IoU 0.397
```

This shows why BPR alone is not enough: the default anchors "cover" 99% of our boxes within a factor of 4, yet the average best IoU is only 0.40. Most small boxes are matched by a 32-pixel anchor that is twice too big on each side.

## Refining by mutation

k-means optimises mean IoU under a fixed number of free $(w, h)$ centres. Often you want to optimise something else (like fitness), or keep a constrained form (like the size × ratio grid an FPN uses). A simple search works for both: randomly scale some parameters by a factor close to 1, keep the change if the score improves, repeat. YOLOv5 calls this a genetic algorithm, but there is no population and no crossover. It is really a (1+1) evolution strategy, a form of random hill climbing.

```python
def evolve(params, build, wh, gens=1000, p=0.9, sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    best, f_best = params.copy(), fitness(wh, build(params))
    for _ in range(gens):
        mutate = rng.random(len(best)) < p                     # which parameters change
        factor = np.where(mutate, rng.normal(1.0, sigma, len(best)), 1.0).clip(0.3, 3.0)
        candidate = best * factor                              # unchanged ones keep factor 1
        f = fitness(wh, build(candidate))
        if f > f_best:
            best, f_best = candidate, f
    return best, f_best

build = lambda p: grid(p[:5], p[5:])                # 5 sizes, then 3 aspect ratios
start = np.array([32, 64, 128, 256, 512, 0.5, 1.0, 2.0])
p, f = evolve(start, build, wh)
print("sizes", p[:5].round(1), "ratios", p[5:].round(2))
print(f"fitness {f:.3f}, mean IoU {iou_wh(wh, build(p)).max(1).mean():.3f}")
```

```
sizes [  14.2   35.4  133.7  183.4 1374.6] ratios [0.46 1.02 2.4 ]
fitness 0.769, mean IoU 0.698
```

Note the line `np.where(mutate, ..., 1.0)`. Multiplying the random factors by the boolean mask instead (`normal(...) * mutate`) is an easy mistake: the parameters that should stay unchanged get the factor 0, which the clip then turns into 0.3, so they shrink every step.

The search moved the smallest size from 32 to 14 and stretched the ratios toward the elongated families. It also shows the method's blind spot. The fifth size drifted to 1375 pixels, and the third and fourth sizes ended up close together. No box is near 1375 pixels, but the score does not care about anchors that match nothing; it only rewards the best anchor per box. A good start (for example from k-means) and sanity bounds on the parameters avoid such results.

## The constraint the shape metrics ignore: stride

Both IoU-of-shapes and the ratio metric pretend that an anchor can sit exactly on every object's centre. In reality anchors sit on a grid with spacing equal to the stride $s$. An object centred between grid points is off by up to $s/2$ in $x$ and in $y$ from the nearest anchor centre. For a square object and a square anchor of the same side $a = m\,s$, that worst-case shift gives

$$
\text{IoU} = \frac{(m - \tfrac12)^2}{2m^2 - (m - \tfrac12)^2}
$$

| anchor side / stride ($m$) | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| worst-case IoU | 0.39 | 0.53 | 0.62 | 0.73 | 0.78 |

With the usual RPN rule (positive at IoU $\ge 0.7$), an anchor needs to be about **6 to 8 times the stride** for every object of its size to reach that threshold. That is why the FPN anchors are $8\times$ the stride (32 on stride 4, up to 512 on stride 64).

This has direct consequences when you map data-driven sizes onto a pyramid:

- **Assign each size to a level where size/stride is roughly 4 to 8.** One level can carry several sizes, which most frameworks allow.
- **If your objects are smaller than about $6\times$ the finest stride**, no anchor tuning fixes the grid. For our 14-pixel objects, even stride 4 gives $m = 3.5$. The usual fixes are to upscale the input (or tile and upscale, see the [tiling note](/notes/tiling-large-images-for-detection/)), to add a finer level, or to lower the positive IoU threshold and rely on the "best anchor per object" rule.
- **Levels with no matching sizes** still cost compute. If no object needs stride 64, that level is only overhead.

## Other pitfalls

- **Aspect-ratio convention.** Some libraries define the ratio as $h/w$, others as $w/h$. Swapping them turns your tall anchors wide. Check with one anchor before training.
- **Outliers.** A few annotation errors (1-pixel boxes, boxes as wide as the image) can pull a k-means centre. Look at the size histogram and drop obvious errors, but do not cut real rare shapes just because they are rare.
- **Fit on the training split** and check BPR on the validation split, like any other fitted hyperparameter.
- **Anchor-free detectors** (FCOS, CenterNet, recent YOLO versions) predict boxes from points without anchor templates. They still have a stride, so the small-object problem in the table above remains.

## Key takeaways

- Measure $(w, h)$ after the training resize, then cluster with $d = 1 - \text{IoU}_{wh}$; Euclidean k-means wastes anchors on the largest boxes.
- Score anchor sets with best possible recall (coverage within a factor of 4) and fitness (closeness); a high BPR alone can hide poor matches.
- Mutation-based search refines anchors under any score and any parameterisation, but it ignores anchors that match nothing, so start from a good guess.
- Anchors must be roughly 6–8× the stride of their level for objects anywhere on the grid to reach IoU 0.7; tiny objects need a finer stride or upscaling, not just smaller anchors.

## Resources

- Redmon, Farhadi, "YOLO9000: Better, Faster, Stronger", CVPR 2017, section 2 "Dimension Clusters" (k-means with 1 − IoU). [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)
- Ultralytics YOLOv5 documentation (AutoAnchor, BPR threshold): [docs.ultralytics.com](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/); implementation in [`utils/autoanchor.py`](https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py)
- Ren, He, Girshick, Sun, "Faster R-CNN", NeurIPS 2015 (IoU thresholds for anchor labels). [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017 (anchor size 8× stride per level). [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- Arthur, Vassilvitskii, "k-means++: The Advantages of Careful Seeding", SODA 2007. [PDF](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- Tian, Shen, Chen, He, "FCOS: Fully Convolutional One-Stage Object Detection", ICCV 2019 (an anchor-free alternative). [arXiv:1904.01355](https://arxiv.org/abs/1904.01355)
