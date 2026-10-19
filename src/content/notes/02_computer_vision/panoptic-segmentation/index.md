---
title: "Panoptic segmentation: things, stuff and one label per pixel"
slug: panoptic-segmentation
date: 2025-09-12
topic: Computer Vision
tags: [panoptic-segmentation, instance-segmentation, semantic-segmentation, mask-rcnn, fpn, panoptic-quality, pytorch]
summary: "What panoptic segmentation asks for, how Panoptic FPN gets there with an instance head and a semantic head, how the two outputs are merged, and how Panoptic Quality scores it."
sources:
  - title: "Panoptic Segmentation"
    author: "Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, Piotr Dollár"
    url: "https://arxiv.org/abs/1801.00868"
    type: paper
  - title: "Panoptic Feature Pyramid Networks"
    author: "Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár"
    url: "https://arxiv.org/abs/1901.02446"
    type: paper
  - title: "Mask R-CNN"
    author: "Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick"
    url: "https://arxiv.org/abs/1703.06870"
    type: paper
  - title: "COCO data format: panoptic segmentation"
    author: "COCO Consortium"
    url: "https://cocodataset.org/#format-data"
    type: docs
  - title: "torchvision detection models"
    author: "PyTorch"
    url: "https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection"
    type: docs
  - title: "Detectron2"
    author: "Facebook AI Research"
    url: "https://github.com/facebookresearch/detectron2"
    type: repo


status: draft
---

Look at a street photo and ask two questions about every pixel: what is it, and which object does it belong to? Semantic segmentation only answers the first one. It tells you a pixel is "person", but not *which* person. Instance segmentation answers the second one for people and cars, but it ignores the sky, the road and the grass. Panoptic segmentation asks for both at once: every pixel gets a class, and pixels that belong to countable objects also get an instance ID. Of the three tasks, it's the closest to a full description of a scene, which is why people care about it for driving, mapping and robotics.

## Things and stuff

Kirillov et al. (2019) split classes into two kinds:

- **Things** are objects you can count, with a clear shape: person, car, bicycle, dog. Each one gets its own instance ID.
- **Stuff** is shapeless regions you can't really count: sky, grass, road, water. All pixels of one stuff class share one segment.

The output is one map where each pixel has a pair $(c, z)$: class $c$ and instance ID $z$. For stuff, $z$ isn't used. A handy trick packs the pair into a single integer:

$$
\text{id} = c \cdot L + z
$$

where $L$ is a "label divisor" larger than the number of instances in any image (for example 1000). Then $c = \lfloor \text{id} / L \rfloor$ and $z = \text{id} \bmod L$. One important rule: **no overlaps**. Each pixel belongs to exactly one segment. That's different from instance segmentation, where masks are allowed to overlap.

## Panoptic FPN: two heads on one backbone

The simplest model that works well is Panoptic FPN (Kirillov, Girshick, He and Dollár, 2019). The idea is pretty simple: take Mask R-CNN and bolt on one small extra branch:

| Branch | Output | Loss |
|---|---|---|
| Backbone + FPN (shared) | Feature maps at strides 4, 8, 16, 32 | none |
| Instance head (Mask R-CNN) | For each detected thing: box (4 numbers), class scores, 28×28 mask | classification + box regression + mask BCE |
| Semantic head | Per-pixel scores for the stuff classes + one "other" class for all thing pixels | per-pixel cross-entropy |
| Panoptic merge | One ID map, H×W | none (a fixed heuristic) |

Both heads are trained together, using a weighted sum of their losses. The merge step has nothing to learn.

The semantic head in the paper is small. Each FPN level gets upsampled to 1/4 of the input size with a few conv + upsample stages, the results are added up, and a 1×1 convolution predicts the classes. Here's a simpler version (one conv per level instead of several stages):

```python
import torch.nn as nn
import torch.nn.functional as F

class SemanticHead(nn.Module):
    def __init__(self, in_ch=256, mid=128, num_classes=4):   # 3 stuff + "other"
        super().__init__()
        self.levels = ["0", "1", "2", "3"]                     # strides 4, 8, 16, 32
        self.convs = nn.ModuleDict({k: nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1), nn.GroupNorm(32, mid), nn.ReLU())
            for k in self.levels})
        self.classifier = nn.Conv2d(mid, num_classes, 1)

    def forward(self, feats, out_size):
        size = feats["0"].shape[-2:]                           # 1/4 resolution
        x = sum(F.interpolate(self.convs[k](feats[k]), size=size,
                              mode="bilinear", align_corners=False)
                for k in self.levels)
        logits = self.classifier(x)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
```

`feats` is the dictionary that torchvision's `resnet_fpn_backbone` (or the `backbone` of `maskrcnn_resnet50_fpn`) returns. Using all the levels, not just the finest one, gives the head both fine detail and wider context.

## What the data loader has to return

For training, each image needs instance targets plus a full per-pixel stuff map. In PyTorch, a detection-style dataset usually returns the image and a dictionary:

```python
target = {
    "boxes":    boxes,      # FloatTensor[N, 4], x1, y1, x2, y2 in pixels
    "labels":   labels,     # Int64Tensor[N], thing class IDs (0 = background in torchvision)
    "masks":    masks,      # UInt8/BoolTensor[N, H, W], one binary mask per thing
    "semantic": semantic,   # LongTensor[H, W], stuff IDs, "other" for thing pixels, 255 = ignore
}
```

Since images come in different sizes, the `collate_fn` returns lists instead of stacked tensors. When you resize the semantic map, use **nearest-neighbour** interpolation. Otherwise class IDs get blended into in-between numbers that mean nothing (class 2.5 isn't a thing). Real annotations usually come in the COCO panoptic format: one PNG per image where each segment has its own colour-encoded ID, plus a JSON file listing each segment's category and whether it's a thing or stuff.

## Merging the two outputs

The two heads can disagree. An instance mask and the semantic map might both claim the same pixel, and instance masks can overlap each other. Panoptic FPN sorts this out with a simple step-by-step rule:

1. Sort the detected instances by confidence and drop the ones below a score threshold.
2. Paste the instance masks one at a time. Skip an instance if most of its mask (say, more than half) is already taken by higher-scoring instances; otherwise keep only its free pixels.
3. Fill whatever is left with the semantic head's stuff prediction. Stuff regions that are too small (the paper uses an area threshold) get removed and left unlabeled.

```python
import torch

def merge_panoptic(inst_masks, inst_scores, inst_classes, sem_pred,
                   score_thr=0.5, overlap_thr=0.5, min_stuff_area=4096, L=1000,
                   other_id=0):
    H, W = sem_pred.shape
    pan = torch.full((H, W), -1, dtype=torch.long)         # -1 = unlabeled
    taken = torch.zeros((H, W), dtype=torch.bool)
    inst_id = 1
    for i in torch.argsort(inst_scores, descending=True).tolist():
        if inst_scores[i] < score_thr:
            break
        m = inst_masks[i] > 0.5
        area = m.sum()
        if area == 0 or (m & taken).sum() / area > overlap_thr:
            continue
        free = m & ~taken
        pan[free] = int(inst_classes[i]) * L + inst_id
        taken |= free
        inst_id += 1
    for c in sem_pred.unique():
        if c == other_id:                                   # "other" = thing pixels
            continue
        region = (sem_pred == c) & ~taken
        if region.sum() >= min_stuff_area:
            pan[region] = int(c) * L                        # stuff: instance ID 0
    return pan
```

For real use, it's better to take a library version (Detectron2, MMDetection) of this step. Those handle thing and stuff ID spaces, ignore labels and the odd edge cases properly.

## Measuring it: Panoptic Quality

Kirillov et al. also came up with one metric for the whole task. A predicted segment and a ground-truth segment of the same class **match** if their IoU is above 0.5. This part confused me at first: why does 0.5 make matches unique? Because segments can't overlap, two predictions can't both cover more than half of the same ground-truth segment. Matched pairs are true positives (TP); unmatched predictions are false positives (FP); unmatched ground-truth segments are false negatives (FN). Then, for each class,

$$
\text{PQ} = \frac{\sum_{(p,g) \in TP} \text{IoU}(p, g)}{|TP| + \tfrac{1}{2}|FP| + \tfrac{1}{2}|FN|}
$$

This splits nicely into two parts:

$$
\text{PQ} = \underbrace{\frac{\sum_{TP} \text{IoU}(p,g)}{|TP|}}_{\text{SQ: segmentation quality}} \times \underbrace{\frac{|TP|}{|TP| + \tfrac{1}{2}|FP| + \tfrac{1}{2}|FN|}}_{\text{RQ: recognition quality}}
$$

So basically: SQ is the average IoU of the matched segments, meaning how good the shapes are. RQ is an F1 score, meaning how well the model finds the right segments at all. The final PQ is averaged over classes, and people often report it separately for things (PQ^Th) and stuff (PQ^St). If finding each object matters more than overall pixel accuracy, PQ (or PQ^Th) is a better metric for picking a model than mean IoU or pixel accuracy, because it directly punishes missed objects and made-up ones.

## Key takeaways

- Panoptic segmentation gives every pixel a class, and gives pixels of "things" an instance ID too, with no overlaps.
- Panoptic FPN = Mask R-CNN for things + a small semantic head for stuff, both on one shared FPN backbone.
- A merge step with nothing to learn sorts out conflicts: confident instances go first, then stuff fills the gaps.
- Resize label maps with nearest-neighbour, and return lists of targets from the data loader.
- Panoptic Quality = Segmentation Quality × Recognition Quality, where a match means IoU > 0.5.

## Resources

- Kirillov, He, Girshick, Rother, Dollár, "Panoptic Segmentation", CVPR 2019. [arXiv:1801.00868](https://arxiv.org/abs/1801.00868)
- Kirillov, Girshick, He, Dollár, "Panoptic Feature Pyramid Networks", CVPR 2019. [arXiv:1901.02446](https://arxiv.org/abs/1901.02446)
- He, Gkioxari, Dollár, Girshick, "Mask R-CNN", ICCV 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- COCO panoptic annotation format: [cocodataset.org/#format-data](https://cocodataset.org/#format-data)
- torchvision detection and segmentation models: [PyTorch docs](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- Detectron2 (includes Panoptic FPN and panoptic evaluation): [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
