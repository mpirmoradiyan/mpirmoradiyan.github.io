---
title: "Panoptic segmentation: things, stuff and one label per pixel"
slug: panoptic-segmentation
date: 2025-09-12
topic: Computer Vision
tags: [panoptic-segmentation, instance-segmentation, semantic-segmentation, mask-rcnn, fpn, panoptic-quality, pytorch]
summary: "What panoptic segmentation asks for, how Panoptic FPN produces it from an instance head and a semantic head, how the two outputs are merged, and how Panoptic Quality scores the result."
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

Semantic segmentation tells you that a pixel is "person", but not *which* person. Instance segmentation separates each person, but ignores the sky, the road and the grass. Panoptic segmentation asks for both at once: every pixel gets a class, and pixels of countable objects also get an instance ID. It is the closest of the three tasks to a complete description of a scene, which is why it matters for driving, mapping and robotics.

## Things and stuff

Kirillov et al. (2019) split classes into two kinds:

- **Things** are countable objects with a clear shape: person, car, bicycle, tree. Each one gets its own instance ID.
- **Stuff** is amorphous, uncountable regions: sky, grass, road, water. All pixels of one stuff class share one segment.

The output is a single map where each pixel has a pair $(c, z)$: class $c$ and instance ID $z$. For stuff, $z$ is not used. A common trick packs the pair into one integer:

$$
\text{id} = c \cdot L + z
$$

where $L$ is a "label divisor" larger than the number of instances in any image (for example 1000). Then $c = \lfloor \text{id} / L \rfloor$ and $z = \text{id} \bmod L$. A key rule: **no overlaps**. Each pixel belongs to exactly one segment. This is different from instance segmentation, where masks may overlap.

## Panoptic FPN: two heads on one backbone

The simplest strong baseline is Panoptic FPN (Kirillov, Girshick, He and Dollár, 2019). It takes Mask R-CNN and adds one light branch:

| Branch | Output | Loss |
|---|---|---|
| Backbone + FPN (shared) | Feature maps at strides 4, 8, 16, 32 | none |
| Instance head (Mask R-CNN) | For each detected thing: box (4 numbers), class scores, 28×28 mask | classification + box regression + mask BCE |
| Semantic head | Per-pixel scores for the stuff classes + one "other" class for all thing pixels | per-pixel cross-entropy |
| Panoptic merge | One ID map, H×W | none (a fixed heuristic) |

The two heads are trained together with a weighted sum of their losses. The merge step has no parameters.

The semantic head in the paper is small. Each FPN level is upsampled to 1/4 of the input size with a few conv + upsample stages, the results are summed, and a 1×1 convolution predicts the classes. Here is a simplified version (one conv per level instead of several stages):

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

`feats` is the dictionary that torchvision's `resnet_fpn_backbone` (or the `backbone` of `maskrcnn_resnet50_fpn`) returns. Using all levels, not only the finest one, gives the head both detail and context.

## What the data loader must return

For training, each image needs instance targets and a dense stuff map. In PyTorch, a detection-style dataset usually returns the image plus a dictionary:

```python
target = {
    "boxes":    boxes,      # FloatTensor[N, 4], x1, y1, x2, y2 in pixels
    "labels":   labels,     # Int64Tensor[N], thing class IDs (0 = background in torchvision)
    "masks":    masks,      # UInt8/BoolTensor[N, H, W], one binary mask per thing
    "semantic": semantic,   # LongTensor[H, W], stuff IDs, "other" for thing pixels, 255 = ignore
}
```

Because images have different sizes, the `collate_fn` returns lists instead of stacked tensors. When resizing the semantic map, use **nearest-neighbour** interpolation so class IDs are never blended into meaningless values. In real projects, annotations usually come in the COCO panoptic format: one PNG per image where each segment has a unique color-encoded ID, plus a JSON file listing each segment's category, and whether it is a thing or stuff.

## Merging the two outputs

The two heads can disagree: an instance mask and the semantic map may claim the same pixel, and instance masks may overlap each other. Panoptic FPN resolves this with a simple, ordered procedure:

1. Sort detected instances by confidence and drop those below a score threshold.
2. Paste instance masks one by one. Skip an instance if most of its mask (e.g. more than half) is already taken by higher-scoring instances; otherwise keep only its free pixels.
3. Fill the remaining pixels with the semantic head's stuff prediction. Remove stuff regions that are too small (the paper uses an area threshold) and leave them unlabeled.

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

In practice, use a library implementation (Detectron2, MMDetection) for this step. It handles thing and stuff ID spaces, ignore labels and edge cases properly.

## Measuring it: Panoptic Quality

Kirillov et al. also proposed one metric for the whole task. A predicted segment and a ground-truth segment of the same class **match** if their IoU is greater than 0.5. Because segments do not overlap, this threshold guarantees each match is unique. Matched pairs are true positives (TP); unmatched predictions are false positives (FP); unmatched ground-truth segments are false negatives (FN). Then, for each class,

$$
\text{PQ} = \frac{\sum_{(p,g) \in TP} \text{IoU}(p, g)}{|TP| + \tfrac{1}{2}|FP| + \tfrac{1}{2}|FN|}
$$

This splits neatly into two factors:

$$
\text{PQ} = \underbrace{\frac{\sum_{TP} \text{IoU}(p,g)}{|TP|}}_{\text{SQ: segmentation quality}} \times \underbrace{\frac{|TP|}{|TP| + \tfrac{1}{2}|FP| + \tfrac{1}{2}|FN|}}_{\text{RQ: recognition quality}}
$$

SQ is the average IoU of the matched segments: how good the shapes are. RQ is an F1 score: how well the model finds the right segments. The final PQ is averaged over classes, and it is often reported separately for things (PQ^Th) and stuff (PQ^St). When object localization matters more than overall pixel accuracy, PQ (or PQ^Th) is a better model-selection metric than mean IoU or pixel accuracy, because it penalizes missed and spurious objects directly.

## Key takeaways

- Panoptic segmentation gives every pixel a class, and gives pixels of "things" an instance ID, with no overlaps.
- Panoptic FPN = Mask R-CNN for things + a light semantic head for stuff, sharing one FPN backbone.
- A parameter-free merge step resolves conflicts: confident instances first, then stuff fills the gaps.
- Use nearest-neighbour resizing for label maps and return lists of targets from the data loader.
- Panoptic Quality = Segmentation Quality × Recognition Quality, with matches defined by IoU > 0.5.

## Resources

- Kirillov, He, Girshick, Rother, Dollár, "Panoptic Segmentation", CVPR 2019. [arXiv:1801.00868](https://arxiv.org/abs/1801.00868)
- Kirillov, Girshick, He, Dollár, "Panoptic Feature Pyramid Networks", CVPR 2019. [arXiv:1901.02446](https://arxiv.org/abs/1901.02446)
- He, Gkioxari, Dollár, Girshick, "Mask R-CNN", ICCV 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- COCO panoptic annotation format: [cocodataset.org/#format-data](https://cocodataset.org/#format-data)
- torchvision detection and segmentation models: [PyTorch docs](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
- Detectron2 (includes Panoptic FPN and panoptic evaluation): [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
