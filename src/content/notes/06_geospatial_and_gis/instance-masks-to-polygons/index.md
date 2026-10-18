---
title: "From instance masks to polygons: contours, morphology and the usual pitfalls"
slug: instance-masks-to-polygons
date: 2025-09-10
topic: Geospatial & GIS
tags: [opencv, scikit-image, shapely, morphology, instance-segmentation, polygons, post-processing]
summary: "The standard tools for turning a raster instance mask into vector polygons (contour tracing, morphological operators, handling fragments and overlapping masks, simplification, georeferencing), with the trade-offs and the mistakes that are easy to make with each."
sources:
  - title: "OpenCV: Contours (findContours, hierarchy) and Morphological Transformations"
    author: "OpenCV"
    url: "https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html"
    type: docs
  - title: "scikit-image: measure.label, measure.regionprops, segmentation.expand_labels"
    author: "scikit-image"
    url: "https://scikit-image.org/docs/stable/api/skimage.measure.html"
    type: docs
  - title: "Shapely: simplify, buffer, affine_transform"
    author: "Shapely"
    url: "https://shapely.readthedocs.io/en/stable/"
    type: docs
  - title: "Topological structural analysis of digitized binary images by border following"
    author: "Suzuki & Abe (1985)"
    type: paper
  - title: "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature"
    author: "Douglas & Peucker (1973)"
    type: paper


status: draft
---

A segmentation model gives you pixels; a GIS wants polygons. Getting from one to the other uses a small set of standard tools: contour tracing, morphological operators, connected-component labelling and line simplification. Each is simple on its own, and each has a way of going quietly wrong. This note goes through the tools one by one on a synthetic mask, with the options and trade-offs for each. Repairing invalid polygons and measuring their shape is covered in [polygon validity and shape metrics](/notes/polygon-validity-and-shape-metrics/).

![Left: a synthetic instance mask with four objects, ragged coloured edge pixels, pinholes, two stray squares and a purple piece split off the bottom of the green object. Middle: the same objects after cleaning, each one solid with a smooth edge and the purple piece absorbed into green. Right: grey contours and coloured simplified polygons with few vertices.](fig-mask-cleanup.png)

## Contours: from a binary mask to rings

An instance mask stores one integer per pixel: 0 for background and $k$ for object $k$. For each object you take the binary mask `lab == k` and trace its border. OpenCV's `findContours` implements the border-following algorithm of Suzuki and Abe; `skimage.measure.find_contours` (marching squares, sub-pixel) and `rasterio.features.shapes` (exact pixel edges) are common alternatives. In OpenCV two flags matter:

- **Retrieval mode.** `RETR_EXTERNAL` returns only outer borders, so holes disappear. `RETR_CCOMP` returns outer borders and holes in a two-level hierarchy, so polygons with holes can be rebuilt.
- **Approximation.** `CHAIN_APPROX_NONE` keeps every border pixel; `CHAIN_APPROX_SIMPLE` keeps only the ends of straight runs. Same shape, fewer vertices.

```python
import numpy as np, cv2
from shapely.geometry import Polygon

def mask_to_polygons(mask):
    """Binary mask -> list of shapely Polygons with holes, in pixel coordinates (x=col, y=row)."""
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    if hier is None:
        return polys
    for i, (nxt, prv, child, parent) in enumerate(hier[0]):
        if parent != -1 or len(contours[i]) < 3:   # skip holes and degenerate pieces
            continue
        holes, j = [], child
        while j != -1:                              # walk the list of this outer ring's holes
            if len(contours[j]) >= 3:
                holes.append(contours[j][:, 0, :])
            j = hier[0][j][0]
        polys.append(Polygon(contours[i][:, 0, :], holes))
    return polys
```

On the raw mask in the figure this returns up to four parts per object and dozens of one-pixel holes, and most results are invalid because tiny holes touch the outer ring or each other. Noise in the raster becomes broken geometry in the vector, which is why masks are usually cleaned before they are traced.

Two details are worth knowing. One object can produce **several outer contours**, so you have to decide whether an object is a `MultiPolygon` or only its largest part. And OpenCV contours run through **pixel centres**, so the polygon is about half a pixel smaller than the pixel footprint. For a 2,077-pixel object the polygon area is 2,001 px²; `polygon.buffer(0.5, join_style="mitre")` brings it back to about 2,089. For small objects that half pixel is a large share of the area.

## Morphology: four operators and two traps

Mathematical morphology probes a binary image with a small shape, the structuring element (kernel):

- **Erosion** keeps a pixel only if the whole kernel around it fits inside the object. It shrinks the object, deletes thin spurs and splits thin bridges.
- **Dilation** adds every pixel the kernel touches. It grows the object and fills small gaps.
- **Opening** is erosion followed by dilation **with the same kernel, the same number of times**. It removes features smaller than the kernel and leaves larger ones about the same size.
- **Closing** is dilation followed by erosion. It fills holes and notches smaller than the kernel.

**Trap 1: unequal steps.** Eroding more times than you dilate is not an opening. It is an opening plus extra erosion, so every object comes out smaller. In the figure below, two erosions and one dilation keep only about half of the object.

**Trap 2: holes inside the object.** Erosion also grows every hole. With many pinholes the growing holes meet and eat the interior, and the following dilation cannot bring it back. Filling holes (`scipy.ndimage.binary_fill_holes`) or a closing before an opening avoids this:

```python
from scipy.ndimage import binary_fill_holes

K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))       # round kernel
m = (lab == 3).astype(np.uint8)
opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, K, iterations=2)  # erode 2x, then dilate 2x
filled = binary_fill_holes(m).astype(np.uint8)
opened_filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, K, iterations=2)
```

![Four views of the same noisy object: the original with pinholes (1998 px); two erosions and one dilation, which leaves a ragged remnant (1066 px); an opening with the pinholes still in, which loses a fifth of the object (1589 px); and an opening after filling holes, which gives a clean solid ellipse (1948 px).](fig-morphology-order.png)

A round kernel (`MORPH_ELLIPSE`) treats all directions the same; a square kernel slowly turns round objects into octagons. A median filter (`cv2.medianBlur`, odd kernel size) is another common edge smoother for binary masks. The right kernel size depends on the pixel size and the smallest detail you need to keep, so there is no universal value.

**Several labels in one image.** Dilating each label separately and pasting the results back into one image makes the result depend on loop order: where two objects grow into each other, the last one written wins. `skimage.segmentation.expand_labels(labels, distance)` grows every label only into background and stops where labels meet.

## Fragments: several pieces of one object

After cleaning, an object can still consist of several connected pieces, or a model can split one object into a big instance and a small one (the purple tip in the first figure). `skimage.measure.label` finds the connected pieces and `regionprops` gives their areas. The common options are:

- **Drop** pieces below a minimum area. Simple, but it deletes real small objects and loses area.
- **Keep only the largest piece** per object. Gives one polygon per object; stray parts vanish.
- **Keep all pieces** as a `MultiPolygon`. Loses nothing, but many tools expect simple polygons.
- **Absorb** a small piece into the neighbouring object it shares the longest border with. Keeps the area, but can glue a real small object onto a big one.

The last option needs a little code. The one-pixel ring around a piece is the set of pixels it touches, and `bincount` counts how many of them belong to each neighbour:

```python
from skimage.measure import label, regionprops

def absorb_small_pieces(lab, min_area):
    out, changed = lab.copy(), True
    while changed:                                    # each merge lowers the number of pieces,
        changed = False                               # so the loop always ends
        for inst in np.unique(out)[1:]:
            comps = label(out == inst)
            for r in regionprops(comps):
                if r.area >= min_area:
                    continue
                piece = comps == r.label
                ring = (cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~piece
                neigh = out[ring]
                neigh = neigh[(neigh != 0) & (neigh != inst)]
                out[piece] = np.bincount(neigh).argmax() if neigh.size else 0
                changed = True
    return out
```

The loop repeats because absorbing one piece can change what its neighbours look like. A threshold like `min_area` is a property of the objects and the pixel size, so it has to be chosen per dataset.

## Overlapping masks: who owns a pixel?

Models that output one binary mask per instance (Mask R-CNN and its relatives) can give the same pixel to two objects. Before tracing, the usual choices are:

- **Highest score wins.** Paint masks from the lowest to the highest confidence, so the most confident one is written last. This is the same as a per-pixel argmax over scores.
- **Remove duplicates first.** If two masks overlap heavily (high IoU), they are probably the same object; mask non-maximum suppression keeps one of them.
- **Resolve later in vector space**, where the overlap can be split geometrically. See [overlap handling with a spatial index](/notes/polygon-overlaps-with-strtree/).

```python
owner = np.zeros(masks[0].shape, np.int32)
for i in np.argsort(scores):          # lowest score first ...
    owner[masks[i]] = i + 1           # ... the most confident mask is painted last and wins
```

## Simplification and map coordinates

Contours follow the pixel staircase, so they carry many vertices with no real information. Douglas–Peucker simplification removes every vertex that lies within a tolerance of the simplified line; `cv2.approxPolyDP` and Shapely's `simplify` both implement it.

```python
from shapely import affinity
from shapely.ops import unary_union

s = p.simplify(1.0, preserve_topology=True)   # p from mask_to_polygons: ~90 -> under 20 vertices

# pixel (col, row) -> map (x, y) for a north-up raster whose top-left corner is (x0, y0)
x0, y0, px = 500000.0, 6400000.0, 0.05
s_map = affinity.affine_transform(s, [px, 0, 0, -px, x0 + px / 2, y0 - px / 2])

# one outline around nearby parts: a "closing" in vector space
merged = unary_union([g.buffer(1.5, join_style="mitre") for g in parts]).buffer(-1.5, join_style="mitre")
```

A tolerance of about one pixel removes the staircase without cutting real corners; here the area changes by less than 0.5 %. `preserve_topology=True` stops the result from crossing itself. Simplifying each polygon on its own does **not** keep shared borders shared, so neighbours can end up with thin gaps or overlaps; topology-aware tools (for example TopoJSON-style simplification or GRASS `v.generalize`) avoid that.

The affine step flips the y axis (rows grow downward, northings grow upward) and adds half a pixel, because the contour vertices sit at pixel centres. The last line answers a common question: `unary_union` of separate polygons is a `MultiPolygon`, not one outline. Taking its first part (`.geoms[0]` in Shapely 2; plain indexing no longer works) just throws the others away. Buffering out, uniting and buffering back by the same distance bridges gaps narrower than twice the distance and returns one polygon. It is the vector counterpart of a morphological closing.

## Key takeaways

- `RETR_CCOMP` keeps holes; one object can yield several contours, so decide how to handle them.
- An opening erodes and dilates the same number of times; holes grow during erosion, so fill them first.
- Dilating labels one by one into a shared image is order-dependent; `expand_labels` is not.
- Fragments can be dropped, kept, reduced to the largest piece or absorbed into a neighbour; each choice trades area against false merges.
- Simplify with a tolerance around one pixel, and remember the half-pixel offset when georeferencing.

## Resources

- OpenCV documentation: [Contours in OpenCV](https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html) (including contour hierarchy) and [Morphological Transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).
- scikit-image documentation: [`measure.label` and `regionprops`](https://scikit-image.org/docs/stable/api/skimage.measure.html), [`segmentation.expand_labels`](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_expand_labels.html).
- SciPy documentation: [`ndimage.binary_fill_holes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_fill_holes.html).
- Shapely documentation: [`simplify`](https://shapely.readthedocs.io/en/stable/reference/shapely.simplify.html), [`buffer`](https://shapely.readthedocs.io/en/stable/reference/shapely.buffer.html), [`affine_transform`](https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.affine_transform).
- Suzuki, S. & Abe, K. (1985). Topological structural analysis of digitized binary images by border following. *Computer Vision, Graphics, and Image Processing*, 30(1), 32–46. The algorithm behind `findContours`.
- Douglas, D. H. & Peucker, T. K. (1973). Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. *Cartographica*, 10(2), 112–122.
- Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press. The classic reference for erosion, dilation, opening and closing.
- He, K., Gkioxari, G., Dollár, P. & Girshick, R. (2017). Mask R-CNN. *ICCV*. Per-instance masks with confidence scores.
