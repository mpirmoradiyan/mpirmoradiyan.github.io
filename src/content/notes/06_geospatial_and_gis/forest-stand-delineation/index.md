---
title: "Forest stand delineation: turning canopy rasters into management polygons"
slug: forest-stand-delineation
date: 2023-05-15
topic: Geospatial & GIS
tags: [forestry, stand-delineation, segmentation, lidar, chm, obia, superpixels, gis]
summary: "What a forest stand is, why drawing stand boundaries automatically is a segmentation problem with fuzzy ground truth, and the main published approaches: feature rasters, region growing and graph-based segmentation, merging to a minimum mapping unit, and how results are evaluated."
sources:
  - title: "Airborne laser data for stand delineation and information extraction"
    author: "Koch, Straub, Dees, Wang & Weinacker (2009)"
    type: paper
  - title: "Automatic segmentation of forest stands using a canopy height model and aerial photography"
    author: "Mustonen, Packalén & Kangas (2008)"
    type: paper
  - title: "Semantic segmentation of forest stands of pure species combining airborne lidar data and very high resolution multispectral imagery"
    author: "Dechesne, Mallet, Le Bris & Gouet-Brunet (2017)"
    type: paper
  - title: "Multiresolution segmentation: an optimization approach for high quality multi-scale image segmentation"
    author: "Baatz & Schäpe (2000)"
    type: paper
  - title: "Efficient graph-based image segmentation"
    author: "Felzenszwalb & Huttenlocher (2004)"
    type: paper
  - title: "SLIC superpixels compared to state-of-the-art superpixel methods"
    author: "Achanta, Shaji, Smith, Lucchi, Fua & Süsstrunk (2012)"
    type: paper
  - title: "Accuracy assessment measures for object-based image segmentation goodness"
    author: "Clinton, Holt, Scarborough, Yan & Gong (2010)"
    type: paper
  - title: "stand_mapping: tools for delineating forest stands"
    author: "David Diaz"
    url: "https://github.com/d-diaz/stand_mapping"
    type: repo


status: draft
---

Forests are managed in **stands**: patches of a few hectares where the trees are similar enough to be treated the same way, thinned in the same year or harvested together. Stand maps were traditionally drawn by hand on aerial photos and corrected in the field. That is slow, and two interpreters rarely draw the same lines. With wall-to-wall LiDAR and drone imagery, a computer can propose the boundaries. This note covers the published ideas behind automatic stand delineation and a small synthetic example that shows the typical pipeline and its weak spots. It builds on the rasters from [airborne LiDAR basics](/notes/airborne-lidar-dtm-dsm-chm/) and the stand parameters in [forest structure from above](/notes/forest-structure-from-above/).

## What makes a stand a stand

A stand is homogeneous in the things that drive management: **species mix, age or development class, height, density** and often site quality. It must also be **large and compact enough** to manage. Forest agencies set a minimum mapping unit, often somewhere between a few tenths of a hectare and a couple of hectares, and they avoid long thin slivers. Some boundaries are physical (roads, streams, clear-cut edges). Others are gradual, such as a slow change in density across a slope.

This gives the problem its character. It is image segmentation, but:

- the "objects" are defined by **statistics of many trees**, not by the value of one pixel;
- the result must respect **size and shape rules**, not just similarity;
- the reference maps are themselves subjective, so a perfect score is impossible.

## Step 1: describe each place with features

A single CHM pixel says almost nothing about the stand: it is either a crown or a gap. The useful signal is in the statistics over a neighbourhood of a few tree spacings. Common features, computed per cell over a moving window or per grid cell of 10–20 m, are:

- **Height**: mean, a high percentile such as the 90th or 95th (close to dominant height), maximum.
- **Density and cover**: share of returns or CHM cells above 2 m, number of detected tree tops per area.
- **Structure**: standard deviation of heights (roughness), crown size, vertical distribution of returns.
- **Spectral**: mean band values or indices from imagery, which carry most of the species information (conifer versus broadleaf, for example).

Scale features to similar ranges before you compare them, or the feature with the largest units decides everything. Then smooth them with an **edge-preserving** filter, such as a median filter, so that the noise inside stands drops while the steps between stands stay sharp.

## Step 2: segment

Published methods fall into a few families.

- **Region growing and multiresolution segmentation.** Start from seed pixels or single pixels, and merge neighbouring regions while the merge keeps them homogeneous. In Baatz and Schäpe's multiresolution segmentation (the core of eCognition and of most object-based image analysis, OBIA), the cost of a merge mixes spectral heterogeneity with a shape term for compactness and smoothness. One "scale parameter" limits how much heterogeneity a segment may hold. Many stand studies use it on CHM plus imagery, for example Mustonen et al. (2008) and Koch et al. (2009).
- **Graph-based segmentation.** Treat pixels as nodes of a graph with edge weights equal to feature differences. Felzenszwalb and Huttenlocher (2004) merge two regions when the edge between them is weak compared with the variation inside each. A scale parameter favours larger regions. It is fast and adapts to both sharp and gradual boundaries.
- **Superpixels, then merging.** Over-segment into many small compact pieces (SLIC is the common choice), build a region adjacency graph, and merge similar neighbours. This separates "find good local boundaries" from "decide which pieces belong together".
- **Clustering in feature space.** k-means, mean shift or self-organising maps group cells with similar features regardless of location, and connected components then give the polygons. Spatial coherence must be added afterwards, for example by smoothing or merging, or the result is speckled.
- **Learned segmentation.** With enough labelled stands, a CNN or a classifier followed by a graph model (Dechesne et al., 2017, for species-pure stands) can learn which feature changes matter. Reference data are the bottleneck.
- **Optimisation.** Some forestry work treats delineation as an optimisation problem, with cellular automata or heuristics that trade within-stand homogeneity against stand size and shape and even management goals.

## Step 3: enforce the minimum mapping unit

Every method above leaves small pieces: gaps, single big trees, transition zones. The standard fix is to **absorb each segment below the minimum area into its most similar neighbour**, smallest first, until none is left:

```python
import numpy as np
from scipy import ndimage as ndi

def absorb_small(labels, feats, min_cells):
    """Merge every segment smaller than min_cells into its most similar neighbour."""
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        if counts.min() >= min_cells:
            return labels
        s = ids[np.argmin(counts)]                              # smallest first
        ring = ndi.binary_dilation(labels == s) & (labels != s) # its neighbours
        f_s = feats[labels == s].mean(0)
        best = min(np.unique(labels[ring]),
                   key=lambda n: np.linalg.norm(feats[labels == n].mean(0) - f_s))
        labels[labels == s] = best
```

After that, convert the label raster to polygons, smooth or simplify the staircase edges, and snap them to hard boundaries such as roads and property lines where those exist. The vector side of this is the same as in [instance masks to polygons](/notes/instance-masks-to-polygons/).

## A synthetic example

I built a 200 m × 200 m CHM at 1 m resolution with five stands that differ in tree height, crown size and density, including a round young stand. The features were mean height, height standard deviation and canopy cover over a 21 m window, median-filtered. Then came graph-based segmentation and a 0.15 ha minimum unit.

![Four panels: a synthetic canopy height model with five stands of different tree sizes and densities; the three smoothed features shown as an RGB image; the graph-based segmentation with 13 segments, including thin pieces along stand edges; and the result after merging pieces below 0.15 ha, with 6 stands whose boundaries mostly follow the true ones, drawn as dashed red lines.](fig-stand-segmentation.png)

The result shows the typical behaviour. Strong contrasts, like young versus mature forest, are found almost exactly. The moving window creates a **transition band** about half a window wide at every boundary. The segmenter tends to cut this band out as its own thin segment, and the minimum-area rule then has to decide where it goes. Two stands with similar features stay hard to separate however you tune the method, because the features simply do not differ enough. The window size is the main trade-off: larger windows give cleaner features inside stands but blurrier, shifted boundaries.

## Evaluating a stand map

Because reference boundaries are subjective, evaluation uses several views:

- **Overlap per reference stand**: for each reference polygon, find the segment that overlaps it most and report the shared area. Clinton et al. (2010) list the standard over- and under-segmentation measures built this way.
- **Boundary distance**: the share of predicted boundary within a tolerance (say 10 m) of a reference boundary.
- **Homogeneity**: the within-stand variance of the attributes that matter (height, volume, species share), compared with the manual map. A map can disagree with the interpreter and still be better for planning.
- **Usability**: number of stands, size distribution and shape compactness against the agency's rules.

## Key takeaways

- A stand is defined by the statistics of many trees plus size and shape rules, so delineation is segmentation on neighbourhood features, not on raw pixels.
- Height percentiles, cover and roughness from LiDAR, plus spectral features for species, are the standard inputs. Scale them, and smooth with an edge-preserving filter.
- Region growing (multiresolution segmentation), graph-based segmentation, superpixels plus merging, and clustering all work. The post-processing to a minimum mapping unit matters as much as the choice of method.
- Window size trades within-stand noise against boundary sharpness, and transition bands are the usual source of small spurious segments.
- Evaluate with overlap, boundary distance and within-stand homogeneity together, since the reference itself is subjective.

## Resources

- B. Koch, C. Straub, M. Dees, Y. Wang, H. Weinacker, "Airborne laser data for stand delineation and information extraction", *International Journal of Remote Sensing*, 30(4), 2009.
- J. Mustonen, P. Packalén, A. Kangas, "Automatic segmentation of forest stands using a canopy height model and aerial photography", *Scandinavian Journal of Forest Research*, 23(6), 2008.
- C. Dechesne, C. Mallet, A. Le Bris, V. Gouet-Brunet, "Semantic segmentation of forest stands of pure species combining airborne lidar data and very high resolution multispectral imagery", *ISPRS Journal of Photogrammetry and Remote Sensing*, 126, 2017.
- M. Baatz, A. Schäpe, "Multiresolution segmentation: an optimization approach for high quality multi-scale image segmentation", *Angewandte Geographische Informationsverarbeitung XII*, 2000.
- P. Felzenszwalb, D. Huttenlocher, "Efficient graph-based image segmentation", *International Journal of Computer Vision*, 59(2), 2004.
- R. Achanta et al., "SLIC superpixels compared to state-of-the-art superpixel methods", *IEEE TPAMI*, 34(11), 2012.
- N. Clinton et al., "Accuracy assessment measures for object-based image segmentation goodness", *Photogrammetric Engineering & Remote Sensing*, 76(3), 2010.
- D. Diaz, [stand_mapping](https://github.com/d-diaz/stand_mapping), an open-source collection of stand delineation tools and benchmarks.
