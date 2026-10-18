---
title: "Overlapping polygons at scale: find pairs with an STRtree, then split the overlaps"
slug: polygon-overlaps-with-strtree
date: 2025-02-18
topic: Geospatial & GIS
tags: [shapely, geopandas, spatial-index, strtree, r-tree, voronoi, polygons, post-processing]
summary: "How a spatial index turns the search for overlapping polygons from O(n²) into roughly O(n log n), the Shapely 2 API that does it in one call, two easy-to-make bugs, and two ways to split an overlap: by priority or fairly with Voronoi cells."
sources:
  - title: "Shapely documentation: STRtree, voronoi_polygons, union_all"
    author: "Shapely"
    url: "https://shapely.readthedocs.io/en/stable/strtree.html"
    type: docs
  - title: "GeoPandas documentation: SpatialIndex.query and sjoin"
    author: "GeoPandas"
    url: "https://geopandas.org/en/stable/docs/reference/api/geopandas.sindex.SpatialIndex.query.html"
    type: docs
  - title: "STR: A simple and efficient algorithm for R-tree packing"
    author: "Leutenegger, Lopez & Edgington (1997)"
    type: paper
  - title: "R-trees: A dynamic index structure for spatial searching"
    author: "Guttman (1984)"
    type: paper


status: draft
---

Detection and segmentation models predict objects one at a time, so neighbouring polygons often overlap: two crowns that claim the same leaves, two buildings that share a wall. Before the polygons can be counted, measured or exported, every overlap needs an owner. The first step is to find which polygons overlap at all, and with tens of thousands of polygons, how you search decides whether that takes milliseconds or minutes. This note covers the search with a spatial index and two ways to split what you find. (Repairing invalid polygons first is covered in [polygon validity and shape metrics](/notes/polygon-validity-and-shape-metrics/).)

## Why all-pairs is too slow

The direct approach tests every polygon against every other one: $n(n-1)/2$ tests. Doubling the number of polygons makes the work four times larger. That is $O(n^2)$, and each test is itself not free, since an exact `intersects` on two polygons with $v$ vertices costs more than a box comparison.

Yet almost all of those tests are pointless. A polygon in one corner of a map cannot touch one in the other corner. What we need is a way to ask "which polygons are near this one?" without looking at all of them.

## The idea behind an STRtree

An **R-tree** groups nearby objects into boxes, then groups those boxes into bigger boxes, and so on up to one root box. Each node stores only the **bounding box** of what lies below it. To find everything that might touch a query shape, you start at the root and descend only into boxes that overlap the query's box. Whole regions of the map are skipped in one comparison.

An **STRtree** (Sort-Tile-Recursive) is an R-tree built in one go from a fixed set of geometries. It sorts the boxes by $x$, cuts them into vertical slices, sorts each slice by $y$ and packs neighbours into full nodes. Building it costs $O(n \log n)$ (the sorting). A query costs about $O(\log n + k)$, where $k$ is the number of boxes it returns. Querying all $n$ polygons therefore costs about $O(n \log n + K)$, with $K$ the total number of candidate pairs, which is close to $O(n)$ times the typical number of neighbours when density does not grow with the map.

The search runs in two phases:

1. **Filter**: the tree returns candidates whose *bounding boxes* intersect the query box. Cheap, but it includes false positives (two round crowns can have touching boxes and still not touch).
2. **Refine**: an exact geometric predicate (`intersects`, `overlaps`, ...) is run on the candidates only.

In Shapely 2 both phases happen in one vectorised call:

```python
import numpy as np, shapely
from shapely.strtree import STRtree

def random_crowns(n, seed=0):
    rng = np.random.default_rng(seed)
    side = np.sqrt(n) * 10                                 # constant density as n grows
    pts = shapely.points(rng.uniform(0, side, (n, 2)))
    return shapely.buffer(pts, rng.uniform(2, 5, n))       # discs with radius 2-5

def overlapping_pairs(geoms):
    tree = STRtree(geoms)
    i, j = tree.query(geoms, predicate="intersects")       # filter + refine, all at once
    keep = i < j                                           # drop self-pairs and (j, i) duplicates
    return i[keep], j[keep]

geoms = random_crowns(4000)
i, j = overlapping_pairs(geoms)
```

Passing an array to `query` returns a $2 \times m$ array: row 0 indexes the input, row 1 the tree. Every polygon intersects itself, so the pairs $(k, k)$ are always present, and each real pair appears twice; `i < j` removes both. With GeoPandas the same call is `gdf.sindex.query(gdf.geometry, predicate="intersects")`, or a self spatial join with `gpd.sjoin(gdf, gdf, predicate="intersects")`.

![Log-log plot of the time to find all intersecting pairs against the number of polygons. The all-pairs line rises four times for each doubling, reaching about 1.8 s at 4,000 polygons. The STRtree line rises about two times per doubling and stays below 0.05 s at 8,000 polygons.](fig-strtree-timing.png)

On random discs the all-pairs loop takes 1.8 s for 4,000 polygons; the tree takes 0.02 s and gives the same pairs. The gap keeps widening: it is a different growth rate, not a constant speed-up.

## Two bugs that look correct

**Comparing indices with geometries.** In Shapely 1.x, `STRtree.query(geom)` returned geometries. In Shapely 2 it returns **integer indices**. Code written for the old API often removes "self" like this:

```python
tree = STRtree(geoms)
matches = [m for m in tree.query(g) if m != g]   # m is an int, g a Polygon: never equal
```

An integer never equals a polygon, so self is never removed and **every** polygon is reported as overlapping something. The fix is to compare indices (`m != k`) or use the array form above.

**Forgetting the refine step.** `tree.query(g)` without a `predicate`, and GeoPandas' `sindex.intersection(bounds)`, return only the box filter. Treating that as "intersects" flags round objects whose boxes touch at the corners. A related trap: these functions return **positions**, while `iterrows()` gives **index labels**. After filtering or sorting a GeoDataFrame the two differ, so compare positions with positions (use `iloc`, or `reset_index(drop=True)` first).

## From pairs to clusters

Pairs alone are awkward to work with. Turn them into a graph (polygons are nodes, overlaps are edges) and take connected components. Each component is an independent cluster that can be resolved on its own, and polygons in clusters of size one need no work at all:

```python
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

n = len(geoms)
adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
n_clusters, cluster_id = connected_components(adj, directed=False)
standalone = np.bincount(cluster_id)[cluster_id] == 1      # polygons that touch nobody
```

The pairs also give overlap statistics cheaply. `shapely.area(shapely.intersection(geoms[i], geoms[j]))` divided by the union area is the IoU of every pair in one vectorised line. A high IoU usually means a duplicate detection (keep one), while a small overlap means two real objects that both reached a little too far (split it).

## Splitting an overlap

![Left: four circles, three of them overlapping, with confidence scores 0.80, 0.95, 0.60 and 0.90. Middle: after a priority split the 0.95 circle is whole and the others are cut where they met it. Right: after a Voronoi split the shared areas are divided along straight lines between the circle centres; dashed lines show the Voronoi cells.](fig-overlap-splitting.png)

There is no single right answer; the common approaches are:

| Approach | What happens to the overlap | Good when | Cost |
|---|---|---|---|
| Suppress duplicates (NMS on IoU) | the weaker polygon is removed entirely | both polygons describe the same object | loses a real object if the threshold is too low |
| Priority | goes whole to the more trusted polygon | there is a clear ranking (score, class, source) | the weaker polygon gets a bite taken out |
| Geometric split (bisector / Voronoi) | divided between the polygons | neither polygon deserves it more | ignores which prediction was better |
| Merge | polygons are united into one | the parts are pieces of one object | wrong when they are two objects |

The second and third need a little code.

**By priority.** Give the whole overlap to the polygon you trust more, usually the one with the higher model score. Each polygon subtracts the union of its **stronger** neighbours:

```python
def neighbours(k, i, j):
    return np.concatenate([j[i == k], i[j == k]])

def split_by_priority(polys, scores, i, j):
    rank = np.argsort(np.argsort(-scores, kind="stable"))   # 0 = most confident; ties by order
    out = polys.copy()
    for k in range(len(polys)):
        nb = neighbours(k, i, j)
        stronger = nb[rank[nb] < rank[k]]
        if stronger.size:
            out[k] = polys[k].difference(shapely.union_all(polys[stronger]))
    return out
```

The direction of the comparison is easy to flip by accident. Subtracting neighbours with score **lower or equal** to your own does the opposite of what you want: the most confident polygon gets cut by every weaker one, and two polygons with equal scores cut each other so the overlap belongs to nobody. A strict ranking with a tie-breaker avoids both problems.

**Fairly, with Voronoi cells.** When neither polygon deserves the overlap more, divide it. For two polygons the natural cut is the perpendicular bisector between their centroids: every point of the overlap goes to the nearer centre. For many polygons that rule is exactly the **Voronoi diagram** of the centroids. Each polygon keeps its uncontested part, plus the part of its overlap that falls in its own cell:

```python
def split_by_voronoi(polys, i, j):
    seeds = shapely.multipoints(shapely.centroid(polys))
    box = shapely.union_all(polys).envelope
    cells = shapely.voronoi_polygons(seeds, extend_to=box, ordered=True)  # cell k <-> seed k
    out = polys.copy()
    for k in range(len(polys)):
        nb = neighbours(k, i, j)
        if nb.size:
            own = polys[k].difference(shapely.union_all(polys[nb]))     # uncontested
            out[k] = own.union(polys[k].intersection(cells.geoms[k]))   # + its share
    return out
```

`ordered=True` (GEOS 3.12 or newer) keeps cell $k$ aligned with seed $k$. Both methods return polygons that no longer overlap. Two limits of the Voronoi split: the bisector ignores size, so a big object loses as much to a small neighbour as the other way round (a weighted, or power, diagram fixes that); and in a cluster of three or more, a sliver of overlap can fall in the cell of a polygon that does not contain it and is then dropped. That is a fraction of a percent in the example above, but check total area if it matters.

## Key takeaways

- All-pairs overlap search is $O(n^2)$; an STRtree makes it about $O(n \log n)$ by filtering on bounding boxes first.
- In Shapely 2, `STRtree(geoms).query(geoms, predicate="intersects")` returns all index pairs; drop `i >= j`.
- `query` returns integer positions, not geometries, and without a predicate it is only a box test.
- Group pairs into connected components and resolve each cluster on its own.
- Overlaps can be suppressed, given to the stronger polygon, split by Voronoi cells or merged; the right choice depends on whether the polygons are the same object.

## Resources

- Shapely documentation: [STRtree](https://shapely.readthedocs.io/en/stable/strtree.html), [`voronoi_polygons`](https://shapely.readthedocs.io/en/stable/reference/shapely.voronoi_polygons.html), [`union_all`](https://shapely.readthedocs.io/en/stable/reference/shapely.union_all.html), and the [Shapely 2.0 migration guide](https://shapely.readthedocs.io/en/stable/migration.html) (the change from geometries to indices in `STRtree.query`).
- GeoPandas documentation: [`SpatialIndex.query`](https://geopandas.org/en/stable/docs/reference/api/geopandas.sindex.SpatialIndex.query.html) and [`sjoin`](https://geopandas.org/en/stable/docs/reference/api/geopandas.sjoin.html).
- SciPy documentation: [`csgraph.connected_components`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.connected_components.html).
- Leutenegger, S. T., Lopez, M. A. & Edgington, J. (1997). STR: A simple and efficient algorithm for R-tree packing. *Proceedings of the 13th International Conference on Data Engineering (ICDE)*, 497–506.
- Guttman, A. (1984). R-trees: A dynamic index structure for spatial searching. *Proceedings of ACM SIGMOD*, 47–57.
- Aurenhammer, F. (1991). Voronoi diagrams: a survey of a fundamental geometric data structure. *ACM Computing Surveys*, 23(3), 345–405.
