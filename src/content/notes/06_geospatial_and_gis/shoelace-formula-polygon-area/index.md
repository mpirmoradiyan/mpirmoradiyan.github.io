---
title: "The shoelace formula: the area of a polygon from its vertices"
slug: shoelace-formula-polygon-area
date: 2026-05-21
topic: Geospatial & GIS
tags: [geometry, polygons, area, orientation, numpy, numerical-precision]
summary: "How the shoelace formula turns a list of vertices into an area, why it works (trapezoids and cross products), what its sign tells you, and the two traps: self-intersecting rings and coordinates far from the origin."
sources:
  - title: "Shoelace formula"
    author: "Wikipedia"
    url: "https://en.wikipedia.org/wiki/Shoelace_formula"
    type: docs
  - title: "Computational Geometry in C (2nd ed.), Chapter 1: Polygon triangulation and area"
    author: "Joseph O'Rourke"
    type: book
  - title: "Shapely documentation: Polygon.area, orient"
    url: "https://shapely.readthedocs.io/"
    type: docs


status: draft
---

Every time a GIS tool reports the area of a field, or OpenCV's `contourArea` measures a blob, the same small formula runs underneath. Given the corners of a polygon in order, it returns the exact area in one pass over the vertices, with no triangulation and no grid. It also returns a sign, which tells you whether the vertices go clockwise or counter-clockwise. That sign is useful on its own.

## The formula

Take a simple polygon (its boundary does not cross itself) with vertices $(x_1, y_1), \dots, (x_n, y_n)$ listed in order around the boundary. Treat the list as circular, so vertex $n+1$ is vertex $1$. Then

$$
A_{\text{signed}} = \frac{1}{2} \sum_{i=1}^{n} \left( x_i\, y_{i+1} - x_{i+1}\, y_i \right), \qquad A = |A_{\text{signed}}|.
$$

The name comes from the way you compute it by hand. Write the coordinates in two columns, repeat the first vertex at the bottom, then multiply along the down-right diagonals ($x_i y_{i+1}$) and the down-left diagonals ($x_{i+1} y_i$). The criss-crossing lines look like the laces of a shoe. Subtract the second sum from the first and halve it.

## Why it works: two pictures

![Left: each edge sweeps a trapezoid down to the x-axis; trapezoids under the top edges are added and those under the bottom edges are subtracted. Right: each edge forms a triangle with the origin; triangles swept counter-clockwise count positive and clockwise ones count negative.](fig-shoelace-decompositions.png)

**Trapezoids.** Drop a vertical line from both ends of each edge down to the x-axis. Each edge then bounds a trapezoid with signed area $\tfrac12 (x_i - x_{i+1})(y_i + y_{i+1})$. For a counter-clockwise polygon the top edges run right to left, so their trapezoids are positive. The bottom edges run left to right, so theirs are negative. Adding them leaves exactly the region between the top and bottom boundaries, which is the polygon. Expanding the product and summing over all edges gives the formula above, because the $x_i y_i$ terms cancel.

**Triangles from the origin.** The term $x_i y_{i+1} - x_{i+1} y_i$ is the 2D cross product of the vectors from the origin to vertex $i$ and to vertex $i+1$. Half of it is the signed area of the triangle (origin, $v_i$, $v_{i+1}$): positive when the triangle is swept counter-clockwise, negative when clockwise. The triangles that reach outside the polygon are counted once with each sign and cancel. Only the polygon's own area survives. The origin can be any point, even one outside the polygon, and the result is the same.

Both pictures are discrete versions of Green's theorem, which says that the area of a region equals the line integral $\tfrac12 \oint (x\,dy - y\,dx)$ around its boundary. On a straight edge that integral is exactly $\tfrac12 (x_i y_{i+1} - x_{i+1} y_i)$.

## Code

The vectorised version shifts the coordinate arrays by one place with `np.roll`, so vertex $i+1$ lines up with vertex $i$ and the wrap-around comes for free:

```python
import numpy as np

def shoelace(xy, signed=False):
    """Area of a simple polygon from an (N, 2) array of vertices in order."""
    x, y = np.asarray(xy, dtype=float).T
    # x_i * y_{i+1} - x_{i+1} * y_i, with the last vertex wrapping to the first
    a = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return a if signed else abs(a)

square = [(0, 0), (1, 0), (1, 1), (0, 1)]
print(shoelace(square, signed=True))         # 1.0  (counter-clockwise)
print(shoelace(square[::-1], signed=True))   # -1.0 (clockwise)

l_shape = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
print(shoelace(l_shape))                     # 5.0  (works for non-convex shapes)
```

The formula needs no convexity: the L-shape above is concave and the answer is exact. You also do not need to repeat the first vertex at the end of the list. If you do repeat it, the extra term is $x_1 y_1 - x_1 y_1 = 0$, so the result does not change.

On a regular 500,000-gon inscribed in a circle of radius 10, a plain Python loop and the NumPy version both give 314.15926535, close to $\pi r^2$ (the polygon is slightly smaller than the circle). On my laptop the NumPy version was about 35 times faster: 3.5 ms against 124 ms.

## The sign is a free orientation test

Counter-clockwise vertex order gives a positive signed area, and clockwise order gives a negative one (in a y-up coordinate system). This is how libraries check and fix winding order:

- The GeoJSON standard (RFC 7946) asks for counter-clockwise outer rings and clockwise holes. Shapely's `orient()` and many GIS tools use the signed area to enforce this.
- A polygon with holes has area $|A_{\text{outer}}| - \sum |A_{\text{hole}}|$. If the rings follow the convention, you can simply add the signed areas.
- In image coordinates the y-axis points down, so the visual sense flips: a ring that looks clockwise on screen has a positive signed area. This is a common source of confusion when contours come from an image and are later drawn on a map.

## Two traps

**Self-intersecting rings.** The formula assumes a simple polygon. For a "bow tie" such as $(0,0), (2,2), (2,0), (0,2)$, the two lobes have opposite orientation and cancel, so the formula returns 0. In general it returns the sum of each region's area times its winding number, which is rarely what you want. Check validity first and repair the shape if needed (see [Polygons from detections: fixing invalid shapes](/notes/polygon-validity-and-shape-metrics/)).

**Coordinates far from the origin.** Each product $x_i y_{i+1}$ can be huge while the difference between products is small. With 64-bit floats this causes catastrophic cancellation:

```python
far = np.array(l_shape) + 1e8          # same L-shape, shifted far away
print(shoelace(far))                    # 8.0, wrong
print(shoelace(far - far.mean(axis=0))) # 5.0, right after centring
```

Projected map coordinates (UTM eastings and northings in the hundreds of thousands to millions of metres) are large enough for this to cost precision on small polygons. The fix is cheap: subtract one vertex or the mean from all coordinates before applying the formula. Area does not change under translation, and the products become small.

One more caveat for GIS: the formula works in the plane. Applied to raw longitude and latitude it gives "square degrees", which mean nothing physically. Project to an equal-area or a suitable local projected coordinate system first, or use a geodesic area routine for large regions.

## Related quantities from the same sums

The same cross-product terms give the polygon's centroid:

$$
c_x = \frac{1}{6A_{\text{signed}}} \sum_{i} (x_i + x_{i+1})\, c_i, \qquad
c_y = \frac{1}{6A_{\text{signed}}} \sum_{i} (y_i + y_{i+1})\, c_i, \qquad
c_i = x_i y_{i+1} - x_{i+1} y_i .
$$

So one loop over the vertices gives area, orientation and centroid together, which is why the formula sits at the heart of most polygon libraries.

## Key takeaways

- Area of a simple polygon $= \tfrac12 \left|\sum_i (x_i y_{i+1} - x_{i+1} y_i)\right|$, one pass, convex or not.
- It is a sum of signed trapezoids or signed triangles (Green's theorem); the parts outside the polygon cancel.
- The sign gives the winding order: positive for counter-clockwise in a y-up system.
- Self-intersecting rings give meaningless results; validate first.
- Centre the coordinates before computing, especially with large projected map coordinates, to avoid cancellation errors.

## Resources

- Wikipedia, "Shoelace formula": derivations via trapezoids, triangles and Green's theorem. <https://en.wikipedia.org/wiki/Shoelace_formula>
- Joseph O'Rourke, *Computational Geometry in C*, 2nd ed., Chapter 1 (area of a polygon via triangles from a fixed point).
- H. Butler et al., RFC 7946, *The GeoJSON Format*, Section 3.1.6 (ring orientation). <https://www.rfc-editor.org/rfc/rfc7946>
- Shapely documentation: `Polygon.area`, `shapely.geometry.polygon.orient`. <https://shapely.readthedocs.io/>
- OpenCV documentation: `contourArea` (with the `oriented` flag). <https://docs.opencv.org/>
