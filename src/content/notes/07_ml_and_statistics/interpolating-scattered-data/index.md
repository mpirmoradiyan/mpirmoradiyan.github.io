---
title: "Interpolating scattered data onto a grid: nearest, linear, cubic, RBF and IDW"
slug: interpolating-scattered-data
date: 2023-02-16
topic: ML & Statistics
tags: [interpolation, scattered-data, delaunay, rbf, idw, scipy, numerical-methods]
summary: "How the common ways of filling a regular grid from irregularly placed samples work, how they behave outside the data and with noise or anisotropy, and how to choose between them."
sources:
  - title: "SciPy interpolation tutorial: scattered data"
    author: "SciPy developers"
    url: "https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_unstructured.html"
    type: docs
  - title: "Transition guide: interp2d removal"
    author: "SciPy developers"
    url: "https://docs.scipy.org/doc/scipy/tutorial/interpolate/interp_transition_guide.html"
    type: docs
  - title: "A two-dimensional interpolation function for irregularly-spaced data"
    author: "Donald Shepard"
    url: "https://doi.org/10.1145/800186.810616"
    type: paper
  - title: "Scattered Data Approximation"
    author: "Holger Wendland"
    type: book
  - title: "Radial Basis Functions: Theory and Implementations"
    author: "Martin D. Buhmann"
    type: book


status: draft
---

Many measurements arrive as scattered points: weather stations, soil samples, GPS tracks, LiDAR ground returns, or a handful of hand-picked control values. Most tools want a regular grid. Interpolation fills the grid from the points, and every method makes a different assumption about what happens *between* samples. I compared the standard methods in early 2023 while experimenting with smooth 2-D weight surfaces built from a few control values. This note keeps the general lessons, on a synthetic example.

## The problem

We have $n$ samples $(\mathbf{x}_i, z_i)$, where $\mathbf{x}_i = (x_i, y_i)$ is a location and $z_i$ the measured value. We want a function $\hat z(\mathbf{x})$ that passes through (or near) the samples and gives a value at every grid node. There is no single right answer: the data alone do not say what happens between points. The method is a choice of prior: "the surface is piecewise flat", "piecewise planar", "as smooth as possible", or "a weighted average of neighbours".

For the comparison below I sampled a smooth function at 80 random points in $[0.05, 0.95]^2$ and interpolated onto a $200 \times 200$ grid covering $[0, 1]^2$. The samples do not reach the edges on purpose.

```python
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator

def f(x, y):                                    # smooth synthetic truth
    return np.sin(3*x) * np.cos(4*y) + 0.5*np.exp(-((x-0.7)**2 + (y-0.3)**2) / 0.02)

rng = np.random.default_rng(0)
pts = rng.uniform(0.05, 0.95, size=(80, 2))     # scattered sample locations
vals = f(pts[:, 0], pts[:, 1])
gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

z_lin = griddata(pts, vals, (gx, gy), method="linear")   # also "nearest", "cubic"
z_rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline")(
    np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
```

![Six panels: the true surface, then nearest, linear, cubic, thin-plate RBF and inverse-distance-weighted reconstructions from 80 black sample points. Nearest shows flat polygonal cells; linear and cubic leave a blank border outside the data; RBF fills the whole square smoothly; IDW shows small bumps around each sample.](fig-methods.png)

## Nearest neighbour

Each grid node takes the value of the closest sample. The result is a mosaic of **Voronoi cells**: every cell is the region closer to one sample than to any other. It is fast, never invents values outside the observed range, and works for categorical data (land-cover classes, labels). It is also blocky, with jumps at cell borders. Nearest neighbour is defined everywhere, so it never returns NaN.

## Linear: barycentric weights on a Delaunay triangulation

Linear interpolation first connects the samples into triangles with a **Delaunay triangulation**, the triangulation that avoids long thin triangles by making sure no sample lies inside another triangle's circumcircle. Inside a triangle with corners $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3$, any point can be written as

$$
\mathbf{x} = \lambda_1 \mathbf{p}_1 + \lambda_2 \mathbf{p}_2 + \lambda_3 \mathbf{p}_3, \qquad \lambda_1 + \lambda_2 + \lambda_3 = 1,\ \lambda_k \ge 0,
$$

where the $\lambda_k$ are the **barycentric coordinates**: how much the point "belongs" to each corner. The interpolated value uses the same weights: $\hat z = \lambda_1 z_1 + \lambda_2 z_2 + \lambda_3 z_3$. The surface is a set of flat triangular facets (a TIN, triangulated irregular network). It is continuous, but its slope jumps at triangle edges. Values stay inside the range of the three corners, so linear interpolation never overshoots.

## Cubic: Clough–Tocher

The `cubic` option in SciPy's `griddata` uses the **Clough–Tocher** scheme on the same triangulation. Each triangle is split into three sub-triangles, and a cubic polynomial is fitted on each so that values *and* first derivatives match across edges. The gradients at the samples are estimated from the data. The result is smooth ($C^1$) and was the most accurate method in my test, but unlike linear it can overshoot: near a sharp peak it may produce values above the highest sample.

## Outside the convex hull: NaN

Linear and cubic only know triangles, and triangles only cover the **convex hull** of the samples: the smallest convex polygon that contains them all. Grid nodes outside it get `NaN` (or the `fill_value` you pass). In my test 26.6% of the grid was outside the hull, the blank border in the figure. This is honest behaviour, since anything out there is extrapolation. Common fixes are:

- add known values at the domain corners or edges, if you really know them;
- fill the NaNs with a nearest-neighbour pass;
- use a method that extrapolates (RBF, IDW), and treat those values with care.

## Radial basis functions

An RBF interpolant is a weighted sum of identical bumps, one centred on each sample:

$$
\hat z(\mathbf{x}) = \sum_{i=1}^{n} w_i\, \varphi\big(\lVert \mathbf{x} - \mathbf{x}_i \rVert\big) + P(\mathbf{x}),
$$

where $\varphi$ is the **kernel** (a function of distance only), the $w_i$ are weights, and $P$ is a low-degree polynomial (for example a plane) that captures the overall trend. The weights come from solving an $n \times n$ linear system that forces $\hat z(\mathbf{x}_i) = z_i$. Common kernels:

- **thin-plate spline**, $\varphi(r) = r^2 \log r$: minimises bending energy, like a thin metal sheet pinned at the samples. A good default with no shape parameter.
- **multiquadric** $\sqrt{1 + (\varepsilon r)^2}$, **inverse multiquadric**, and **Gaussian** $e^{-(\varepsilon r)^2}$: these have a shape parameter $\varepsilon$ that must match the data spacing. Too large gives isolated spikes; too small gives an ill-conditioned system.
- **linear** $\varphi(r) = r$ and **cubic** $r^3$: simple, polyharmonic kernels.

RBFs fill the whole grid, including outside the hull, where they fall back to the polynomial trend. On my example the thin-plate RBF matched the cubic method inside the hull (RMSE 0.011) and kept a low error (0.024) over the full square.

### Smoothing for noisy data

Exact interpolation is a poor idea for noisy samples, because the surface must pass through every noisy value. The `smoothing` parameter $\lambda$ relaxes this: the system becomes $(\Phi + \lambda I)\mathbf{w} = \mathbf{z}$, where $\Phi$ is the matrix of kernel values between samples and $I$ the identity. Larger $\lambda$ trades fit for smoothness, similar to ridge regression.

![A noisy one-dimensional sine wave with three RBF fits: smoothing 0 wiggles through every point, smoothing 0.03 follows the true sine closely, smoothing 1 is too flat.](fig-rbf-smoothing.png)

On 25 noisy samples of a sine wave, the RMSE against the true curve was 0.26 with no smoothing, 0.065 with $\lambda = 0.03$, and 0.28 with $\lambda = 1$. Choose $\lambda$ by cross-validation, not by eye.

## Inverse distance weighting

IDW (Shepard's method) estimates each node as a weighted average of nearby samples, with weights that fall with distance:

$$
\hat z(\mathbf{x}) = \frac{\sum_i w_i z_i}{\sum_i w_i}, \qquad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)^p},
$$

where $d$ is the distance and $p$ the power (2 is common). Larger $p$ makes it more local. It is easy to write with a k-d tree:

```python
from scipy.spatial import cKDTree

def idw(pts, vals, query, k=8, p=2):
    d, i = cKDTree(pts).query(query, k=k)       # k nearest samples per query point
    w = 1.0 / np.maximum(d, 1e-12)**p           # avoid dividing by zero at a sample
    return (w * vals[i]).sum(1) / w.sum(1)

z_idw = idw(pts, vals, np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
```

IDW never overshoots: every estimate is a convex average of samples. Its known weakness is the **bull's-eye** effect visible in the figure: small plateaus around each sample, because the gradient goes to zero there. Far from all samples it flattens towards a local mean. It was less accurate than RBF or cubic on my smooth surface (RMSE 0.078 inside the hull), but it is robust and cheap.

## Anisotropy and units

Every method here measures plain Euclidean distance, so the units of $x$ and $y$ matter. If the field changes much faster in one direction (rows of a field, a river valley, or $x$ in metres and $y$ in kilometres), rescale the coordinates first so that "one unit" means a similar change in both directions. On a synthetic field that varied ten times faster in $y$ than in $x$, shrinking $x$ by 10 before a thin-plate RBF cut the RMSE from 0.142 to 0.012. Kriging handles this formally with an anisotropic variogram.

## Cost

- **Nearest, linear, cubic**: one Delaunay triangulation, $O(n \log n)$, then a point location per query. With 4,000 samples and 250,000 query points, linear `griddata` took 0.4 s on my laptop.
- **Global RBF**: solving the dense $n \times n$ system costs $O(n^3)$, and each query touches all $n$ samples. The same job took 15 s. Passing `neighbors=50` to `RBFInterpolator` fits a small local RBF per query instead, which took 6.7 s and scales to large $n$.
- **IDW** with a k-d tree: $O(\log n)$ per query, very fast.

## Old API: `interp2d` is gone

Older tutorials use `scipy.interpolate.interp2d` and the class `Rbf`. `interp2d` was deprecated in SciPy 1.10 and removed in 1.14; it now raises `NotImplementedError`. It also behaved badly on scattered data. Use `griddata` (or `LinearNDInterpolator`, `CloughTocher2DInterpolator`) for scattered points, `RBFInterpolator` instead of `Rbf`, and `RegularGridInterpolator` or `RectBivariateSpline` when the input is already on a grid. Note that `RBFInterpolator` takes points as an `(n, d)` array, not separate `x`, `y` arguments.

For gridding LiDAR ground points into a terrain model, where TIN and IDW are the usual choices, see [airborne LiDAR: DTM, DSM and CHM](/notes/airborne-lidar-dtm-dsm-chm/).

## Key takeaways

- Every interpolator encodes an assumption about the surface between samples; there is no neutral choice.
- Linear (Delaunay barycentric) is safe and never overshoots; cubic (Clough–Tocher) and thin-plate RBF are smoother and more accurate on smooth fields but can overshoot.
- Triangulation-based methods return NaN outside the convex hull; RBF and IDW extrapolate, which is not the same as being right.
- Use RBF smoothing (chosen by cross-validation) for noisy data, and rescale coordinates for anisotropic fields.
- `interp2d` no longer exists in SciPy; use `griddata`, `RBFInterpolator` or `RegularGridInterpolator`.

## Resources

- SciPy documentation, "Scattered data interpolation (griddata)" and `RBFInterpolator`: https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_unstructured.html
- SciPy documentation, "Interpolate transition guide" (replacing `interp2d`): https://docs.scipy.org/doc/scipy/tutorial/interpolate/interp_transition_guide.html
- Donald Shepard, "A two-dimensional interpolation function for irregularly-spaced data", ACM National Conference (1968): https://doi.org/10.1145/800186.810616
- Holger Wendland, *Scattered Data Approximation*, Cambridge University Press (2004), chapters on RBFs and conditionally positive definite kernels.
- Martin D. Buhmann, *Radial Basis Functions: Theory and Implementations*, Cambridge University Press (2003).
- R. W. Clough and J. L. Tocher, "Finite element stiffness matrices for analysis of plates in bending" (1965), origin of the Clough–Tocher element.
