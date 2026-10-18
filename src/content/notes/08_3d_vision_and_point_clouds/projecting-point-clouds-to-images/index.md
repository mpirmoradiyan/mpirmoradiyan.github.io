---
title: "Projecting point clouds to images: top-down rasters, spherical panoramas and the z-buffer"
slug: projecting-point-clouds-to-images
date: 2025-10-01
topic: 3D Vision & Point Clouds
tags: [point-cloud, projection, orthographic, spherical-projection, range-image, z-buffer, bev, lidar]
summary: "How to turn a 3D point cloud into a 2D image with an orthographic (top-down) or a spherical (panoramic) projection, how to decide which point owns a pixel, and how to pick the resolution."
sources:
  - title: "3D Data Science with Python, Chapter 4"
    author: "Florent Poux (O'Reilly, 2025)"
    type: book
  - title: "SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud"
    author: "Wu, Wan, Yue & Keutzer (2018)"
    url: "https://arxiv.org/abs/1710.07368"
    type: paper
  - title: "RangeNet++: Fast and Accurate LiDAR Semantic Segmentation"
    author: "Milioto, Vizzo, Behley & Stachniss (2019)"
    url: "https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf"
    type: paper
  - title: "Multi-View 3D Object Detection Network for Autonomous Driving"
    author: "Chen, Ma, Wan, Li & Xia (2017)"
    url: "https://arxiv.org/abs/1611.07759"
    type: paper


status: draft
---

Images have decades of tools behind them: 2D convolutions, pretrained networks, fast viewers, simple file formats. Point clouds have far fewer. A common trick is therefore to project the cloud onto an image, do the work in 2D, and carry the results back to the points. Two projections cover most cases: an **orthographic** one that looks straight down (a map or "bird's-eye view") and a **spherical** one that looks out in every direction from a single point (a panorama, or the native view of a spinning LiDAR). Both raise the same question: when several points land in one pixel, which one do you keep? I coded both while following Chapter 4 of *3D Data Science with Python*. This note explains them and fixes two problems in my first versions.

![Four images of one synthetic scene with a ground plane, a red box and a blue pole. First, a top view where the last point in the file wins: the box is speckled with grey ground. Second, the same view keeping the highest point: the box is solid. Third, an equirectangular colour panorama from a viewpoint 1.6 m above the ground. Fourth, the matching range image, with near ground in light colours and holes where no points fall.](fig-projections.png)

## Orthographic (top-down) projection

An orthographic projection drops one coordinate. For a top-down view you drop $z$ and lay a regular grid with cell size $\rho$ (metres per pixel) over $x$ and $y$:

$$
\text{col} = \left\lfloor \frac{x - x_{\min}}{\rho} \right\rfloor, \qquad \text{row} = \left\lfloor \frac{y_{\max} - y}{\rho} \right\rfloor.
$$

The row uses $y_{\max} - y$ because image rows grow downwards while northing grows upwards. Forget this and your map comes out upside down. The image size is $\lfloor (x_{\max}-x_{\min})/\rho \rfloor + 1$ by $\lfloor (y_{\max}-y_{\min})/\rho \rfloor + 1$.

Distances and areas are preserved: one pixel is always $\rho \times \rho$ metres, wherever it is. That is why this view is the basis of every GIS raster made from LiDAR (elevation models, canopy height models, intensity images) and of the "BEV" input used by many 3D object detectors for driving.

## Spherical (equirectangular) projection

A spherical projection puts the viewer at a centre $\mathbf{c}$ and records the direction to each point. Shift the cloud so that $\mathbf{c}$ is the origin, then convert $(x, y, z)$ to spherical coordinates:

$$
r = \sqrt{x^2 + y^2 + z^2}, \qquad \theta = \operatorname{atan2}(y, x), \qquad \varphi = \arccos\!\left(\frac{z}{r}\right).
$$

Here $r$ is the distance from the centre, $\theta \in (-\pi, \pi]$ is the **azimuth**, the angle around the vertical $z$ axis, and $\varphi \in [0, \pi]$ is the **polar angle**, measured down from straight up. (My first version described $\theta$ as a rotation around the $y$ axis. With `atan2(y, x)` it is a rotation around $z$.) Map the two angles linearly onto an image of height $H$ and width $W = 2H$:

$$
u = \frac{\theta + \pi}{2\pi}\,W, \qquad v = \frac{\varphi}{\pi}\,H.
$$

Width is twice the height because azimuth spans $2\pi$ and the polar angle spans $\pi$, so each pixel covers the same angle in both directions. This is the equirectangular format used for 360° photos. Storing $r$ in each pixel instead of colour gives a **range image**.

Unlike the top view, the pixel size here is an angle, not a length. A pixel covers more ground the farther away it looks, so nearby surfaces are densely sampled and distant ones are sparse. Above the horizon there is often nothing at all.

A spinning LiDAR fits this projection well. Its lasers are fixed at known elevation angles, so if you use one image row per laser and one column per firing step, every measurement gets its own pixel and nothing collides. Networks such as SqueezeSeg and RangeNet++ segment LiDAR scans in exactly this form and then copy the labels back to the points.

## Who owns a pixel: the z-buffer

Many points fall into the same pixel. The answer you want depends on the view:

- **Top view:** keep the **highest** point, because that is what you see from above. (For a terrain model you would keep the lowest ground point instead.)
- **Spherical view:** keep the **nearest** point, because it hides everything behind it.

This is the z-buffer from computer graphics. My first top-view code looped over the points and simply wrote each one into its pixel, so the last point in the file won. LAS files are usually stored in acquisition order, not by height, so wherever ground and roof points share a pixel the result is random. In the figure, the shuffled-order version shows grey ground speckled through the red box. In 144 box pixels, 20 showed the wrong colour.

A vectorised z-buffer avoids both the bug and the slow Python loop. Sort points by pixel, then by the key you want to minimise, and keep the first point in each pixel:

```python
import numpy as np

def zbuffer(pixel_id, key, n_pixels):
    """Index of the point with the smallest key in each pixel (-1 if empty)."""
    order = np.lexsort((key, pixel_id))               # sort by pixel, then by key
    first = np.unique(pixel_id[order], return_index=True)[1]
    winner = np.full(n_pixels, -1)
    winner[pixel_id[order][first]] = order[first]
    return winner

def ortho_top(points, res):
    xmin, ymax = points[:, 0].min(), points[:, 1].max()
    col = ((points[:, 0] - xmin) / res).astype(int)
    row = ((ymax - points[:, 1]) / res).astype(int)   # rows grow downwards
    H, W = row.max() + 1, col.max() + 1
    return zbuffer(row * W + col, -points[:, 2], H * W).reshape(H, W)  # highest wins

def spherical(points, center, H=180):
    p = points - center
    r = np.linalg.norm(p, axis=1)
    theta = np.arctan2(p[:, 1], p[:, 0])                # azimuth around z
    phi = np.arccos(np.clip(p[:, 2] / r, -1, 1))        # polar angle from +z
    W = 2 * H
    u = np.minimum(((theta + np.pi) / (2 * np.pi) * W).astype(int), W - 1)
    v = np.minimum((phi / np.pi * H).astype(int), H - 1)
    return zbuffer(v * W + u, r, H * W).reshape(H, W), r  # nearest wins
```

Returning the **index** of the winning point, not just its colour, is useful. You can then fill any channel (colour, height, intensity, class) from the same lookup, and after running a 2D model you can send each pixel's prediction back to its point. Points that lost the z-buffer test have no pixel. They need a second step, such as a nearest-neighbour vote in 3D, to get a label.

## Choosing the resolution

The pixel size trades holes against lost detail.

- If $\rho$ is **smaller than the point spacing**, many pixels stay empty. Fill them by interpolation, morphological closing, or by splatting each point onto a small disc instead of a single pixel.
- If $\rho$ is **much larger**, many points compete for each pixel and fine structure, like the thin pole in the figure, shrinks to a few pixels.

A good starting point for a top view is a pixel size close to the average point spacing, which is about $1/\sqrt{\text{density}}$ for a density in points per square metre. For a spherical image, match the angular resolution of the scanner or the angle between neighbouring points at the distance you care about.

## What a projection loses

Every projection throws away one dimension. The top view loses everything under the highest surface, which you can partly keep by storing several channels per pixel (minimum height, maximum height, point count). The spherical view loses everything behind the nearest surface and depends on where you put the centre. A spherical image from the middle of an airborne tile, for instance, is a valid picture but a strange one, because the data were never captured from there. For a true perspective camera with intrinsics and a pose, see [the camera extrinsic matrix](/notes/camera-extrinsic-matrix/). For other ways to store 3D data without dropping a dimension, see [representing 3D data](/notes/point-cloud-representations/).

## Key takeaways

- Orthographic projection keeps metric distances. Use $y_{\max} - y$ for rows so north is up.
- Spherical projection maps azimuth and polar angle onto a $2H \times H$ grid. It is the natural image of a spinning LiDAR.
- When points collide, decide on purpose: highest point for a top view, nearest point for a panorama. Writing points in file order gives random results.
- Keep the winning point's index per pixel so results can travel back to 3D.
- Choose the pixel size from the point spacing, and plan for holes.

## Resources

- Florent Poux, *3D Data Science with Python* (O'Reilly, 2025), Chapter 4, sections on orthographic and spherical projections. My code-along followed the chapter; the vectorised z-buffer is my own rewrite.
- B. Wu, A. Wan, X. Yue, K. Keutzer, ["SqueezeSeg"](https://arxiv.org/abs/1710.07368), ICRA 2018 (range-image segmentation of LiDAR).
- A. Milioto, I. Vizzo, J. Behley, C. Stachniss, ["RangeNet++"](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), IROS 2019 (spherical projection and back-projection of labels).
- X. Chen, H. Ma, J. Wan, B. Li, T. Xia, ["Multi-View 3D Object Detection Network for Autonomous Driving"](https://arxiv.org/abs/1611.07759), CVPR 2017 (bird's-eye-view and front-view projections of LiDAR).
