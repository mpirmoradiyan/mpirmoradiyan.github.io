---
title: "Representing 3D data: points, voxels, octrees, meshes and depth maps"
slug: point-cloud-representations
date: 2025-10-01
topic: 3D Vision & Point Clouds
tags: [point-cloud, voxels, octree, mesh, depth-map, rgbd, lidar, 3d-data]
summary: "What each common 3D representation stores, what it costs in memory, how to convert between them, and which one to pick for a given task."
sources:
  - title: "3D Data Science with Python, Chapters 3 and 4"
    author: "Florent Poux (O'Reilly, 2025)"
    type: book
  - title: "Open3D documentation: point clouds, voxelization, octrees, meshes, RGBD images"
    author: "Open3D"
    url: "https://www.open3d.org/docs/release/"
    type: docs
  - title: "LAS Specification 1.4 R15"
    author: "ASPRS"
    url: "https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf"
    type: docs
  - title: "Octree encoding: a new technique for the representation, manipulation and display of arbitrary 3-D objects by computer"
    author: "Meagher (1980)"
    type: paper


status: draft
---

A photo is always a grid of pixels. 3D data has no single natural form. A laser scanner gives a bag of points, a game engine wants triangles, a neural network may want a regular grid, and an RGB-D camera gives an image whose pixels hold distances. Each form makes some operations cheap and others hard, so the first design decision in any 3D project is which representation to work in. I worked through these while coding along with Chapters 3 and 4 of Florent Poux's *3D Data Science with Python*. This note collects the ideas in a library-neutral way.

![Left: a synthetic scene of a ground plane and a sphere as raw points. Middle: the same scene as a 0.25 m voxel grid. Right: on a log scale, the number of occupied octree nodes grows about four times per level, while a dense grid grows eight times.](fig-representations.png)

## Raw points

A point cloud is an $N \times 3$ array of coordinates, often with extra columns per point: colour, intensity, a class label, a normal vector. Two facts shape everything you do with it:

- **No order.** Row 17 and row 18 need not be neighbours in space. Any algorithm must give the same answer when the rows are shuffled.
- **No regular spacing.** Density depends on the distance to the sensor, the scan angle and occlusion. There are no "pixels" and no fixed neighbourhood.

So the basic question "which points are near this one?" needs a spatial index such as a k-d tree or one of the grids below.

File formats differ in small ways that matter. Plain text (`.xyz`, `.txt`, `.csv`) is easy to read but large and slow. PLY stores points, and optionally faces, as text or binary. LAS/LAZ, the standard for airborne LiDAR, stores coordinates as **scaled integers**: the real value is $x = X \cdot s_x + o_x$, where $X$ is the stored integer, $s_x$ is the scale (for example 0.01 m) and $o_x$ the offset from the header. Readers normally apply this for you, but if you touch the raw integer fields you must apply it yourself. LAS colours are 16-bit, so divide by 65,535, not 255, to get values in $[0, 1]$.

## Voxel grids

A voxel grid cuts space into cubes of side $s$ and assigns each point to the cube that contains it:

$$
\mathbf{i} = \left\lfloor \frac{\mathbf{p} - \mathbf{p}_{\min}}{s} \right\rfloor,
$$

where $\mathbf{p}$ is the point, $\mathbf{p}_{\min}$ the grid origin and $\mathbf{i}$ the integer index of its voxel. Once points share an index you can store per voxel whatever you need: occupancy (yes or no), a count, a mean colour, or the centroid. Keeping one centroid per voxel is the standard way to **downsample** a cloud to an even density.

```python
import numpy as np

def voxel_downsample(points, size):
    """Replace all points in each occupied voxel by their centroid."""
    idx = np.floor((points - points.min(0)) / size).astype(np.int64)
    keys, inverse = np.unique(idx, axis=0, return_inverse=True)
    sums = np.zeros((len(keys), 3))
    np.add.at(sums, inverse, points)             # sum the points per voxel
    counts = np.bincount(inverse)[:, None]
    return sums / counts, keys                    # centroids, voxel indices
```

The cost of a voxel grid depends on how you store it. A **dense** 3D array has $(L/s)^3$ cells for a scene of size $L$, so halving the voxel size multiplies memory by eight. Most of those cells are empty, because scanners see surfaces, not volumes. On the synthetic scene in the figure (a 4 m plane with a sphere), only 5% of the cells in the bounding box are occupied at 5 cm. A **sparse** grid, such as a hash map from index to value or the `unique` call above, stores only the occupied ones.

Voxels are a good fit when you need a regular structure: 3D convolutions, occupancy maps for robot navigation, volume estimates, or quick neighbourhood lookups ("the 26 voxels around this one").

## Octrees

An octree is a voxel grid built level by level. Start with one cube around the whole cloud. Split it into eight children of half the side length, and keep only the children that contain points. Repeat to a chosen maximum depth $d$. At depth $d$ the cell size is $L / 2^d$.

The win is that empty space is never subdivided. A dense grid at depth $d$ has $8^d$ cells. For points that lie on surfaces, the number of occupied nodes grows roughly by four per level, the way the area of a surface grows when you halve the measuring unit. In the figure, at depth 8 the dense grid would have 16.8 million cells, while the octree has about 28,000 occupied leaves, close to the number of points. Once each leaf holds about one point, deeper levels add nothing.

The same tree gives you every resolution at once. You can stop at a shallow level for a coarse preview (point cloud viewers and streaming formats such as Entwine Point Tiles work this way), or descend only where you need detail. Octrees also make neighbour search and collision tests fast, because whole empty branches are skipped.

## Meshes

A triangle mesh stores vertices ($V$, an $n \times 3$ array) and faces ($F$, an $m \times 3$ array of vertex indices). Unlike points, a mesh describes a **surface**: it knows what lies between samples, it has a well-defined area and normals, and it renders with proper shading. Normals are not always stored in the file. Compute them before rendering, or the mesh will look flat and unlit.

Going from mesh to points is easy. To sample a mesh **uniformly**, pick each triangle with probability proportional to its area, then pick a random point inside it with barycentric coordinates:

```python
def sample_mesh(V, F, n, rng=np.random.default_rng(0)):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(F), n, p=area / area.sum())   # big triangles get more points
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1                                   # fold the square onto the triangle
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[tri] + u[:, None] * (b - a)[tri] + v[:, None] * (c - a)[tri]
```

Picking triangles uniformly instead of by area would crowd points into small triangles. Going from points to a mesh is the hard direction (surface reconstruction, for example Poisson or ball pivoting) and needs good normals and even density.

## Depth maps and RGB-D images

A depth map is an image whose pixel $(u, v)$ holds the distance $Z$ along the camera's viewing axis. Paired with a colour image it is an RGB-D image, the output of sensors such as the Kinect or many phone cameras. It is often called **2.5D**: one depth per pixel, so anything hidden behind the first surface is missing.

With the camera intrinsics (focal lengths $f_x, f_y$ in pixels and principal point $c_x, c_y$), each pixel turns back into a 3D point in the camera frame:

$$
X = \frac{(u - c_x)\,Z}{f_x}, \qquad Y = \frac{(v - c_y)\,Z}{f_y}.
$$

```python
Z = depth_u16.astype(float) / 1000.0          # stored in millimetres -> metres
u, v = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
X, Y = (u - cx) * Z / fx, (v - cy) * Z / fy
cloud = np.stack([X, Y, Z], -1).reshape(-1, 3)[Z.ravel() > 0]   # 0 = no reading
```

Keep depth in its native 16-bit form and convert with the sensor's **depth scale**. In my code-along I divided a 16-bit depth image by 256 to squeeze it into 8 bits so it would display. That is fine for a picture, but each grey level then spans 25.6 cm, so the depth is no longer usable for 3D. To place the points in a world frame, apply the camera pose afterwards (see [the camera extrinsic matrix](/notes/camera-extrinsic-matrix/)).

Depth maps are compact and fit straight into 2D CNNs, which is why many networks work on depth or range images and not on raw points. Projecting a full cloud into such an image is the subject of [projecting point clouds to images](/notes/projecting-point-clouds-to-images/).

## Which one to use

| Representation | Stores | Good for | Weak at |
|---|---|---|---|
| Points | samples + attributes | raw sensor data, measurement, classification per point | neighbourhoods, surfaces, rendering |
| Voxel grid | values on a regular 3D grid | downsampling, occupancy maps, 3D CNNs, volumes | memory at fine resolution (if dense), loses detail below $s$ |
| Octree | adaptive grid, empty space pruned | multi-resolution viewing, streaming, fast search | more complex to update |
| Mesh | surface (vertices + triangles) | rendering, area, simulation, CAD/BIM | creating it from noisy points |
| Depth / RGB-D | one depth per pixel from one viewpoint | 2D networks, real-time sensing | occluded parts, only one view |

In practice a pipeline moves between them: scan to points, voxel-downsample, index with an octree or k-d tree, maybe reconstruct a mesh for display, or render depth images for a network.

## Key takeaways

- A point cloud is unordered and unevenly dense, so neighbourhood queries need a spatial structure.
- Voxelization is just integer division of coordinates. Keeping one centroid per voxel evens out density.
- Dense voxel grids grow as $8^d$; octrees and sparse grids store only occupied cells, which grow about $4^d$ for surface data.
- Meshes describe surfaces. Sample them by triangle area to get uniform points.
- Depth maps become points through the pinhole model, but only if you keep the full-precision depth and the correct depth scale.

## Resources

- Florent Poux, *3D Data Science with Python* (O'Reilly, 2025), Chapter 3 (point cloud basics) and Chapter 4 (3D data representations). My code-along followed this chapter; the code above is my own rewrite.
- Open3D documentation: [point clouds, voxelization, octrees, meshes and RGBD images](https://www.open3d.org/docs/release/).
- ASPRS, [LAS Specification 1.4](https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf) (scaled integer coordinates, 16-bit colour).
- D. Meagher, "Octree encoding: a new technique for the representation, manipulation and display of arbitrary 3-D objects by computer", Rensselaer Polytechnic Institute technical report, 1980.
