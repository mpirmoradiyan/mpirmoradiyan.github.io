---
title: "Nadir and off-nadir viewing: the viewing angle of every pixel in an aerial photo"
slug: off-nadir-viewing-geometry
date: 2023-02-20
topic: Geospatial & GIS
tags: [photogrammetry, central-projection, off-nadir, gsd, drone, uav, camera-geometry]
summary: "How to work out the angle each pixel of a drone photo looks at the ground from, without mixing up units, and how that angle, camera tilt and terrain change pixel size and what you see across the frame."
sources:
  - title: "Elements of Photogrammetry with Applications in GIS (4th ed.), chapters on vertical and tilted photographs"
    author: "Paul R. Wolf, Bon A. DeWitt, Benjamin E. Wilkinson"
    type: book
  - title: "Manual of Photogrammetry (6th ed.)"
    author: "ASPRS, ed. J. Chris McGlone"
    type: book
  - title: "Remote Sensing and Image Interpretation (7th ed.), chapter 3: Basic principles of photogrammetry"
    author: "Thomas M. Lillesand, Ralph W. Kiefer, Jonathan W. Chipman"
    type: book
  - title: "Pix4D support: Ground sampling distance (GSD) in photogrammetry"
    author: "Pix4D"
    url: "https://support.pix4d.com/hc/en-us/articles/202559809"
    type: docs


status: draft
---

A drone camera pointing straight down sees the middle of the frame from directly above, but it sees the corners at an angle of 40° or more. That angle decides how much of a tree's side or a wall you can see, how far the light travels through haze, and how much tall objects lean in the photo. So if you want to measure things in aerial images, it helps to know the viewing angle of every pixel. These notes are about how to compute it correctly from the camera's metadata, and what it means for pixel size. How to read the metadata itself, and the basic ground sampling distance (GSD) formula, are in [reading drone image metadata](/notes/drone-image-metadata-exif-xmp/).

## The central projection

A frame camera is a **central projection**. In plain words: every ray from the ground goes through one point, the perspective centre (the lens), and lands on the flat sensor behind it. For a vertical (nadir, straight-down) photo, the point right below the camera is the **nadir point**, and it falls on the principal point, the centre of the image.

From similar triangles, the scale of a vertical photo is the focal length over the height above the ground: $S = f / h$. The thing that clicked for me is that scale belongs to a height, not to the photo. A hilltop 30 m above the valley floor is 30 m closer to the camera, so it shows up at a bigger scale than the valley. Every elevation in the scene has its own scale. That one fact explains why a single photo isn't a map, and it's where relief displacement and the need for orthorectification come from.

In pixels, the same relation is $\text{GSD} = h / f_{px}$, where $f_{px}$ is the focal length measured in pixels:

$$
f_{px} = \frac{f}{p}, \qquad p = \frac{s_w}{W},
$$

with $f$ the focal length (mm), $s_w$ the sensor width (mm), $W$ the image width (px) and $p$ the pixel pitch on the sensor (mm/px). For a 1-inch sensor (13.2 mm, 5472 px) and an 8.8 mm lens, $f_{px} \approx 3648$ px.

## The viewing angle of one pixel

Take a pixel at column $u$ and row $v$. Its offset from the principal point $(c_x, c_y)$ is $x = u - c_x$, $y = v - c_y$ in pixels. In the camera frame, the ray through this pixel points in the direction $(x, y, f_{px})$. The angle between that ray and the optical axis is

$$
\theta = \arctan\frac{\sqrt{x^2 + y^2}}{f_{px}}.
$$

For a straight-down camera the optical axis is vertical, so $\theta$ is also the **off-nadir angle**: the angle between the line of sight and a plumb line. The top and the bottom of the fraction are both in pixels, so the units cancel.

```python
import numpy as np

def off_nadir_angle(W, H, f_mm, sensor_w_mm):
    p = sensor_w_mm / W                     # pixel pitch on the sensor (mm/px)
    f_px = f_mm / p                         # focal length in pixels
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x, y = u - W / 2, v - H / 2             # offsets from the principal point (px)
    return np.degrees(np.arctan(np.hypot(x, y) / f_px))

theta = off_nadir_angle(5472, 3648, 8.8, 13.2)
# centre ~0 deg, middle of the long edge 36.9 deg, corner 42.0 deg
```

The result looks like a bowl: circles of equal angle around the principal point (left panel of the figure below). The angle only depends on the camera, not on how high you fly.

## A units mistake I made

When I first tried this, I wrote the slant distance from the drone to a ground point as

```python
dist = np.sqrt(h**2 + dx**2 + dy**2)       # h in metres, dx and dy in pixels: wrong
angle = np.arccos(h / dist)
```

The idea is fine (the ray, the vertical and the ground offset make a right triangle), but the height is in metres while `dx` and `dy` are pixel offsets. Adding them makes no sense. At 100 m height, the middle of the long edge is 2736 px from the centre, and the wrong formula gives 87.9° instead of 36.9°. The fix is to put both sides of the triangle in the same unit. Either turn the pixel offset into metres on the ground with the GSD, or compare it to the focal length in pixels:

```python
h, dx = 100.0, 2736.0                          # metres, pixels
gsd = 13.2 * h / (8.8 * 5472)                  # metres per pixel on the ground
right = np.degrees(np.arccos(h / np.hypot(h, dx * gsd)))   # 36.9 deg, both in metres
same = np.degrees(np.arctan(dx / 3648.0))                  # 36.9 deg, both in pixels
```

Both give the same answer because $dx \cdot \text{GSD} / h = dx / f_{px}$. There's a simple test that catches this kind of bug: for a straight-down camera, the viewing angle must not change when you change the flying height.

Two smaller shortcuts are also worth avoiding:

- **Spreading angles evenly across the frame**, for example `np.linspace(-fov/2, fov/2, W)`. On a flat sensor it's the *tangent* of the angle that grows evenly with pixel position, not the angle itself. A quarter of the way from the centre to the edge, the even guess gives 18.4° instead of 20.6°.
- **Combining the horizontal and vertical angles as $\sqrt{\alpha_x^2 + \alpha_y^2}$.** That only works for small angles. At the corner of this camera it gives 45.4° instead of 42.0°. Combine the pixel offsets first, then take the arctangent.

## How pixel size changes across the frame

I assumed pixels near the edge cover more ground. For a truly vertical camera over flat ground, that's actually not true. A ground point at horizontal distance $R$ from the nadir point shows up $r = f_{px} R / h$ pixels from the centre, which is just a scaling. So the GSD is the same, $h / f_{px}$, everywhere in the frame. What does change towards the edges is:

- the **slant range**, which grows as $h / \cos\theta$ (about 25% longer at 37°), so haze effects and the size of the lens blur on the ground go up a little;
- the **viewing direction**: at the edge the camera sees objects partly from the side, so vertical surfaces (tree crowns, walls) become visible and tall objects lean outwards;
- the **lighting geometry**: the angle between the sun, the surface and the camera changes, so the same surface can look brighter or darker at the edge.

The GSD does change across the frame in two other cases.

**Tilt.** If the gimbal is pitched $t$ degrees away from straight down, a ray at angle $\alpha$ from the optical axis (in the tilt direction) hits the ground at $\varphi = t + \alpha$ from the vertical. Differentiating the ground position $h \tan\varphi$ gives the pixel size along and across the tilt direction:

$$
\text{GSD}_{\text{along}} = \frac{h \cos^2\alpha}{f_{px}\cos^2\varphi}, \qquad \text{GSD}_{\text{across}} = \frac{h \cos\alpha}{f_{px}\cos\varphi}.
$$

With $t = 0$ both come back to $h / f_{px}$. With a 15° tilt at 100 m, the far edge of the frame has 3.9 cm pixels along the tilt direction and the near edge 2.3 cm, compared with 2.7 cm for a straight-down photo. Pixels also stop being square.

```python
def gsd_profile(v_px, h, f_px, tilt_deg):
    a = np.arctan(v_px / f_px)                 # ray angle from the optical axis
    phi = np.radians(tilt_deg) + a             # ray angle from the vertical
    along = h * np.cos(a) ** 2 / (f_px * np.cos(phi) ** 2)
    across = h * np.cos(a) / (f_px * np.cos(phi))
    return along, across
```

I checked this against a brute-force projection of neighbouring pixels onto the ground.

**Terrain.** The $h$ in every formula is the height above the ground point you're looking at, not the drone's altitude. Over a slope, the uphill side of the frame has smaller pixels than the downhill side. And flying 100 m above the ground, a tree crown 25 m tall is imaged with pixels 25% smaller than the ground under it.

![Left: a heat map of the off-nadir angle for every pixel of a 5472 by 3648 nadir image, rising in circles from 0 degrees at the centre to about 42 degrees at the corners. Right: GSD along the vertical centre line at 100 m above flat ground. For a nadir camera the along and across lines coincide as one flat line at 2.74 cm. With 10 and 20 degree tilt, pixels shrink on the side toward nadir and grow on the far side, more steeply along the tilt direction than across it.](fig-off-nadir-angle-and-gsd.png)

## Where this leads

The off-nadir angle is what drives [relief displacement](/notes/relief-displacement/), and orthomosaic software tries to keep it small when it picks a photo for each ground cell (see [orthorectification and true orthophotos](/notes/orthorectification-and-true-orthophotos/)). The full matrix camera model is in the [camera extrinsic matrix](/notes/camera-extrinsic-matrix/) note, and [radial lens distortion](/notes/radial-lens-distortion/) shows how to correct edge pixels for real lenses first.

## Key takeaways

- The scale of a vertical photo is $f / h$, and it's different for every elevation in the scene.
- The off-nadir angle of a pixel is $\arctan(\sqrt{x^2 + y^2} / f_{px})$ with everything in pixels. It doesn't depend on flying height.
- Never mix metres and pixels in one triangle. Convert with the GSD or with the focal length in pixels.
- For a straight-down camera over flat ground, the GSD is the same across the frame. Tilt and terrain make it vary, and the viewing direction always changes towards the edges.

## Resources

- P. R. Wolf, B. A. DeWitt, B. E. Wilkinson, *Elements of Photogrammetry with Applications in GIS*, 4th ed., McGraw-Hill, 2014: vertical photographs (scale, ground coordinates) and tilted photographs.
- J. C. McGlone (ed.), *Manual of Photogrammetry*, 6th ed., ASPRS, 2013: camera geometry and the central projection.
- T. M. Lillesand, R. W. Kiefer, J. W. Chipman, *Remote Sensing and Image Interpretation*, 7th ed., Wiley, 2015, chapter 3: basic principles of photogrammetry.
- Pix4D support, [Ground sampling distance (GSD) in photogrammetry](https://support.pix4d.com/hc/en-us/articles/202559809).
