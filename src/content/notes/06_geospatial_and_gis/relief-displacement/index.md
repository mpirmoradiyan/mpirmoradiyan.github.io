---
title: "Relief displacement: why tall objects lean outward in aerial photos"
slug: relief-displacement
date: 2023-02-20
topic: Geospatial & GIS
tags: [photogrammetry, relief-displacement, central-projection, drone, uav, tree-height, forestry]
summary: "Why the top of a tree or building appears farther from the image centre than its base, how the relation d = r·h/H follows from similar triangles, how to measure height from it, and what it means for trees and buildings in drone images."
sources:
  - title: "Elements of Photogrammetry with Applications in GIS (4th ed.), chapter 6: Vertical photographs (relief displacement)"
    author: "Paul R. Wolf, Bon A. DeWitt, Benjamin E. Wilkinson"
    type: book
  - title: "Remote Sensing and Image Interpretation (7th ed.), chapter 3: Basic principles of photogrammetry"
    author: "Thomas M. Lillesand, Ralph W. Kiefer, Jonathan W. Chipman"
    type: book
  - title: "Manual of Photogrammetry (6th ed.)"
    author: "ASPRS, ed. J. Chris McGlone"
    type: book


status: draft
---

Look at any drone photo of a forest or a town and you will see that trees and buildings near the edges seem to fall away from the centre. You see their sides, and their tops sit farther out than their bases. This is **relief displacement**. It is not a lens fault and it cannot be calibrated away. It is a direct result of the central projection, and it is the reason a raw photo cannot be used as a map. The same effect is also useful: the amount of lean tells you how tall an object is. This note derives the classic formula and shows what it means in practice.

## Where the lean comes from

In a vertical photo, every ray passes through the lens, which sits at height $H$ above the ground, directly above the nadir point. (For the geometry of the camera and of each pixel's viewing angle, see [nadir and off-nadir viewing](/notes/off-nadir-viewing-geometry/).)

Take a tree of height $h$ standing at horizontal distance $R$ from the nadir point. Its base and its top have the same horizontal position, but the top is $h$ metres closer to the camera. Each point is imaged at a scale equal to the focal length over its own distance below the lens:

$$
r_{\text{base}} = \frac{f\,R}{H}, \qquad r_{\text{top}} = \frac{f\,R}{H - h}.
$$

Because $H - h < H$, the top lands farther from the image centre than the base. Both points lie on the same line through the nadir point, so the shift is **radial**: everything leans straight away from the nadir point, like the spokes of a wheel.

## The formula

The displacement in the image is $d = r_{\text{top}} - r_{\text{base}}$. Substituting and simplifying gives the textbook relation:

$$
d = \frac{r\,h}{H},
$$

where

- $d$ is the relief displacement in the image (from the base to the top),
- $r$ is the radial distance in the image from the nadir point to the **displaced** point (the top),
- $h$ is the height of the object above its base,
- $H$ is the flying height above the base of the object.

$d$ and $r$ must be in the same image unit (millimetres on the sensor or pixels). $h$ and $H$ must be in the same ground unit (metres). The formula is a ratio, so it does not matter which units you pick as long as you do not mix them. That is the same rule that trips people up when computing viewing angles from pixel offsets and heights in metres.

Three things follow straight from it:

1. At the nadir point ($r = 0$) there is no displacement. A tree directly below the drone shows only its top.
2. Displacement grows linearly with distance from the centre and with object height.
3. Flying higher reduces it, but only at a fixed position on the ground. At a fixed position *in the frame* (a fixed off-nadir angle $\theta$), the lean measured at ground scale is $h\tan\theta \cdot H/(H-h)$, which barely depends on $H$ once $H \gg h$. A wide-angle lens flown high still sees the same tilted view at its edges. What really reduces the lean is a narrower field of view (a longer lens) or using only the central part of each photo.

## Measuring height from the lean

Turned around, the formula gives the height:

$$
h = \frac{d\,H}{r}.
$$

The code below projects a synthetic 22 m tree with a nadir camera and recovers its height from the image:

```python
import numpy as np
f_px, H = 3648.0, 100.0          # focal length (px), flying height above the ground (m)

def project(X, Y, Z):
    """Nadir camera at (0, 0, H): ground metres -> image pixels from the centre."""
    s = f_px / (H - Z)           # scale at elevation Z
    return X * s, Y * s

xb, yb = project(45.0, 20.0, 0.0)     # stem base
xt, yt = project(45.0, 20.0, 22.0)    # tree top
r_base, r_top = np.hypot(xb, yb), np.hypot(xt, yt)
d = r_top - r_base                    # 507 px
h = d * H / r_top                     # 22.00 m
```

The base is at 1796 px from the centre and the top at 2303 px, so the tree appears 507 px long, which is about 14 m at ground scale. One pixel of error in $d$ changes the height by $H / r \approx 4$ cm here, but much more for objects near the centre, where $r$ is small. The method works for buildings, poles and isolated trees where both the base and the top are visible. In a closed forest the stem base is usually hidden, which is why tree heights are normally taken from a surface model built from stereo overlap or from LiDAR (see [airborne LiDAR: DTM, DSM and CHM](/notes/airborne-lidar-dtm-dsm-chm/)). Stereo parallax is in fact the two-photo version of the same idea.

![Left: a synthetic nadir frame with 35 trees on a grid. Each black dot is a stem base and a green circle marks the crown top, joined by a brown line. Near the red cross at the centre the lines are short. Toward the edges they get longer and all point radially away from the centre. Right: the lean of a 20 m tree at ground scale against the off-nadir angle of its base, for flying heights of 60, 100 and 150 m. All three curves rise from 0 to roughly 20 to 27 m at 42 degrees and stay close to the dashed h times tan theta line.](fig-relief-displacement.png)

## What it means for trees and buildings in drone images

- **Crowns are not above their stems in the image.** At 37° off nadir and 100 m above the ground, a 20 m tree's top is displaced about 19 m at ground scale. If you mark tree positions on a raw photo and place them on a map with a single scale, the errors are at least the tree height times $\tan\theta$.
- **Objects look different across the frame.** Near the centre you see the top of a crown. Near the edge you see its side, and its outline is longer in the radial direction. A detector or a measurement that assumes a top-down view will behave differently at the edges than in the centre.
- **Tall objects hide what is behind them.** The leaning crown or building covers the ground on its far side (seen from the nadir point). Small trees next to tall ones can disappear at the edges of a frame and be visible in the next photo.
- **Use the right nadir point.** For a tilted photo, displacement is radial from the true nadir point, the point vertically below the lens, not from the image centre. A few degrees of gimbal tilt moves that point by hundreds of pixels.
- **Terrain counts too.** $H$ is the height above the object's base, so a tree on a ridge, closer to the drone, leans more than an identical tree in a valley at the same image position.

Relief displacement is also what an orthophoto is meant to remove. Removing it for the ground is easy if you have a terrain model. Removing it for trees and buildings is much harder, as explained in [orthorectification and true orthophotos](/notes/orthorectification-and-true-orthophotos/).

## Key takeaways

- Tall objects lean radially away from the nadir point because their tops are closer to the camera and imaged at a larger scale.
- $d = r\,h / H$: displacement grows with radial distance and object height and falls with flying height. Keep image units and ground units separate.
- At a fixed off-nadir angle, the lean at ground scale is close to $h\tan\theta$, so the viewing angle, not the flying height, is what matters most.
- Inverting the formula, $h = d\,H / r$, measures heights when the base and the top are both visible.

## Resources

- P. R. Wolf, B. A. DeWitt, B. E. Wilkinson, *Elements of Photogrammetry with Applications in GIS*, 4th ed., McGraw-Hill, 2014, chapter 6 (vertical photographs: relief displacement and heights from displacement).
- T. M. Lillesand, R. W. Kiefer, J. W. Chipman, *Remote Sensing and Image Interpretation*, 7th ed., Wiley, 2015, chapter 3 (relief displacement of vertical features).
- J. C. McGlone (ed.), *Manual of Photogrammetry*, 6th ed., ASPRS, 2013.
