---
title: "The camera extrinsic matrix: rotation, translation and change of basis"
slug: camera-extrinsic-matrix
date: 2024-10-04
topic: Signal & Image Processing
tags: [camera-model, extrinsics, homogeneous-coordinates, rotation, image-formation, geometry]
summary: "How rotation, translation and a change of basis combine into the extrinsic matrix that turns world coordinates into camera coordinates, the first step of image formation."
sources:
  - title: "Camera Extrinsic Matrix with Example in Python (Image Formation and Camera Calibration series, part 2)"
    author: "Neeraj Krishna"
    url: "https://towardsdatascience.com/camera-extrinsic-matrix-with-example-in-python-cfe80acab8dd"
    type: course
  - title: "Image-formation-and-camera-calibration"
    author: "Neeraj Krishna (wingedrasengan927)"
    url: "https://github.com/wingedrasengan927/Image-formation-and-camera-calibration"
    type: repo
  - title: "Foundations of Computer Vision, Chapter 39: Camera Modeling and Calibration"
    author: "Antonio Torralba, Phillip Isola, William T. Freeman"
    url: "https://visionbook.mit.edu/imaging_geometry.html"
    type: book


status: draft
---

To form an image, a camera has to answer a simple question for every 3D point: where is it *relative to me*? Points are usually given in a shared world frame (a map, a room, a drone survey), but projection onto the image only makes sense in the camera's own frame, with the optical axis pointing forward. The **extrinsic matrix** does that conversion. It is built from just a rotation and a translation, but the order and direction of those two are easy to get wrong. I worked through it following Neeraj Krishna's tutorial series on image formation; this note is my own summary.

## Rotating a point in 2D

Rotate a point $(x, y)$ about the origin by an angle $\theta$, counter-clockwise. Write the point in polar form, $x = r\cos\alpha$ and $y = r\sin\alpha$. After rotation it sits at angle $\alpha + \theta$:

$$
x' = r\cos(\alpha + \theta) = r\cos\alpha\cos\theta - r\sin\alpha\sin\theta = x\cos\theta - y\sin\theta,
$$

$$
y' = r\sin(\alpha + \theta) = r\sin\alpha\cos\theta + r\cos\alpha\sin\theta = x\sin\theta + y\cos\theta.
$$

This is linear, so it is a matrix product:

$$
\begin{bmatrix}x'\\y'\end{bmatrix} =
\underbrace{\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}}_{R(\theta)}
\begin{bmatrix}x\\y\end{bmatrix}.
$$

In 3D there is one such matrix for each axis. For example, a rotation about the $Y$ axis is

$$
R_y(\theta) = \begin{bmatrix}\cos\theta & 0 & \sin\theta\\ 0 & 1 & 0\\ -\sin\theta & 0 & \cos\theta\end{bmatrix}.
$$

Rotation matrices are orthogonal: $R^{-1} = R^\top$, and $\det R = 1$.

## Moving the point vs. moving the axes

There are two ways to look at the same matrix.

- **Transforming the point**: the axes stay put and the point moves. $R(\theta)$ moves a point by $+\theta$.
- **Changing the basis**: the point stays put and the axes move. If we rotate the axes by $+\theta$, the point's coordinates in the new axes are the same as if the point had rotated by $-\theta$. So the change-of-basis matrix is $R(-\theta) = R(\theta)^{-1} = R(\theta)^\top$.

The same holds for translation. Moving a point by $t$ adds $t$. Moving the origin of the axes by $t$ subtracts $t$ from every coordinate.

The rule to remember: **the change-of-basis matrix is the inverse of the matrix that moves the frame.** If we know how the camera was rotated and moved in the world, inverting that motion tells us how world points look from the camera.

## Homogeneous coordinates

Rotation is a matrix product, but translation is an addition. To chain them as matrix products, we add a fourth coordinate equal to 1: the point $(X, Y, Z)$ becomes $(X, Y, Z, 1)$. Then a translation by $t$ is a $4 \times 4$ matrix,

$$
T(t) = \begin{bmatrix}I & t\\ 0^\top & 1\end{bmatrix},
\qquad
\text{and a rotation is}\quad
\begin{bmatrix}R & 0\\ 0^\top & 1\end{bmatrix}.
$$

Now any sequence of rotations and translations is a single matrix product.

## Building the extrinsic matrix

Describe the camera's **pose** in the world by:

- $R_c$: a rotation whose columns are the camera's $x$, $y$ and $z$ axes written in world coordinates;
- $C$: the camera centre in world coordinates.

The camera is placed by first rotating it by $R_c$ and then moving it to $C$. Its pose matrix is therefore $T(C)\,R_c$ (rightmost acts first). The extrinsic matrix is the inverse of the pose:

$$
\big(T(C)\,R_c\big)^{-1} = R_c^{-1}\,T(-C) = \begin{bmatrix}R_c^\top & -R_c^\top C\\ 0^\top & 1\end{bmatrix}.
$$

Dropping the last row gives the usual $3 \times 4$ form,

$$
E = \big[\,R \mid t\,\big], \qquad R = R_c^\top, \qquad t = -R_c^\top C,
$$

and a world point maps to camera coordinates as

$$
X_{\text{cam}} = E\begin{bmatrix}X_w\\1\end{bmatrix} = R_c^\top (X_w - C).
$$

In words: subtract the camera centre, then express the result in the camera's axes. Notice that $t$ is **not** the camera position. The camera position is $C = -R^\top t$. Mixing the two up is one of the most common bugs in camera code.

## A worked example

Put the camera at $C = (3, 0, -3)$, rotated by $-45°$ about $Y$, so its optical axis ($z_c$) points back towards the world origin.

![Top view of the world frame and a camera rotated -45 degrees about Y, with two world points and their rays to the camera](fig-world-and-camera-frames.png)

```python
import numpy as np

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

R_c = rot_y(-np.pi / 4)                 # camera axes in world coordinates
C = np.array([3.0, 0.0, -3.0])          # camera centre in world coordinates

E = np.hstack([R_c.T, (-R_c.T @ C)[:, None]])     # 3x4 extrinsic matrix [R | t]

for P_w in ([0, 0, 0], [1, 0, 0]):
    X_cam = E @ np.append(P_w, 1.0)               # homogeneous world point
    x_img = X_cam[:2] / X_cam[2]                  # pinhole projection, f = 1
    print(P_w, X_cam.round(3), x_img.round(3))
# [0, 0, 0] [0. 0. 4.243] [0. 0.]      -> straight ahead, 4.24 units away
# [1, 0, 0] [0.707 0. 3.536] [0.2 0.]  -> a little to the right
```

The world origin lands exactly on the optical axis at depth $\sqrt{3^2 + 3^2} \approx 4.24$, as the picture suggests, so it projects to the image centre. The point $(1, 0, 0)$ is a bit to the right and a bit closer.

In the tutorial notebook I followed, the camera was rotated by $\pi/4$ about $Y$, moved by $(0, -8, 0)$, and the extrinsic matrix was computed as `np.linalg.inv(R_ @ T_)`. That is the same idea. Be careful, though, that libraries differ in whether their rotation helpers return the active matrix (rotate the point) or the passive one (rotate the axes), which flips the sign of the angle. Always check one point by hand.

## From camera coordinates to pixels

The extrinsic matrix is only half of the camera model. The **intrinsic matrix** $K$ then maps camera coordinates to pixels:

$$
\lambda\begin{bmatrix}u\\v\\1\end{bmatrix} = K\,[\,R \mid t\,]\begin{bmatrix}X_w\\1\end{bmatrix},
\qquad
K = \begin{bmatrix}f_x & s & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1\end{bmatrix},
$$

where $f_x, f_y$ are the focal lengths in pixels, $(c_x, c_y)$ is the principal point, $s$ is the skew (usually 0) and $\lambda$ is the depth, divided out at the end. Real lenses add distortion on top of this (see [Radial lens distortion](/notes/radial-lens-distortion/)).

## Key takeaways

- A rotation matrix moves points; the change-of-basis matrix that moves the axes is its inverse (its transpose).
- Homogeneous coordinates turn translation into a matrix product so rotation and translation can be chained.
- The extrinsic matrix is the inverse of the camera pose: $E = [R_c^\top \mid -R_c^\top C]$, so $X_{\text{cam}} = R_c^\top(X_w - C)$.
- The translation $t$ in $[R \mid t]$ is not the camera position; the position is $C = -R^\top t$.
- Full projection is $K[R \mid t]$, followed by division by depth.

## Resources

- Neeraj Krishna, "Camera Extrinsic Matrix with Example in Python", Towards Data Science, part 2 of the series "Image Formation and Camera Calibration" (the 2D rotation and change-of-basis derivations and the worked example my notebook followed): <https://towardsdatascience.com/camera-extrinsic-matrix-with-example-in-python-cfe80acab8dd>
- Companion code repository: <https://github.com/wingedrasengan927/Image-formation-and-camera-calibration>
- Antonio Torralba, Phillip Isola and William T. Freeman, *Foundations of Computer Vision*, MIT Press, 2024, Chapter 38 "Representing Images and Geometry" (homogeneous coordinates) and Chapter 39 "Camera Modeling and Calibration" (intrinsic and extrinsic parameters): <https://visionbook.mit.edu/imaging_geometry.html>
- Richard Hartley and Andrew Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004, Chapter 6 "Camera Models".
