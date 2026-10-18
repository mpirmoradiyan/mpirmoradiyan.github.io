---
title: "Haze, the scattering model and the dark channel prior"
slug: haze-model-and-dark-channel-prior
date: 2022-11-14
topic: Signal & Image Processing
tags: [dehazing, dark-channel-prior, guided-filter, image-formation, image-restoration]
summary: "How haze is modelled as a depth-dependent blend of the scene and the airlight, and how the dark channel prior inverts that blend from a single image."
sources:
  - title: "Single Image Haze Removal Using Dark Channel Prior (CVPR 2009; extended in IEEE TPAMI 33(12), 2011)"
    author: "Kaiming He, Jian Sun, Xiaoou Tang"
    type: paper
  - title: "Guided Image Filtering (ECCV 2010; extended in IEEE TPAMI 35(6), 2013)"
    author: "Kaiming He, Jian Sun, Xiaoou Tang"
    type: paper
  - title: "Vision and the Atmosphere (IJCV 48(3), 2002)"
    author: "Srinivasa G. Narasimhan, Shree K. Nayar"
    type: paper
  - title: "Visibility in Bad Weather from a Single Image (CVPR 2008)"
    author: "Robby T. Tan"
    type: paper
  - title: "Gated Context Aggregation Network for Image Dehazing and Deraining (WACV 2019)"
    author: "Dongdong Chen et al."
    url: "https://github.com/cddlyf/GCANet"
    type: paper


status: draft
---

Haze and fog wash out an image: distant objects become pale, low in contrast and tinted towards the colour of the sky. This is not random noise. It follows a simple physical model, and once you write that model down, removing haze becomes an inverse problem with two unknowns per pixel. The dark channel prior is a remarkably simple trick for solving it from one image. I implemented it, together with the guided filter it relies on, while experimenting with foggy images in 2022.

## The haze image formation model

Light from a scene point travels to the camera through air full of small particles. Two things happen along the way. Part of the light from the object is scattered out of the path, so the direct signal fades. At the same time, ambient light (from the sky, the sun, the fog itself) is scattered *into* the path and adds a veil. The standard model, from Koschmieder's work on atmospheric visibility and popularised in vision by Narasimhan and Nayar, is

$$
I(x) = J(x)\,t(x) + A\,\big(1 - t(x)\big), \qquad t(x) = e^{-\beta d(x)}.
$$

- $I(x)$ is the observed hazy colour at pixel $x$ (an RGB vector in $[0,1]$).
- $J(x)$ is the scene radiance we want: the haze-free colour.
- $A$ is the **atmospheric light** (airlight), the colour of the haze at infinite distance. It is one RGB vector for the whole image.
- $t(x) \in [0,1]$ is the **transmission**: the fraction of the object's light that survives the trip.
- $\beta$ is the scattering coefficient of the atmosphere and $d(x)$ the distance to the scene point.

So each hazy pixel is a blend of the true colour and the airlight, and the blend weight depends only on depth. Near objects ($t \approx 1$) look normal. Far objects ($t \to 0$) converge to $A$. This also explains the two effects I saw when I looked at the minimum and maximum of the three colour channels of foggy images. Blending every pixel towards a nearly grey $A$ raises the minimum channel (dark pixels disappear), and it shrinks the gap between the largest and smallest channel, which is a rough measure of saturation.

![Histograms of the min channel and of max minus min for a synthetic clean scene and its hazy version; haze shifts the min channel up and collapses max minus min](fig-min-max-channels.png)

On a synthetic scene, the median of the min channel rose from 0.10 to 0.68 after adding haze, while the median of max − min fell from 0.32 to 0.08.

## The dark channel prior

Given only $I$, the model has too many unknowns: $J$ and $t$ at every pixel plus $A$. He, Sun and Tang (2009) noticed a statistical regularity of outdoor haze-free images. In almost every small patch that is not sky, at least one pixel has a very low value in at least one colour channel. Shadows, dark objects and strongly coloured surfaces (green leaves have low red, blue water has low red and green) make sure of this. They define the **dark channel**

$$
J^{\text{dark}}(x) = \min_{y \in \Omega(x)} \; \min_{c \in \{r,g,b\}} J^c(y),
$$

where $\Omega(x)$ is a small square patch around $x$ (they used 15×15). The prior says $J^{\text{dark}}(x) \approx 0$ for haze-free outdoor images. In a hazy image, the airlight term fills those dark pixels in, so the dark channel of $I$ becomes bright where haze is thick. The dark channel of a hazy image is, roughly, a haze-thickness map.

## From the prior to the transmission

Divide the model by $A$ channel by channel, take the min over colours and over the patch, and assume $t$ is constant inside the patch:

$$
\min_{y\in\Omega(x)}\min_c \frac{I^c(y)}{A^c} = t(x)\,\min_{y\in\Omega(x)}\min_c \frac{J^c(y)}{A^c} + 1 - t(x).
$$

The prior makes the first term on the right vanish, which gives

$$
\tilde t(x) = 1 - \omega \min_{y\in\Omega(x)}\min_c \frac{I^c(y)}{A^c}.
$$

The factor $\omega$ (0.95 in the paper) keeps a little haze on purpose. Without it, distant objects look flat and unnatural, because people use haze as a depth cue.

The **atmospheric light** is estimated from the dark channel too. Take the 0.1% brightest pixels of the dark channel (the most haze-opaque region) and, among them, pick the pixel with the highest intensity in the input. This is more robust than just picking the brightest pixel in the image, which is often a white car or a lamp.

Finally, invert the model:

$$
J(x) = \frac{I(x) - A}{\max\big(t(x), t_0\big)} + A.
$$

The floor $t_0$ (typically 0.1) matters. Where $t$ is tiny, dividing by it amplifies noise and JPEG artefacts enormously.

## Refining the transmission with a guided filter

Because $\tilde t$ is computed from patch minima, it is blocky and leaks across object edges, leaving halos in the result. The original paper refined it with soft matting, which is slow. The later **guided filter** (He et al., 2010) does the same job in linear time. It assumes the output $q$ is a local linear function of a guide image $G$ (here the grey hazy image) in each window: $q = a\,G + b$. Solving a small least-squares problem per window gives

$$
a = \frac{\operatorname{cov}(G, p)}{\operatorname{var}(G) + \epsilon}, \qquad b = \bar p - a\,\bar G,
$$

where $p$ is the input to be filtered (the rough transmission) and $\epsilon$ controls how much smoothing happens in flat regions. The coefficients are then averaged over overlapping windows. Everything is made of box filters, so the cost does not depend on the window size. Where the guide has an edge, $\operatorname{var}(G)$ is large, $a \approx 1$, and the edge is copied into $t$; in flat areas $a \approx 0$ and $t$ is smoothed.

## The whole method in a few lines

```python
import numpy as np, cv2

def dark_channel(img, size=15):
    m = img.min(axis=2)                                   # min over colours
    return cv2.erode(m, np.ones((size, size), np.uint8))  # min over a patch

def estimate_A(img, dc):
    n = max(1, dc.size // 1000)                           # brightest 0.1 %
    idx = np.argsort(dc.ravel())[-n:]
    cand = img.reshape(-1, 3)[idx]
    return cand[cand.sum(1).argmax()]

def guided_filter(guide, p, r=30, eps=1e-3):
    box = lambda x: cv2.boxFilter(x, cv2.CV_64F, (r, r))
    mI, mp = box(guide), box(p)
    a = (box(guide * p) - mI * mp) / (box(guide * guide) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * guide + box(b)

def dehaze(I, omega=0.95, t0=0.1):                        # I: float RGB in [0, 1]
    A = estimate_A(I, dark_channel(I))
    t = 1 - omega * dark_channel(I / A)
    t = guided_filter(I.mean(axis=2), t)
    J = (I - A) / np.maximum(t, t0)[..., None] + A
    return np.clip(J, 0, 1), t, A
```

I tested it on a synthetic scene: coloured textured blocks, a depth that grows from bottom to top, and haze added with the model above ($\beta = 0.7$, slightly bluish $A$). The RMSE against the clean image dropped from 0.42 for the hazy input to 0.07 after dehazing.

![Five panels: synthetic clean scene, its hazy version, the dark channel, the refined transmission, and the recovered image; a white patch at bottom left is wrongly treated as thick haze](fig-dark-channel-pipeline.png)

## Where the prior breaks

The white patch at the bottom left of the figure is the important lesson. It is a bright, nearly colourless surface close to the camera, like snow. Its dark channel is high *even without haze*, so the method decides it is far away in thick fog: the estimated transmission there was 0.09 against a true 0.59, and the "recovered" colour is wrong. In this example the bright patch also pulled the airlight estimate towards white. The same failure appears with:

- **Sky**, which has no dark pixels. The $t_0$ floor and $\omega$ limit the damage, but sky regions often get colour shifts or noise.
- **White or grey objects** (buildings, roads, vehicles) that look like haze to the prior.
- **Snow-covered scenes**, where most of the image violates the prior at once. For such images, plain contrast enhancement on the luminance channel is often the safer tool (see [contrast enhancement](/notes/contrast-enhancement-he-clahe-gamma/)).
- **Non-uniform light**, such as night fog with street lamps, where a single global $A$ is wrong.

## Learned dehazing

Deep networks replace the hand-made prior with one learned from data. DehazeNet (Cai et al., 2016) regresses the transmission and still uses the physical model for recovery. AOD-Net (Li et al., 2017) folds $t$ and $A$ into a single learned map and outputs $J$ directly. GCANet (Chen et al., 2019) is an end-to-end encoder–decoder with smoothed dilated convolutions and gated fusion of features from several levels; it predicts the residual between hazy and clean images. CycleGAN-based methods learn from unpaired hazy and clear images.

The catch is training data. Real pairs of hazy and clean images of the same scene are rare, so most models are trained on synthetic haze made with exactly the formula above and a depth map (for example the RESIDE benchmark). They can be excellent on benchmarks and disappointing on real fog whose colour, density or texture differs. The dark channel prior has no training set to mismatch, which is why it remains a strong baseline and a good first thing to try.

## Key takeaways

- Haze follows $I = J\,t + A(1-t)$ with $t = e^{-\beta d}$: a depth-dependent blend of the scene and the airlight.
- Haze raises the minimum colour channel and lowers saturation; the dark channel of a hazy image is a rough haze-density map.
- The dark channel prior gives $t = 1 - \omega\,\text{dark}(I/A)$; recovery divides by $\max(t, t_0)$ to avoid amplifying noise.
- A guided filter turns the blocky transmission into an edge-aware one cheaply, using only box filters.
- The prior fails on sky, white objects and snow; learned methods avoid hand-made priors but inherit the gap between synthetic and real haze.

## Resources

- He, K., Sun, J., Tang, X. "Single Image Haze Removal Using Dark Channel Prior." CVPR 2009; extended version in IEEE TPAMI 33(12), 2011.
- He, K., Sun, J., Tang, X. "Guided Image Filtering." ECCV 2010; extended version in IEEE TPAMI 35(6), 2013.
- Narasimhan, S. G., Nayar, S. K. "Vision and the Atmosphere." IJCV 48(3), 2002.
- Tan, R. T. "Visibility in Bad Weather from a Single Image." CVPR 2008.
- Cai, B. et al. "DehazeNet: An End-to-End System for Single Image Haze Removal." IEEE TIP, 2016.
- Li, B. et al. "AOD-Net: All-in-One Dehazing Network." ICCV 2017.
- Chen, D. et al. "Gated Context Aggregation Network for Image Dehazing and Deraining." WACV 2019. Code: https://github.com/cddlyf/GCANet
- Zhao, J. et al. "Image dehazing method via a cycle generative adversarial network." IET Image Processing, 2021.
- Li, B. et al. "Benchmarking Single-Image Dehazing and Beyond" (RESIDE). IEEE TIP, 2019.
