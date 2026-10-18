---
title: "Thresholding an image: the mean, Otsu, Ridler–Calvard, Gaussian mixtures and adaptive thresholds"
slug: global-thresholding-otsu-and-mixtures
date: 2022-03-03
topic: Signal & Image Processing
tags: [thresholding, otsu, isodata, gaussian-mixture, adaptive-threshold, segmentation]
summary: "How to pick a grey-level threshold automatically from the histogram, how Otsu, Ridler–Calvard and a two-component Gaussian mixture relate, and when a single global threshold fails and a local one is needed."
sources:
  - title: "A Threshold Selection Method from Gray-Level Histograms"
    author: "Nobuyuki Otsu (IEEE Trans. Systems, Man, and Cybernetics, 1979)"
    url: "https://doi.org/10.1109/TSMC.1979.4310076"
    type: paper
  - title: "Picture Thresholding Using an Iterative Selection Method"
    author: "T. W. Ridler, S. Calvard (IEEE Trans. Systems, Man, and Cybernetics, 1978)"
    url: "https://doi.org/10.1109/TSMC.1978.4310039"
    type: paper
  - title: "Digital Image Processing, 4th ed., Chapter 10 (Thresholding)"
    author: "Rafael C. Gonzalez, Richard E. Woods"
    type: book
  - title: "Pattern Recognition and Machine Learning, Section 9.2 (Mixtures of Gaussians)"
    author: "Christopher M. Bishop"
    type: book
  - title: "OpenCV tutorial: Image Thresholding"
    author: "OpenCV"
    url: "https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html"
    type: docs
  - title: "Practical Python and OpenCV (thresholding chapter)"
    author: "Adrian Rosebrock (PyImageSearch)"
    url: "https://pyimagesearch.com/practical-python-opencv/"
    type: book


status: draft
---

Thresholding is the simplest segmentation there is: every pixel brighter than $t$ is foreground, every other pixel is background. It is still useful, for coins on a table, text on paper, bright snow against dark ground, or as a quick baseline before training a model. The whole difficulty is choosing $t$. This note goes from the naive choice (the mean) to three principled ways of reading $t$ off the histogram, and then to adaptive thresholds for images where one number cannot work.

All examples use a synthetic image: bright discs (grey level 160) on a dark background (70), with Gaussian noise of standard deviation 18. The discs cover 18% of the pixels.

## The naive baseline: threshold at the mean

My first segmentation notebook simply compared each pixel to the image mean. It runs, but it only works when the two classes have about the same size. The mean is pulled towards the larger class. Here the background is 82% of the image, so the mean is 86, much closer to the background peak (70) than to the middle of the gap, and many noisy background pixels end up as foreground. Only 84.8% of the pixels are labelled correctly.

## Otsu: maximise the separation between the two classes

Otsu (1979) asks a cleaner question: which threshold splits the histogram into two groups that are as different as possible? Let $p_i$ be the fraction of pixels at level $i$. For a threshold $t$, the two classes have weights and means

$$
\omega_0(t) = \sum_{i \le t} p_i, \quad \omega_1 = 1 - \omega_0, \quad
\mu_0(t) = \frac{1}{\omega_0}\sum_{i \le t} i\,p_i, \quad \mu_1(t) = \frac{1}{\omega_1}\sum_{i > t} i\,p_i.
$$

The **between-class variance** is $\sigma_B^2(t) = \omega_0\,\omega_1\,(\mu_0 - \mu_1)^2$. The total variance of the image does not depend on $t$, and it equals the between-class variance plus the within-class variance. So maximising $\sigma_B^2$ is the same as minimising the weighted spread inside each class. Using cumulative sums, every threshold is tested in one pass:

```python
import numpy as np

def otsu(gray):                                   # gray: uint8 image
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    w0 = np.cumsum(p)                             # weight of the class below t
    mu = np.cumsum(p * np.arange(256))            # cumulative mean
    mu_T = mu[-1]                                 # global mean
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_T * w0 - mu) ** 2 / (w0 * (1 - w0))
    return int(np.nanargmax(sigma_b))
```

(The formula in the code is the same $\sigma_B^2$ rewritten with cumulative sums.) On the synthetic image it gives $t = 113$, the same as `skimage.filters.threshold_otsu` and OpenCV's `THRESH_OTSU`, and 99.3% of pixels are correct.

## Ridler–Calvard: the iterative mean of means

Ridler and Calvard (1978), also called the *isodata* method, use a fixed-point iteration:

1. Start with a guess, for example the global mean.
2. Split the pixels at $t$ and compute the mean of each side, $m_0$ and $m_1$.
3. Set $t \leftarrow (m_0 + m_1)/2$ and repeat until $t$ stops changing.

```python
def ridler_calvard(gray, tol=0.5):
    t = gray.mean()
    while True:
        t_new = (gray[gray <= t].mean() + gray[gray > t].mean()) / 2
        if abs(t_new - t) < tol:
            return t_new
        t = t_new
```

It converges to 114 here, almost the same as Otsu. That is no accident: a stationary point of Otsu's criterion satisfies (approximately) $t = (\mu_0 + \mu_1)/2$, so the two methods usually land in the same place. Ridler–Calvard is just a cheap way to find it. In my original script I used the `mahotas` implementations of both (`otsu` and `rc`) and got near-identical results too.

## A Gaussian mixture: a probabilistic threshold

Otsu and Ridler–Calvard implicitly treat the two classes as equally likely. A **two-component Gaussian mixture** models the histogram as

$$
p(v) = \pi_0\,\mathcal{N}(v;\,\mu_0, \sigma_0^2) + \pi_1\,\mathcal{N}(v;\,\mu_1, \sigma_1^2),
$$

where $\pi_k$ are the class proportions and $\mu_k, \sigma_k$ the class means and standard deviations. The parameters are fitted with the expectation–maximisation (EM) algorithm. A natural threshold is where the posterior probability of the bright class crosses 0.5, which is the Bayes decision boundary:

```python
from sklearn.mixture import GaussianMixture

x = gray.reshape(-1, 1).astype(float)             # one sample per PIXEL
gmm = GaussianMixture(n_components=2, random_state=0).fit(x)
levels = np.arange(256).reshape(-1, 1)
bright = np.argmax(gmm.means_.ravel())
t_gmm = int(np.argmax(gmm.predict_proba(levels)[:, bright] > 0.5))
```

The fit recovers means of 69.5 and 159.1 and weights 0.82 and 0.18, very close to the truth. The threshold is 121: higher than Otsu, because the background is much more common, so a pixel in the gap is more likely to be background. Unlike Otsu, the mixture also gives a probability per pixel and lets each class have its own variance.

I used this idea on bright, snowy images to split pixels into a "snow" mode and an "everything else" mode. One mistake is worth flagging. An early draft fitted the mixture to the **histogram counts** (the 256 bin heights) instead of to the pixel values. That fits a distribution to the heights of the bars, which has nothing to do with grey levels. Fit on the pixels themselves (or on bin centres weighted by counts).

![Left: the synthetic image. Middle: its histogram with the two fitted Gaussians and the thresholds from the mean (86), Otsu (113), Ridler–Calvard (114) and the mixture (121). Right: Otsu masks without and with a 5×5 Gaussian blur](fig-thresholds.png)

## Blur first

Every method above treats pixels independently, so noise near the threshold turns into salt-and-pepper speckle. A small Gaussian blur before thresholding averages each pixel with its neighbours, which narrows both histogram peaks and deepens the valley between them. Here a 5×5 blur barely changes the threshold (112) but raises the accuracy from 99.3% to 99.9%, and the speckle disappears (right panel). See [linear filters and the Gaussian blur](/notes/linear-filters-and-gaussian-blur/).

## When one threshold is not enough: adaptive thresholding

A global threshold assumes that "bright" means the same thing everywhere. Add a lighting ramp (left side darker, right side brighter by 90 grey levels in total) and that assumption breaks: the dark discs on the left and the bright background on the right now overlap in grey level. Global Otsu falls to 86.7% accuracy and the right edge fills with noise.

**Adaptive thresholding** computes a separate threshold for every pixel from its neighbourhood: $T(x, y) = \text{local mean}(x, y) - C$, where the local mean is a box or Gaussian average over a $b \times b$ window and $C$ is a small offset.

```python
import cv2
blur = cv2.GaussianBlur(uneven, (5, 5), 0)
mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 51, -10)   # block size 51, C = -10
```

![Uneven lighting: the input, the global Otsu mask that fails on the bright side, and the adaptive mask that follows the lighting but leaves holes in the largest discs](fig-adaptive.png)

The adaptive mask follows the lighting and reaches 96.4%. It also shows the method's main weakness: inside a disc larger than the window, the local mean is the disc itself, so the centre is compared against its own brightness and breaks up. The window must be clearly larger than the objects and smaller than the scale over which the lighting changes. A negative $C$ (foreground must be at least 10 levels above the local mean) keeps flat background from being labelled foreground. Other ways to handle uneven lighting are to remove the illumination first (divide by a heavily blurred copy, or apply [CLAHE](/notes/contrast-enhancement-he-clahe-gamma/)) and then use a global method.

## Key takeaways

- Thresholding at the mean is biased towards the larger class; use a histogram-based method instead.
- Otsu maximises the between-class variance; Ridler–Calvard iterates $t = (m_0 + m_1)/2$ and usually lands at the same place.
- A two-component Gaussian mixture, fitted on pixel values, gives a Bayes threshold that accounts for unequal class sizes and variances, plus a per-pixel probability.
- Blur before thresholding to remove speckle.
- Use adaptive thresholds for uneven lighting, with a window larger than the objects.

## Resources

- Otsu, N. (1979). *A Threshold Selection Method from Gray-Level Histograms*. IEEE Trans. SMC 9(1). https://doi.org/10.1109/TSMC.1979.4310076
- Ridler, T. W., Calvard, S. (1978). *Picture Thresholding Using an Iterative Selection Method*. IEEE Trans. SMC 8(8). https://doi.org/10.1109/TSMC.1978.4310039
- Gonzalez, R. C., Woods, R. E. *Digital Image Processing*, 4th ed., Chapter 10 (thresholding, Otsu's method, variable thresholding).
- Bishop, C. M. *Pattern Recognition and Machine Learning*, Section 9.2 (mixtures of Gaussians and EM).
- OpenCV tutorial, *Image Thresholding*: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- Rosebrock, A. *Practical Python and OpenCV* (PyImageSearch), thresholding chapter: the style of exercise where I first practised adaptive, Otsu and Ridler–Calvard thresholds with OpenCV and `mahotas`.
