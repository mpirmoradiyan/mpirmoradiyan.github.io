---
title: "Colour transfer and histogram matching: making one image look like another"
slug: colour-transfer-and-histogram-matching
date: 2023-04-27
topic: Signal & Image Processing
tags: [histogram-matching, colour-transfer, reinhard, lab, normalisation, domain-shift]
summary: "How histogram matching and Reinhard's mean-and-variance colour transfer move an image's colour statistics towards a reference, why the colour space matters, and why no-data pixels must be left out of the statistics."
sources:
  - title: "Color Transfer between Images"
    author: "Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley (IEEE CG&A, 2001)"
    url: "https://doi.org/10.1109/38.946629"
    type: paper
  - title: "Digital Image Processing, 4th ed., Section 3.3 (Histogram processing: histogram matching)"
    author: "Rafael C. Gonzalez, Richard E. Woods"
    type: book
  - title: "scikit-image documentation: exposure.match_histograms"
    author: "scikit-image"
    url: "https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.match_histograms"
    type: docs


status: draft
---

Two photos of the same kind of scene rarely have the same colours. The light changes, the camera's white balance changes, the exposure changes, and a season can turn green ground white. A model trained on one set of images then sees the other set as a different domain. A cheap first fix is to move the colour statistics of an image towards those of a reference image. This note covers the two classic ways to do it: **histogram matching**, which copies the full distribution of each channel, and **Reinhard colour transfer**, which copies only the mean and the standard deviation. I tried both while looking for a way to make difficult images (for example snow-covered scenes) look more like ordinary ones.

## Histogram matching: copy the whole distribution

Treat the pixel values of one channel as samples of a random variable. Its cumulative distribution function (CDF) $F_s(v)$ says what fraction of source pixels have a value at most $v$. The reference has its own CDF $F_r$. Histogram matching maps each source value to the reference value that sits at the same quantile:

$$
v' = F_r^{-1}\big(F_s(v)\big).
$$

A source pixel at the 30th percentile becomes the reference's 30th-percentile value. Histogram equalisation is the special case where the reference CDF is a straight line (a flat histogram); see [contrast enhancement](/notes/contrast-enhancement-he-clahe-gamma/).

The mapping is monotonic, so the order of brightness inside the channel is kept: if pixel A was brighter than pixel B, it still is. Only the spacing between levels changes. Here is the whole algorithm for one channel:

```python
import numpy as np

def match_channel(src, ref):
    # unique levels, the index of each pixel's level, and level counts
    s_vals, s_idx, s_counts = np.unique(src.ravel(), return_inverse=True, return_counts=True)
    r_vals, r_counts = np.unique(ref.ravel(), return_counts=True)
    s_cdf = np.cumsum(s_counts) / src.size      # quantile of each source level
    r_cdf = np.cumsum(r_counts) / ref.size
    mapped = np.interp(s_cdf, r_cdf, r_vals)    # invert the reference CDF
    return mapped[s_idx].reshape(src.shape)

matched = np.stack([match_channel(src[..., c], ref[..., c]) for c in range(3)], -1)
```

This gives the same result as `skimage.exposure.match_histograms(src, ref, channel_axis=-1)` to within one grey level (the difference is rounding in the interpolation).

**Limits of the per-channel version.** Matching R, G and B separately makes each marginal histogram right, but it ignores how the channels move together. A colour that never occurs in the reference can be created from three individually "correct" channels. Matching is also aggressive: it stretches sparse parts of the histogram, which amplifies noise and can produce banding in smooth areas. You can see the extra grain in the third panel of the figure below.

## Reinhard transfer: copy the mean and the spread

Reinhard et al. (2001) proposed something much simpler. For each channel, standardise the source and then give it the reference's mean and standard deviation:

$$
v' = \frac{\sigma_r}{\sigma_s}\,(v - \mu_s) + \mu_r,
$$

where $\mu_s, \sigma_s$ are the source channel's mean and standard deviation and $\mu_r, \sigma_r$ are the reference's. It is a linear map per channel, so it cannot create banding and it changes noise only by the factor $\sigma_r/\sigma_s$.

The important detail in the paper is the **colour space**. In RGB the three channels are strongly correlated: a brighter pixel is usually brighter in all three. Changing one channel on its own then also changes the colour in a way that is hard to control. Reinhard et al. worked in $l\alpha\beta$, a space by Ruderman et al. designed to decorrelate the cone responses of natural images, with one achromatic axis and two opponent colour axes (yellow–blue and red–green). In that space you can treat the three channels independently with much less cross-talk. In practice CIELAB is a common substitute: it also separates lightness from two opponent colour axes. See [colour spaces for image processing](/notes/colour-spaces-for-image-processing/).

```python
import cv2, numpy as np

def reinhard_lab(src, ref, mask_src=None, mask_ref=None):
    # src, ref: RGB uint8. Masks: True where a pixel is valid.
    s = cv2.cvtColor(src.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    r = cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    ms = np.ones(s.shape[:2], bool) if mask_src is None else mask_src
    mr = np.ones(r.shape[:2], bool) if mask_ref is None else mask_ref
    mu_s, sd_s = s[ms].mean(0), s[ms].std(0)
    mu_r, sd_r = r[mr].mean(0), r[mr].std(0)
    sd_s = np.where(sd_s < 1e-6, 1.0, sd_s)        # guard a flat channel
    out = (s - mu_s) * (sd_r / sd_s) + mu_r
    out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB) * 255
    return np.clip(out, 0, 255).astype(np.uint8)
```

Converting float images (values in 0–1) to Lab gives the real CIELAB ranges ($L^*$ from 0 to 100, $a^*$ and $b^*$ roughly −128 to 127). Doing the arithmetic on the 8-bit Lab encoding also works, but every value is rounded twice.

My first version did the transfer directly on the BGR channels. That is a reasonable white-balance-like correction when the difference is mostly a global colour cast, because scaling each channel separately is the classic von Kries model of illumination change. It is less safe when the two images have different content, because the correlated channels are pushed independently and hues can drift.

![A dark, warm-tinted synthetic scene, a neutral reference, and three corrected versions: histogram matching, per-channel RGB Reinhard and Lab Reinhard](fig-colour-transfer.png)

On this synthetic example all three methods bring the mean colour close to the reference (mean RGB about 129, 149, 146 for the reference and within about 2 levels for each method). They differ in the details: histogram matching adds visible grain, while the two Reinhard versions are smooth and differ mainly in how saturated the vegetation becomes.

## Pitfall: statistics must come from valid pixels only

Many real images contain pixels that are not part of the scene: black borders after a warp, areas outside a mask, no-data regions in a raster. In my snow experiment the input was an image where one class of pixels had been masked out and set to black, and I computed the mean and standard deviation over the whole image. The black pixels pull the mean down and inflate the spread, so the transfer over-brightens and washes out the real pixels.

![Reinhard transfer on an image with a black no-data region: statistics that include the black pixels give a washed-out result, statistics on valid pixels only give the expected colours](fig-nodata-pitfall.png)

The numbers make the point. With 40% of the image set to black, the mean of the valid pixels after transfer was about (164, 181, 147) when the black pixels were included in the statistics, against (130, 148, 145) when they were excluded; the reference mean is (129, 149, 146). The same rule applies to histogram matching: build both CDFs from valid pixels only, and leave the no-data pixels untouched in the output.

Two smaller guards are worth keeping:

- **Flat channels.** If a channel has zero spread (a constant image, or a fully masked tile), $\sigma_s = 0$ and the formula divides by zero. Replace it by 1 or skip the channel.
- **Clipping.** The linear map can push values outside the valid range. Clip after converting back, and expect some loss in highlights if the reference is much brighter.

## When it helps, and when it does not

Colour transfer is useful when the difference between two sets of images is mostly global: a camera with different white balance, a different time of day, a different exposure. It is a fast preprocessing step to reduce that kind of domain shift before training or inference, and it can also be used the other way round, as an augmentation that pushes training images towards many references (see [photometric colour augmentation](/notes/photometric-colour-augmentation/)).

It does not help when the content itself is different. A snowy forest and a summer forest differ because of what is on the ground, not because of the light. Forcing the snowy image to have summer statistics turns the snow into an odd grey-green and changes the very evidence a model needs. Global methods also cannot fix effects that vary across the image, such as haze that grows with distance (see [haze and the dark channel prior](/notes/haze-model-and-dark-channel-prior/)) or a vignetted corner. For those you need a local or physical model.

## Key takeaways

- Histogram matching maps each value through $F_r^{-1}(F_s(v))$; it copies the full per-channel distribution but can amplify noise and ignores the joint colour distribution.
- Reinhard transfer copies only mean and standard deviation per channel; it is smooth and robust, and works best in a decorrelated, opponent space such as $l\alpha\beta$ or CIELAB.
- Compute all statistics on valid pixels only; black or no-data pixels badly bias the result.
- Guard against zero standard deviation and clip after converting back.
- Use it for global acquisition differences, not for real changes in scene content.

## Resources

- Reinhard, E., Ashikhmin, M., Gooch, B., Shirley, P. (2001). *Color Transfer between Images*. IEEE Computer Graphics and Applications 21(5). https://doi.org/10.1109/38.946629
- Ruderman, D. L., Cronin, T. W., Chiao, C.-C. (1998). *Statistics of cone responses to natural images: implications for visual coding*. JOSA A 15(8). (The $l\alpha\beta$ space.)
- Gonzalez, R. C., Woods, R. E. *Digital Image Processing*, 4th ed., Section 3.3: histogram equalisation and histogram matching.
- scikit-image documentation, `exposure.match_histograms`: https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.match_histograms
