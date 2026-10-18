---
title: "Statistical models of natural images: from pixel histograms to the 1/f power spectrum"
slug: statistical-models-of-natural-images
date: 2024-07-04
topic: Signal & Image Processing
tags: [image-statistics, power-spectrum, gaussian-model, generative-models, fourier]
summary: "Why treating pixels as independent fails, how neighbouring pixels are correlated, and how the 1/f power spectrum gives a simple Gaussian model of natural images."
sources:
  - title: "Foundations of Computer Vision, Chapter 27: Statistical Image Models"
    author: "Antonio Torralba, Phillip Isola, William T. Freeman"
    url: "https://visionbook.mit.edu/stat_image_models_revised.html"
    type: book
  - title: "Statistics of natural images: scaling in the woods"
    author: "Daniel L. Ruderman and William Bialek"
    url: "https://doi.org/10.1103/PhysRevLett.73.814"
    type: paper


status: draft
---

Natural images are a tiny, very structured subset of all possible pixel arrays. A random array of pixels almost never looks like a photo. A **statistical image model** tries to describe that subset with a probability distribution $p(\ell)$. A good model lets us generate images, remove noise, fill in missing parts and tell texture from noise. Modern generative models are the far end of this road; this note covers the first, simplest steps, following Chapter 27 of *Foundations of Computer Vision*.

## Model 1: independent pixels

The simplest model says each pixel is drawn independently from the same distribution. For an 8-bit grey image the distribution has 256 parameters:

$$
p_\theta(\ell[n, m] = k) = \theta_k, \qquad k = 0, \dots, 255, \qquad \sum_k \theta_k = 1.
$$

The maximum-likelihood estimate of $\theta_k$ is just the normalised histogram: the fraction of pixels with value $k$. Sampling a new image means drawing every pixel from that histogram.

```python
import numpy as np

def fit_histogram(img):                       # img: uint8 array
    return np.bincount(img.ravel(), minlength=256) / img.size

def sample_independent(theta, shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(256, size=shape, p=theta).astype(np.uint8)
```

![An image, its grey-level histogram, and a new image sampled with every pixel independent](fig-independent-pixels.png)

The sample has the right overall brightness and nothing else. I also tried averaging the histograms of seven different photos into one model; the samples were the same kind of noise. That is no surprise. The histogram does not care where pixels are, so shuffling all the pixels of a photo would give it exactly the same likelihood. The book notes that this model fits only a few kinds of images, such as a star field. Histograms are still useful features, but they are not a model of images.

## Pixels are not independent

Pick a pixel and look at another pixel a distance $d$ away. When $d$ is small they almost always have similar values; as $d$ grows the link weakens.

![Left: correlation between pixel pairs against their horizontal distance for two images. Right: radially averaged power spectra on log-log axes](fig-correlation-and-spectrum.png)

The left panel measures this over whole images: the correlation is about 0.98 for neighbours and falls slowly with distance. (My first attempt compared one pixel with a handful of others along a diagonal; the result was noisy, and it also hit an overflow because subtracting two `uint8` values wraps around. Convert to float before subtracting.)

## The power spectrum of natural images

Correlation as a function of distance and the power spectrum carry the same information: by the Wiener–Khinchin theorem, the power spectrum is the Fourier transform of the autocorrelation. The power spectrum is easier to read. For most natural images, the Fourier magnitude falls off roughly as a power of the radial frequency $w = \sqrt{u^2 + v^2}$:

$$
|\mathcal{L}(u, v)| \approx \frac{A}{w^{\alpha}}, \qquad \text{so the power } |\mathcal{L}|^2 \approx \frac{A^2}{w^{2\alpha}},
$$

with $\alpha$ close to 1 for typical scenes. On a log-log plot that is a straight line, and the right panel shows exactly that. For my two test images the fitted power slopes were about $-2.8$, so $\alpha \approx 1.4$. Values between 1 and 1.5 are common, depending on the scene and on how the spectrum is measured. A spectrum like this has no preferred scale: zooming in on a natural scene gives roughly the same statistics, which is why it is called **scale invariant**.

(In my notebook I first computed "power" as the square of the *log* magnitude. The power spectrum is the square of the magnitude itself; take the log only for display.)

## Model 2: a Gaussian with a 1/f spectrum

Now let the pixels be jointly Gaussian with mean $\mu$ and covariance $C$:

$$
p(\ell) \propto \exp\left(-\tfrac{1}{2}(\ell - \mu)^\top C^{-1}(\ell - \mu)\right),
$$

with the image flattened into a vector. If the statistics are **stationary** (the same everywhere in the image) and the image wraps around at its borders, $C$ is circulant, and every circulant matrix is diagonalised by the Fourier basis:

$$
C = F D F^{H},
$$

where $F$ holds the Fourier basis vectors and $D$ is diagonal. The diagonal entries of $D$ are the power at each frequency, which we just measured: $D \propto 1/w^{2\alpha}$. So the whole model is described by one number, $\alpha$.

Sampling is easy in the Fourier domain: draw independent Gaussian noise at every frequency, scale it by $1/w^{\alpha}$, and transform back. This is the same as filtering white noise.

```python
def sample_gaussian_image(n=256, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    w = np.hypot(fx, fy)
    w[0, 0] = 1.0                                  # avoid dividing by zero at DC
    noise = np.fft.fft2(rng.standard_normal((n, n)))
    return np.fft.ifft2(noise / w**alpha).real     # amplitude falls as 1/w^alpha

img = sample_gaussian_image(alpha=1.5)
print(img.shape)
```

![Samples from the Gaussian model with amplitude spectrum 1/w to the power 0.5, 1 and 1.5](fig-gaussian-samples.png)

The samples look like clouds or rough terrain. They have the right correlations and the right balance of coarse and fine detail, which is much better than independent pixels. But there are no edges, objects or sharp boundaries. A Gaussian model only captures second-order statistics (the power spectrum), and its phases are random. As [the DFT note](/notes/discrete-fourier-transform-of-images/) shows, the phase is where edges and structure live.

I also tried a third, home-made model: estimating a mean and spread for each pixel location from its four neighbours across a few photos, then sampling each pixel independently from its own Gaussian. That only produced a noisy blur of the average image, since it still treats the sampled pixels as independent, and my spread estimate (the standard deviation of *squared* neighbour values) was wrong anyway. Real neighbourhood models, Markov random fields, sample each pixel *conditioned on the values already sampled* around it. That is the subject of the later sections of the chapter.

## Where this leads

The book continues with better models that capture more than the spectrum: the dead-leaves model (images as overlapping opaque shapes, which naturally produces edges and a $1/w$ spectrum), the wavelet marginal model (filter outputs on natural images are sparse, with heavy-tailed, peaky histograms), and non-parametric Markov random fields, which lead to texture synthesis. Deep generative models are the modern version of the same goal.

## Key takeaways

- An independent-pixel model is just the histogram; it ignores all spatial structure and produces noise.
- Nearby pixels are strongly correlated, and the correlation decays slowly with distance.
- Natural images have a power-law spectrum, $|\mathcal{L}| \approx 1/w^{\alpha}$ with $\alpha$ around 1: a straight line on log-log axes.
- A stationary Gaussian model has covariance $C = FDF^H$ with $D \propto 1/w^{2\alpha}$; sampling it means shaping white noise in the Fourier domain.
- Gaussian samples have the right texture of light and shade but no edges, because structure lives in the phase.

## Resources

- Antonio Torralba, Phillip Isola and William T. Freeman, *Foundations of Computer Vision*, MIT Press, 2024, Chapter 27 "Statistical Image Models" (independent pixels, dead leaves, the Gaussian model, wavelet marginals, Markov random fields): <https://visionbook.mit.edu/stat_image_models_revised.html>
- Daniel L. Ruderman and William Bialek, "Statistics of natural images: Scaling in the woods", *Physical Review Letters* 73, 1994: <https://doi.org/10.1103/PhysRevLett.73.814>
- David J. Field, "Relations between the statistics of natural images and the response properties of cortical cells", *Journal of the Optical Society of America A* 4(12), 1987.
- Test images in the figures: `camera` and `astronaut` from scikit-image's `skimage.data`: <https://scikit-image.org/docs/stable/api/skimage.data.html>
