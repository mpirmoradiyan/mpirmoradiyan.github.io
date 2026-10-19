---
title: "PCA vs ICA: uncorrelated is not the same as independent"
slug: pca-vs-ica
date: 2019-12-08
topic: Machine Learning
tags: [university-lab, pca, ica, dimensionality-reduction, whitening, blind-source-separation]
summary: "What PCA and ICA each look for, why ICA starts by running PCA, and why the two can give similar-looking loadings without meaning the same thing."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Independent component analysis: algorithms and applications"
    author: "Aapo Hyvärinen, Erkki Oja"
    url: "https://doi.org/10.1016/S0893-6080(00)00026-5"
    type: paper
  - title: "The Elements of Statistical Learning, Sections 14.5 and 14.7"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book
  - title: "fastICA (R package)"
    author: "J. L. Marchini, C. Heaton, B. D. Ripley"
    url: "https://cran.r-project.org/package=fastICA"
    type: docs


status: draft
---

PCA and ICA both turn a set of measured variables into a few new ones, built as linear combinations of the old ones. In a lab on near-infrared spectra I ran both and got loading plots that looked almost the same, and I concluded they "convey the same information". That's not really right. The similarity came from how ICA is computed, not from the two methods wanting the same thing.

## What PCA looks for

PCA finds orthogonal directions of maximum variance. The first component is the unit vector $w_1$ that maximizes $\operatorname{Var}(w_1^\top x)$; the next one does the same while staying orthogonal to the first, and so on. In practice these are the eigenvectors of the covariance matrix $\Sigma$:

$$
\Sigma = U \Lambda U^\top ,
$$

where the columns of $U$ are the loadings and the diagonal of $\Lambda$ holds the variance along each component. The scores $U^\top (x - \bar x)$ are **uncorrelated**. That's all PCA promises. It only uses means and covariances (second-order statistics).

## What ICA looks for

ICA assumes the data are a mix of hidden sources: $x = A s$, where the entries of $s$ are statistically **independent** and $A$ is an unknown mixing matrix. The goal is an unmixing matrix $W$ with $\hat s = W x$. The classic picture is two microphones in a room with two speakers, where each microphone records a different blend.

Independence is much stronger than "uncorrelated". It says the joint density factorizes, $p(s_1, s_2) = p(s_1)\,p(s_2)$, which involves all higher moments, not just covariances. And there's a catch: if the sources are Gaussian, ICA can't work, because any rotation of independent Gaussians is still independent Gaussians. ICA needs non-Gaussian sources, and it finds them by searching for the least Gaussian directions (fastICA uses an approximation of negentropy; kurtosis is a simpler measure).

## Why ICA starts with PCA

Most ICA algorithms first **whiten** the data: center it, rotate to the principal axes, and scale each axis to unit variance,

$$
z = \Lambda^{-1/2} U^\top (x - \bar x), \qquad \operatorname{Cov}(z) = I .
$$

After whitening, the remaining unmixing step is just a **rotation**. Here's the key point: every rotation of $z$ is still white (uncorrelated with unit variance), so second-order statistics can't tell those rotations apart. PCA stops here, at one arbitrary rotation. ICA keeps going and picks the rotation that makes the components as non-Gaussian as possible.

```r
set.seed(2)
n <- 2000
S <- cbind(runif(n, -1, 1), sign(rnorm(n)) * rexp(n))   # two independent, non-Gaussian sources
A <- matrix(c(1, 1, 0.3, 1), 2)                         # mixing matrix
X <- S %*% t(A)                                         # what we observe

# Step 1, PCA + whitening
Xc <- scale(X, scale = FALSE)
e  <- eigen(cov(Xc))
Z  <- Xc %*% e$vectors %*% diag(1 / sqrt(e$values))

# Step 2, ICA: among all rotations of Z, pick the least Gaussian one
kurt  <- function(v) mean(v^4) / mean(v^2)^2 - 3
rot   <- function(t) matrix(c(cos(t), sin(t), -sin(t), cos(t)), 2)
theta <- seq(0, pi / 2, length.out = 181)
score <- sapply(theta, function(t) sum(abs(apply(Z %*% rot(t), 2, kurt))))
Y <- Z %*% rot(theta[which.max(score)])

round(cor(Z, S), 2)   # whitened PCs: each still a mix of both sources (about 0.5 and 0.9)
round(cor(Y, S), 2)   # ICA components: each matches one source (about 1.00), up to sign
```

## The mistake in my lab report

In R's `fastICA`, the returned matrix `K` is the pre-whitening step: it projects the centered data onto the first `n.comp` principal components and scales them. `W` is the rotation found in that whitened space. So the columns of `K %*% W`, which I plotted as "ICA loadings", are by construction combinations of the first two PCA loadings. With only two components kept, ICA can only rotate inside the plane spanned by PC1 and PC2. That's why the plots looked like the PCA ones.

In the report I wrote that the independence maximization "cannot be obtained unless in the directions of the two principal components". That's not right. ICA was *restricted* to the PC subspace because I asked for two components after whitening. Inside that plane it can pick any rotation, and the independent directions are generally not the principal axes. Similar-looking loadings would have been a sign that the rotation was small or that the spectra didn't contain clearly non-Gaussian sources, not that PCA and ICA are equivalent.

## When to use which

- **PCA**: compression, noise reduction, removing collinearity before regression, a quick look at the main directions of variation. Components are ordered by variance.
- **ICA**: separating mixed signals (audio, EEG, spectra of mixtures) when you believe there are independent, non-Gaussian sources. Components have no natural order, sign or scale.

## Key takeaways

- PCA gives uncorrelated components ordered by variance, using only covariances.
- ICA looks for independent components and needs non-Gaussian sources.
- ICA usually whitens with PCA first, then finds the best rotation; any rotation of white data is still white.
- Keeping $k$ components in fastICA restricts ICA to the first $k$ principal components, so similar loadings aren't evidence that the methods agree.

## Resources

- 732A99 Machine Learning, Linköping University (course lab, autumn 2019)
- Aapo Hyvärinen and Erkki Oja, "Independent component analysis: algorithms and applications", *Neural Networks* 13 (2000): https://doi.org/10.1016/S0893-6080(00)00026-5
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 14.5 (principal components) and Section 14.7 (ICA): https://hastie.su.domains/ElemStatLearn/
- R package `fastICA` documentation (meaning of `K`, `W`, `A`): https://cran.r-project.org/package=fastICA
