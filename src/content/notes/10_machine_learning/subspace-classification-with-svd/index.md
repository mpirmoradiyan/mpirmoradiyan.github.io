---
title: "Classifying by distance to a subspace: a tiny SVD digit classifier"
slug: subspace-classification-with-svd
date: 2020-03-11
topic: Machine Learning
tags: [university-lab, svd, linear-algebra, orthogonal-projection, classification, numpy]
summary: "How to classify a vector by projecting it onto a low-dimensional subspace learned for each class with the SVD, and how to compute that projection correctly."
sources:
  - title: "732A74 Introduction to Python (course lab)"
    author: "Linköping University"
    type: course
  - title: "Matrix Methods in Data Mining and Pattern Recognition, Chapter 10 (Classification of handwritten digits)"
    author: "Lars Eldén"
    type: book
  - title: "Linear Algebra and Learning from Data, Part I (SVD and low-rank approximation)"
    author: "Gilbert Strang"
    type: book


status: draft
---

This is one of my favourite small ideas from the Python course, even though our lab implementation (a pair lab) had several bugs. The idea: all images of the digit 4 look roughly alike, so as vectors they lie close to some low-dimensional subspace. Learn one such subspace per digit with the SVD. To classify a new image, measure how far it is from each subspace and pick the nearest. No training loop, no gradient descent, just linear algebra.

## Images as vectors, classes as subspaces

The scikit-learn digits dataset has 8×8 grayscale images, so each image is a vector $x \in \mathbb{R}^{64}$. Stack all training images of one digit $d$ as rows of a matrix $A_d$ (one row per image, 64 columns).

The thin SVD writes

$$
A_d = U_d \Sigma_d V_d^\top,
$$

where the rows of $V_d^\top$ are orthonormal vectors in $\mathbb{R}^{64}$ (the right singular vectors), $\Sigma_d$ is diagonal with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$, and $U_d$ holds the coordinates of each image in that basis. The first few rows of $V_d^\top$ are the directions that explain the most of the class's images: the best rank-$k$ approximation of $A_d$ (in the 2-norm and Frobenius norm) uses exactly the top $k$ singular vectors (Eckart–Young theorem).

So take $V_{d,k}$ = the first $k$ rows of $V_d^\top$, a $k \times 64$ matrix with orthonormal rows. Its row space is the "digit $d$ subspace".

Note this is not quite PCA: we don't subtract the class mean first, so the first singular vector mostly points towards the average image of that digit. For this classifier that's fine, and arguably useful.

## Projection and distance

Because the rows of $V_{d,k}$ are orthonormal, projecting $x$ onto their span is easy:

$$
c = V_{d,k}\, x \quad (\text{coordinates, } k \times 1), \qquad \hat x_d = V_{d,k}^\top c = V_{d,k}^\top V_{d,k}\, x.
$$

The distance from $x$ to the subspace is the length of what's left over:

$$
\text{dist}_d(x) = \lVert x - \hat x_d \rVert_2.
$$

Predict the digit with the smallest distance: $\hat y = \arg\min_d \text{dist}_d(x)$.

The general projection formula is $\hat x = R^\top (R R^\top)^{-1} R x$ for any matrix $R$ whose rows span the subspace. With orthonormal rows, $R R^\top = I$ and the inverse disappears. In our lab code we wrote the coefficients as `R.dot(x) / R.dot(R.T)`, an element-wise division of a vector by a matrix. That's not a matrix inverse: it broadcasts into a $k \times k$ array, and since the off-diagonal entries of $RR^\top$ are tiny rounding noise rather than exact zeros, you divide by numbers near zero. The correct version is just `R @ x`, or `np.linalg.solve(R @ R.T, R @ x)` if the rows weren't orthonormal.

## The whole classifier

```python
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X_train, y_train, X_test, y_test = X[:600], y[:600], X[600:], y[600:]

def fit_subspaces(X, y, k):
    bases = {}
    for label in np.unique(y):
        A = X[y == label]                          # one row per training image
        _, s, Vt = np.linalg.svd(A, full_matrices=False)
        r = min(k, int(np.sum(s > 1e-10 * s[0])))  # don't keep more than the rank
        bases[label] = Vt[:r]                      # orthonormal rows spanning the class subspace
    return bases

def classify(bases, x):
    dists = {lab: np.linalg.norm(x - V.T @ (V @ x)) for lab, V in bases.items()}
    return min(dists, key=dists.get)

for k in [1, 3, 10, 30]:
    bases = fit_subspaces(X_train, y_train, k)
    acc = np.mean([classify(bases, x) == t for x, t in zip(X_test, y_test)])
    print(f"k = {k:2d}: test accuracy {acc:.3f}")
```

With this split, test accuracy goes from about 0.85 at $k = 1$ to about 0.96 at $k = 10$, then drops a little at $k = 30$. That shape makes sense: too small a $k$ and the subspace can't describe the different ways people write a digit; too large and every subspace gets big enough to fit other digits too, so the distances stop being informative. $k$ is a hyperparameter you'd normally pick with a validation set.

## Mistakes we made, and the right idea

Besides the projection formula, our lab version mixed up what $k$ means. We kept only the *first* singular vector per class and then used $k$ to choose the $k$ training *images* closest to it. But $k$ should be the number of singular vectors (the dimension of the subspace); the subspace itself is the thing you store, not a few hand-picked images.

Two smaller bugs are classic:

- Our hit-rate function compared each predicted label with the loop index `i` instead of the true label `labels[i]`.
- One evaluation measured accuracy on the training rows themselves, which tells you how well the subspaces fit the data they came from, not how well they generalize.

And the class constructor ignored its own arguments and read global variables, so two "independent" classifiers would actually share data. Encapsulation only helps if the methods use `self` and the arguments.

## Key takeaways

- One subspace per class, from the top $k$ right singular vectors of that class's data matrix.
- With orthonormal basis rows $V$, the projection of $x$ is $V^\top V x$; the distance is $\lVert x - V^\top V x\rVert$.
- Never "divide by a matrix" element-wise; use the orthonormality or `np.linalg.solve`.
- $k$ is the subspace dimension, a hyperparameter with a sweet spot. Choose it on held-out data.
- Evaluate on data the model hasn't seen, against the true labels.

## Resources

- 732A74 Introduction to Python, Linköping University, lab 5 (2020).
- Lars Eldén, *Matrix Methods in Data Mining and Pattern Recognition* (SIAM), Chapter 10: classification of handwritten digits with SVD bases.
- Gilbert Strang, *Linear Algebra and Learning from Data*, Part I: SVD, Eckart–Young, low-rank approximation.
- scikit-learn, [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
