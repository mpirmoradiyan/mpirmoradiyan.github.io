---
title: "Condition numbers: why solving the normal equations can fail"
slug: condition-number-and-ill-conditioned-regression
date: 2020-01-26
topic: Computational Statistics
tags: [university-lab, linear-algebra, condition-number, least-squares, multicollinearity, numerical-stability]
summary: "What the condition number of a matrix measures, why highly correlated features make X'X numerically singular, and why scaling, QR and ridge help."
sources:
  - title: "732A90 Computational Statistics (course lab)"
    author: "Linköping University"
    type: course
  - title: "Numerical Linear Algebra, Lectures 11-12 and 18-19"
    author: "Lloyd N. Trefethen, David Bau III"
    type: book
  - title: "Matrix Computations (4th ed.), Chapter 5"
    author: "Gene H. Golub, Charles F. Van Loan"
    type: book
  - title: "R documentation: kappa()"
    author: "R Core Team"
    url: "https://stat.ethz.ch/R-manual/R-devel/library/base/html/kappa.html"
    type: docs


status: draft
---

Linear regression has a neat closed form, $\hat\beta = (X^\top X)^{-1} X^\top y$, and it's tempting to just type it in. In our first computational statistics lab we did exactly that on a spectroscopy data set (100 absorbance channels of meat samples, which move almost in lockstep), and R refused: "system is computationally singular". The number that explains this is the condition number.

## What the condition number measures

Take a linear system $A x = b$, where $A$ is a square matrix, $b$ is the known right-hand side and $x$ is the unknown. Now nudge $b$ a little, to $b + \delta b$. The solution moves to $x + \delta x$. The question is: how big can the relative change in $x$ be compared to the relative change in $b$? The answer is

$$
\frac{\|\delta x\|}{\|x\|} \le \kappa(A)\, \frac{\|\delta b\|}{\|b\|}, \qquad \kappa(A) = \|A\|\, \|A^{-1}\|,
$$

where $\|\cdot\|$ is a vector norm and its matching matrix norm, and $\kappa(A)$ is the **condition number**. In the 2-norm, $\kappa(A) = \sigma_{\max} / \sigma_{\min}$, the ratio of the largest to the smallest singular value.

The way I picture it: $A$ stretches space more in some directions than others. If one direction is squashed almost flat ($\sigma_{\min}$ tiny), then undoing $A$ has to blow that direction back up, and any noise in that direction gets blown up with it.

Rounding errors act like a tiny $\delta b$ (and $\delta A$) of relative size about $\varepsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$. So a rough rule is: **you lose about $\log_{10} \kappa$ digits of accuracy.** With $\kappa \approx 10^{16}$ there's nothing left, and the matrix is "numerically singular" even if it isn't exactly singular on paper.

## Why collinear features ruin $X^\top X$

If two columns of $X$ are almost the same, $X$ has a direction (their difference) that it maps to almost zero. So $\sigma_{\min}(X)$ is tiny and $\kappa(X)$ is large.

The normal equations make this worse, because

$$
\kappa(X^\top X) = \kappa(X)^2,
$$

where $X$ is the $n \times p$ design matrix (rows are samples, columns are features). Forming $X^\top X$ squares the condition number, so it doubles the number of digits you lose.

In the lab, `kappa()` of $X^\top X$ came out at about $1.16 \times 10^{15}$, and `solve()` reported a reciprocal condition number of $7.1 \times 10^{-17}$. After standardising every column with `scale()`, the condition number dropped to about $4.9 \times 10^{11}$. Better, and `solve()` now ran, but the coefficients it gave were in the hundreds with alternating signs, which is the classic look of an ill-conditioned fit.

In our lab report we wrote that the system "does not have an answer as the matrix is singular" and "not invertible". That's a bit too strong. With 215 samples and around 100 columns, $X^\top X$ is very likely invertible in exact arithmetic. The point is that it's so badly conditioned that, in double precision, the inverse can't be computed reliably. "Numerically singular" or "ill-conditioned" is the accurate wording.

## What helps

- **Scaling** puts all columns on the same scale. It removes the part of the ill-conditioning that comes from units (one column in grams, another in thousands), but not the part that comes from correlation. That's why it helped by a few orders of magnitude and no more.
- **QR decomposition** solves least squares from $X$ directly ($X = QR$, then solve $R \beta = Q^\top y$), so you only pay $\kappa(X)$, not $\kappa(X)^2$. This is what `lm()` does under the hood.
- **Ridge regression** adds $\lambda I$ to $X^\top X$. Every eigenvalue gets at least $\lambda$, so $\kappa$ drops to at most $(\sigma_{\max}^2 + \lambda)/\lambda$. You trade a little bias for a lot of stability.
- **Fewer, or combined, features** such as PCA or PLS, which is the usual route for spectra.

```r
set.seed(1)
n <- 200
w <- seq(0, 1, length.out = 50)                 # 50 "wavelengths"
base <- rnorm(n)
# 50 smooth, almost identical columns: strongly collinear
X <- sapply(w, function(k) 3 + base * (1 + k) + rnorm(n, sd = 1e-3))
y <- X[, 1] - X[, 2] + rnorm(n)

kappa(X, exact = TRUE)                          # about 4.5e4
kappa(crossprod(X), exact = TRUE)               # about 2.0e9: the square
Xs <- scale(X)
kappa(crossprod(Xs), exact = TRUE)              # about 4.2e8: scaling helps a bit

b_qr <- qr.coef(qr(X), y)                       # least squares via QR on X itself
lambda <- 1
A_ridge <- crossprod(Xs) + lambda * diag(ncol(Xs))
kappa(A_ridge, exact = TRUE)                    # about 1e4: ridge fixes the conditioning
```

The synthetic numbers are smaller than in the lab, but the pattern is the same: squaring when forming $X^\top X$, a modest gain from scaling, a big gain from regularisation.

For how QR is used to fit a regression and get standard errors, see [least squares by hand](/notes/least-squares-qr-and-standardized-residuals/). For where the rounding errors come from in the first place, see [floating-point traps](/notes/floating-point-rounding-and-cancellation/).

## Key takeaways

- $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ bounds how much relative input error can grow in the solution. Expect to lose about $\log_{10}\kappa$ digits.
- Correlated features make $X$ ill-conditioned, and $X^\top X$ squares the problem.
- "Computationally singular" means "too ill-conditioned for double precision", not necessarily "singular".
- Scaling fixes units, not correlation. Use QR for plain least squares and ridge, PCA or PLS when features are strongly collinear.

## Resources

- 732A90 Computational Statistics, Linköping University, Lab 1 (2020).
- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, SIAM, 1997, Lecture 11 (least squares problems), Lecture 12 (conditioning and condition numbers) and Lectures 18-19 (conditioning and stability of least squares).
- Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press, 2013, Chapter 5.
- Arthur E. Hoerl and Robert W. Kennard, "Ridge regression: biased estimation for nonorthogonal problems", *Technometrics* 12(1), 1970.
- R documentation, `kappa()`: https://stat.ethz.ch/R-manual/R-devel/library/base/html/kappa.html
