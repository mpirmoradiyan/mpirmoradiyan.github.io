---
title: "Least squares by hand: normal equations, QR, and what standardized residuals really are"
slug: least-squares-qr-and-standardized-residuals
date: 2019-09-28
topic: ML & Statistics
tags: [university-lab, linear-regression, least-squares, qr-decomposition, residual-diagnostics, r]
summary: "How a linear regression is actually computed, why QR is preferred over inverting X'X, how the coefficient standard errors come out, and why standardized residuals need the leverage of each point."
sources:
  - title: "732A94 Advanced R Programming (course lab)"
    author: "Linköping University"
    type: course
  - title: "Linear Models with R, 2nd ed., Chapters 2 (Estimation) and 6 (Diagnostics)"
    author: "Julian J. Faraway"
    type: book
  - title: "Numerical Linear Algebra, Lectures 7, 8 and 11 (QR factorization and least squares)"
    author: "Lloyd N. Trefethen, David Bau III"
    type: book
  - title: "R documentation: influence.measures / rstandard"
    author: "R Core Team"
    url: "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html"
    type: docs


status: draft
---

`lm()` hides a lot. In my Advanced R course we rebuilt it as a small package (pair work): fit the coefficients, compute standard errors, t-values and p-values, and draw the two classic diagnostic plots. Doing it by hand made me see which parts are just algebra and which parts are easy to get subtly wrong. Two things stuck: how you solve the least-squares problem matters numerically, and "standardized residuals" means something more specific than "residuals divided by their standard deviation".

## The model

$$
y = X\beta + \varepsilon,
$$

where $y$ is the $n \times 1$ vector of responses, $X$ the $n \times p$ design matrix (a column of ones for the intercept plus one column per predictor, or per dummy level for a factor), $\beta$ the $p \times 1$ vector of unknown coefficients, and $\varepsilon$ the errors, assumed independent with mean 0 and variance $\sigma^2$. After fitting we get the estimate $\hat\beta$, fitted values $\hat y = X\hat\beta$ and residuals $e = y - \hat y$.

In our vignette I wrote the model as $\hat y = X\beta + \hat e$. That mixes things up: the *observed* $y$ equals $X\beta$ plus error, while the *fitted* $\hat y$ is exactly $X\hat\beta$ with no error term.

## Way 1: normal equations

Least squares minimizes $\lVert y - X\beta \rVert^2$. Setting the gradient to zero gives the normal equations

$$
X^\top X \hat\beta = X^\top y \quad\Rightarrow\quad \hat\beta = (X^\top X)^{-1} X^\top y.
$$

That's what we implemented with `solve(t(X) %*% X)`. It's fine for small, well-behaved problems. The issue is numerical: the condition number (how much rounding errors get amplified) of $X^\top X$ is the *square* of that of $X$. If predictors are nearly collinear, $X$ might have condition number $10^6$, so $X^\top X$ has $10^{12}$, and with double precision (about 16 digits) you've thrown away most of your accuracy.

## Way 2: QR decomposition

Factor $X = QR$, where $Q$ is $n \times p$ with orthonormal columns ($Q^\top Q = I$) and $R$ is $p \times p$ upper triangular. Then

$$
X^\top X \hat\beta = X^\top y \;\Rightarrow\; R^\top Q^\top Q R \hat\beta = R^\top Q^\top y \;\Rightarrow\; R\hat\beta = Q^\top y.
$$

Since $R$ is triangular, you solve it by back-substitution: last coefficient first, then work upwards. You never form $X^\top X$, so you never square the condition number. This is what `lm()` does internally. As a bonus, $(X^\top X)^{-1} = R^{-1}R^{-\top}$, which you need for the standard errors.

```r
X <- model.matrix(Petal.Length ~ Species, data = iris)
y <- iris$Petal.Length
qx <- qr(X)
beta <- backsolve(qr.R(qx), crossprod(qr.Q(qx), y))   # solves R beta = Q'y
n <- nrow(X); p <- ncol(X)
e <- y - X %*% beta
sigma2 <- sum(e^2) / (n - p)                          # residual variance
XtX_inv <- chol2inv(qr.R(qx))                         # (X'X)^{-1} from R
se <- sqrt(diag(XtX_inv) * sigma2)
cbind(beta, se, t = beta / se)                         # matches summary(lm(...))
```

## Standard errors, t and p

The residual variance estimate is $\hat\sigma^2 = \frac{e^\top e}{n - p}$; dividing by the degrees of freedom $n - p$ (not $n$) makes it unbiased, because fitting $p$ coefficients uses up $p$ directions of freedom. Then

$$
\widehat{\operatorname{Var}}(\hat\beta) = \hat\sigma^2 (X^\top X)^{-1}, \qquad t_j = \frac{\hat\beta_j}{\sqrt{\hat\sigma^2 [(X^\top X)^{-1}]_{jj}}},
$$

and the two-sided p-value is $2\,P(T_{n-p} > |t_j|)$, with $T_{n-p}$ a Student-t variable with $n - p$ degrees of freedom. Our code got this part right.

## Standardized residuals need leverage

For the Scale-Location plot, R plots $\sqrt{|r_i|}$ against the fitted values, where $r_i$ is the **standardized** (internally studentized) residual. We computed it as a z-score of the residuals, $(e_i - \bar e)/\operatorname{sd}(e)$. That's not the same thing, and the difference is the useful part.

Residuals don't all have the same variance, even if the errors do. With the hat matrix $H = X(X^\top X)^{-1}X^\top$ (it "puts the hat on" $y$: $\hat y = Hy$), the residuals are $e = (I - H)y$ and

$$
\operatorname{Var}(e_i) = \sigma^2 (1 - h_{ii}),
$$

where $h_{ii}$, the $i$-th diagonal element of $H$, is the **leverage** of point $i$: how unusual its predictor values are. A point with high leverage pulls the fitted line towards itself, so its residual is *smaller* than its error. The standardized residual corrects for that:

$$
r_i = \frac{e_i}{\hat\sigma \sqrt{1 - h_{ii}}}.
$$

With a z-score, high-leverage points look harmless exactly when you most want to notice them. Also, with an intercept the residuals already sum to zero, so subtracting $\bar e$ does nothing. In R, `rstandard(fit)` gives $r_i$, and `hatvalues(fit)` gives $h_{ii}$; with QR, $h_{ii}$ is the squared length of row $i$ of $Q$, `rowSums(qr.Q(qx)^2)`.

## Key takeaways

- Model: $y = X\beta + \varepsilon$; fit: $\hat y = X\hat\beta$, residuals $e = y - \hat y$.
- Normal equations square the condition number; QR solves $R\hat\beta = Q^\top y$ and is what `lm()` uses.
- $\hat\sigma^2 = e^\top e / (n - p)$ and $\operatorname{Var}(\hat\beta) = \hat\sigma^2 (X^\top X)^{-1}$ give the t-tests.
- Standardized residuals divide by $\hat\sigma\sqrt{1 - h_{ii}}$, not by the sample sd of the residuals. Leverage matters.

## Resources

- 732A94 Advanced R Programming, Linköping University, lab 4 (2019).
- Julian J. Faraway, *Linear Models with R*, 2nd ed., Chapter 2 (estimation, QR) and Chapter 6 (diagnostics, leverage, standardized residuals).
- Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*, Lectures 7–8 (QR) and 11 (least squares).
- R documentation, [influence.measures / rstandard / hatvalues](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html).
