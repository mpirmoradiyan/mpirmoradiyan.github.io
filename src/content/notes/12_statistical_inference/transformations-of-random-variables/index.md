---
title: "Transforming random variables: the CDF method, the Jacobian, and the Cauchy distribution"
slug: transformations-of-random-variables
date: 2020-09-06
topic: Statistical Inference
tags: [university-lab, probability, change-of-variables, cauchy-distribution, joint-distributions]
summary: "How to find the density of a function of random variables with the CDF method and the change-of-variables formula, worked through for 1/X and X/Y, which both turn out to be Cauchy."
sources:
  - title: "732A63 Probability Theory (course seminar exercises)"
    author: "Linköping University"
    type: course
  - title: "An Intermediate Course in Probability (2nd ed.), Chapter 1"
    author: "Allan Gut"
    type: book
  - title: "Statistical Inference (2nd ed.), Sections 2.1 and 4.3"
    author: "George Casella, Roger L. Berger"
    type: book


status: draft
---

A lot of probability comes down to one question: if I know the distribution of $X$ (or of $X$ and $Y$ together), what's the distribution of some function of them? The first seminar of the probability theory course was full of these, and my handwritten solutions from that week show both standard methods, plus a couple of small slips worth fixing. Two results stand out because they're surprising: if $X$ is Cauchy, so is $1/X$; and the ratio of two independent standard normals is Cauchy too.

## Warm-up: marginals and independence from a joint density

If a joint density factorises as $f_{X,Y}(x,y) = g(x)\,h(y)$ over a rectangular region, then $X$ and $Y$ are independent and each marginal is just the normalised factor. The marginal is obtained by integrating the other variable out:

$$
f_X(x) = \int f_{X,Y}(x, y)\, dy,
$$

where the integral runs over all $y$ values where the density is positive.

For example, a density proportional to $e^{-x^2/2}\, e^{-3y}$ on $-\infty < x < \infty$, $y > 0$ splits into a normal part and an exponential part. Integrating out $y$ leaves something proportional to $e^{-x^2/2}$, so $X \sim N(0, 1)$; integrating out $x$ leaves something proportional to $e^{-3y}$, so $Y$ is exponential with rate 3 and density $3e^{-3y}$. The trick is to recognise the kernel (the part that depends on the variable) and let the normalising constant take care of itself. Independence then makes expectations of products easy: $E[g(X)\,k(Y)] = E[g(X)]\,E[k(Y)]$, and second moments come from $E[Y^2] = \text{Var}(Y) + (E Y)^2$.

## Method 1: through the CDF

To get the density of $Y = g(X)$, write the CDF of $Y$ as a probability about $X$, then differentiate:

$$
F_Y(y) = P(g(X) \le y), \qquad f_Y(y) = \frac{d}{dy} F_Y(y).
$$

The care is all in turning "$g(X) \le y$" into a set of $x$ values correctly.

## Method 2: change of variables

If $g$ is strictly monotone and differentiable with inverse $h = g^{-1}$, then

$$
f_Y(y) = f_X\big(h(y)\big)\, \left|\frac{dh(y)}{dy}\right|,
$$

where the absolute value of the derivative (the Jacobian, in one dimension) accounts for how $g$ stretches or squeezes intervals. If $g$ is monotone on separate pieces, you add one such term per piece.

## Example: $1/X$ for a Cauchy $X$

The standard Cauchy density is $f_X(x) = \dfrac{1}{\pi(1+x^2)}$ for all real $x$. Let $Y = 1/X$. Here $h(y) = 1/y$ and $|h'(y)| = 1/y^2$. The map $x \mapsto 1/x$ is decreasing on $x<0$ and on $x>0$ separately, and each $y \neq 0$ comes from exactly one $x$, so

$$
f_Y(y) = \frac{1}{\pi\big(1 + 1/y^2\big)} \cdot \frac{1}{y^2} = \frac{1}{\pi(y^2 + 1)}.
$$

So $Y$ is standard Cauchy again.

My handwritten version had two slips on the way to the right answer. First, I wrote $F_Y(y) = P(1/X \le y) = P(X \ge 1/y)$. That's only true when $X$ and $y$ are both positive. For $y > 0$, $1/X \le y$ also holds for every negative $X$, so the correct statement is $P(1/X \le y) = P(X < 0) + P(X \ge 1/y)$. The derivative is the same, which is why the final answer still came out right, but the step itself was wrong. Second, I wrote $f_X(1 - 1/y)$ where it should have been $f_X(1/y)$, which was just a copying slip. The change-of-variables route above avoids the first trap altogether, since it never needs the sign cases.

## Example: the ratio of two standard normals

Let $X, Y$ be independent $N(0,1)$ and $U = X/Y$. Use the CDF method:

$$
P(U \le u) = P\!\left(\frac{X}{Y} \le u\right).
$$

Multiplying through by $Y$ flips the inequality when $Y < 0$:

$$
P(U \le u) = P(X \le uY,\; Y > 0) + P(X \ge uY,\; Y < 0).
$$

Since $X$ is symmetric around 0 and independent of $Y$, $P(X \ge uY \mid Y = y) = P(X \le -uy \mid Y=y)$, and for $y < 0$, $-uy = u|y|$. So both pieces combine to

$$
P(U \le u) = P\big(X \le u|Y|\big) = \int_{-\infty}^{\infty} \Phi(u|y|)\, \varphi(y)\, dy,
$$

where $\varphi$ and $\Phi$ are the standard normal density and CDF. In my notes I wrote $P(X/Y \le v) = P(X \le v|Y|)$ directly. That's the right result, but it needs the symmetry argument above; for a non-symmetric $X$ it would be false.

My notes stopped at the double integral. To finish, differentiate under the integral with respect to $u$:

$$
f_U(u) = \int_{-\infty}^{\infty} |y|\, \varphi(u|y|)\, \varphi(y)\, dy = 2\int_0^\infty \frac{y}{2\pi} e^{-(1+u^2)y^2/2}\, dy = \frac{1}{\pi(1+u^2)}.
$$

The last integral is $\int_0^\infty y\,e^{-a y^2/2}dy = 1/a$ with $a = 1 + u^2$. So $X/Y$ is standard Cauchy.

## Why the Cauchy is a troublemaker

The Cauchy has such heavy tails that $E|X| = \infty$: it has no mean and no variance. The law of large numbers doesn't apply, and the average of $n$ Cauchy draws is itself standard Cauchy for every $n$, so it never settles down. A quick simulation shows both results above and this behaviour:

```r
set.seed(9)
n <- 1e5
x <- rcauchy(n)
ratio <- rnorm(n) / rnorm(n)
p <- c(0.1, 0.25, 0.5, 0.75, 0.9)
round(rbind(cauchy = qcauchy(p),             # -3.08 -1.00 0.00 1.00 3.08
            inv_x  = quantile(1 / x, p),     # about the same
            ratio  = quantile(ratio, p)), 2) # about the same
# Running means of Cauchy draws do not converge
round(sapply(c(1e2, 1e3, 1e4, 1e5), function(m) mean(x[1:m])), 2)
```

This is also a practical warning: ratios of noisy quantities whose denominators can be near zero can behave like Cauchy variables, with occasional huge values and unstable averages.

## Key takeaways

- A joint density that factorises over a rectangle means independence; marginals come from integrating the other variable out.
- CDF method: write $P(g(X) \le y)$ as a probability about $X$, carefully, including sign cases, then differentiate.
- Change of variables: $f_Y(y) = f_X(h(y))\,|h'(y)|$, summed over monotone pieces.
- $1/X$ for Cauchy $X$, and $X/Y$ for independent standard normals, are both standard Cauchy.
- The Cauchy has no mean; sample averages of Cauchy data don't converge.

## Resources

- 732A63 Probability Theory, Linköping University, seminar exercises (autumn 2020).
- Allan Gut, *An Intermediate Course in Probability*, 2nd ed., Springer, 2009, Chapter 1 (multivariate random variables and transformations).
- George Casella and Roger L. Berger, *Statistical Inference*, 2nd ed., Duxbury, 2002, Section 2.1 (distributions of functions of a random variable) and Section 4.3 (bivariate transformations).
