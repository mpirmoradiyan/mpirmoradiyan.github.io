---
title: "Benjamini–Hochberg: controlling the false discovery rate when testing thousands of features"
slug: benjamini-hochberg-fdr
date: 2019-12-09
topic: Statistical Inference
tags: [university-lab, multiple-testing, false-discovery-rate, hypothesis-testing, feature-selection, high-dimensional]
summary: "Why testing every feature at 0.05 floods you with false positives, how the Benjamini–Hochberg step-up procedure controls the false discovery rate instead, and two easy ways to implement it wrong."
sources:
  - title: "732A99 Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Controlling the false discovery rate: a practical and powerful approach to multiple testing"
    author: "Yoav Benjamini, Yosef Hochberg"
    url: "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x"
    type: paper
  - title: "The Elements of Statistical Learning, Section 18.7 (Feature Assessment and the Multiple-Testing Problem)"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book


status: draft
---

When you run one test at the 5% level, a false alarm happens 5% of the time. When you run 4,702 tests, one per word feature in a text-classification dataset, as in one of my labs, you expect about 235 false alarms even if *no* feature matters. So "how many p-values are below 0.05" is the wrong question. The Benjamini–Hochberg (BH) procedure answers a better one: which features can I call significant while keeping the share of false discoveries among them small?

## What can go wrong when testing many hypotheses

Suppose we test $m$ null hypotheses and reject $R$ of them, and $V$ of those rejections are wrong (true nulls we rejected). Two ways to control errors:

- **Family-wise error rate**: $\text{FWER} = P(V \ge 1)$, the chance of even one false rejection. Bonferroni controls it by testing each hypothesis at $\alpha/m$. With $m$ in the thousands that's extremely strict, and you find almost nothing.
- **False discovery rate**: $\text{FDR} = E\left[\frac{V}{\max(R, 1)}\right]$, the expected *fraction* of rejections that are false. Controlling FDR at 5% means "among the features I flag, on average at most 5% are noise". For screening many features this is usually what you want.

## The BH step-up procedure

1. Sort the p-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Find the **largest** $k$ such that
$$
p_{(k)} \le \frac{k}{m}\,\alpha .
$$
3. Reject the hypotheses with the $k$ smallest p-values, $p_{(1)}, \dots, p_{(k)}$. All of them.

Benjamini and Hochberg proved this keeps FDR $\le \frac{m_0}{m}\alpha \le \alpha$ (with $m_0$ the number of true nulls) when the tests are independent, and it's known to hold under some kinds of positive dependence too.

The picture that helped me: plot the sorted p-values against their rank $j$, and draw the line $j\alpha/m$. Find the last point that's under the line. Everything to the left of it is rejected, even points that pop above the line before that.

## Two ways I got it wrong

**1. Counting raw p-values.** My lab report said "281 null hypotheses are rejected if we use the rejection threshold 0.05". That's the count of unadjusted p-values below 0.05, with no correction at all. With 4,702 tests, a few hundred of those can be pure noise.

**2. Comparing point by point.** In a later version I sorted the p-values and labelled each one "reject" if $p_{(j)} < j\alpha/m$. That's not BH either. BH is a *step-up* rule: you look for the largest $k$ under the line and reject everything up to it. A p-value that sits slightly above its own line but before the cut-off point is still rejected. The point-by-point version can only reject fewer.

A related trap is computing BH-adjusted p-values by hand. They're defined from the largest rank downward,

$$
\tilde p_{(j)} = \min_{i \ge j} \left( \frac{m}{i}\, p_{(i)} \right) \text{ (capped at 1)},
$$

so each adjusted value is a running minimum of the *adjusted* values above it. When practising this for the exam I wrote the recursion with the previous *raw* p-value instead of the previous adjusted one, which gives wrong numbers. `p.adjust(p, method = "BH")` does it correctly; rejecting where $\tilde p \le \alpha$ is the same as the step-up rule.

```r
bh_reject <- function(p, alpha = 0.05) {
  m <- length(p)
  o <- order(p)                                  # sort p-values, smallest first
  below <- which(p[o] <= (1:m) / m * alpha)      # sorted p_(j) under the line j * alpha / m
  k <- if (length(below)) max(below) else 0      # the LARGEST such j
  reject <- rep(FALSE, m)
  reject[o[seq_len(k)]] <- TRUE                  # reject p_(1), ..., p_(k), all of them
  reject
}
set.seed(6)
real <- rep(c(FALSE, TRUE), c(900, 100))         # 900 true nulls, 100 real effects
p <- ifelse(real, rbeta(1000, 0.3, 30), runif(1000))
rules <- list(raw = p < 0.05, bonferroni = p < 0.05 / 1000, bh = bh_reject(p))
sapply(rules, function(r) c(rejected = sum(r), false = sum(r & !real)))
#           raw  bonferroni  bh
# rejected  134          13  57
# false      40           0   3
all.equal(bh_reject(p), p.adjust(p, "BH") <= 0.05)   # TRUE
```

In this simulation the raw rule flags 134 features, but 40 of them are noise (30%). Bonferroni is clean but finds only 13. BH finds 57 with 3 false ones, about 5%, which is what it promises.

## When to use it

- Screening many features (genes, words, pixels, sensors) before a closer look.
- When some false positives are acceptable but you want their share bounded.
- Not when a single false positive is costly; then you want FWER control (Bonferroni or Holm).

Also keep in mind that each p-value is only as good as its test. A $t$-test per word count assumes roughly normal, independent data, which sparse count features often aren't.

## Key takeaways

- With $m$ tests at level $\alpha$, expect about $m\alpha$ false positives even when nothing is real.
- FDR is the expected share of false discoveries among rejections; Bonferroni controls the much stricter FWER.
- BH: sort p-values, find the largest $k$ with $p_{(k)} \le k\alpha/m$, reject the $k$ smallest.
- It's a step-up rule, not a point-by-point comparison; use `p.adjust(..., "BH")` to avoid hand-rolled mistakes.

## Resources

- 732A99 Machine Learning, Linköping University (course lab, autumn 2019)
- Yoav Benjamini and Yosef Hochberg, "Controlling the false discovery rate: a practical and powerful approach to multiple testing", *Journal of the Royal Statistical Society B* 57(1) (1995): https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 18.7: https://hastie.su.domains/ElemStatLearn/
- R documentation for `p.adjust`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html
