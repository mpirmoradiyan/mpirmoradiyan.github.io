---
title: "Monotonic constraints in gradient-boosted trees: building domain knowledge into the model"
slug: monotonic-constraints-in-gradient-boosting
date: 2026-09-25
topic: ML & Statistics
tags: [gradient-boosting, monotonic-constraints, decision-trees, regularization, interpretability]
summary: "What a monotonic constraint means, how tree boosting libraries enforce it during split finding, and when it makes a model more accurate and easier to trust."
sources:
  - title: "XGBoost documentation: Monotonic Constraints"
    author: "XGBoost developers"
    url: "https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html"
    type: docs
  - title: "LightGBM documentation: Parameters (monotone_constraints)"
    author: "LightGBM developers"
    url: "https://lightgbm.readthedocs.io/en/latest/Parameters.html#monotone_constraints"
    type: docs
  - title: "scikit-learn User Guide: Monotonic Constraints"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/ensemble.html#monotonic-constraints"
    type: docs
  - title: "Monotonic Calibrated Interpolated Look-Up Tables"
    author: "Maya Gupta et al."
    url: "https://jmlr.org/papers/v17/15-243.html"
    type: paper


status: draft
---

Often you know the *direction* of a relationship before you see any data. A bigger house should not lower the predicted price, all else equal. A higher dose should not lower the predicted effect in the range you care about. A flexible model trained on a small, noisy sample does not know this. It will happily learn little dips and bumps that are pure noise. A monotonic constraint tells the model: "along this feature, predictions may only go up" (or only down). It is one of the cheapest ways to add domain knowledge to a tree ensemble.

## What "monotone" means here

Let $f(x_1, \dots, x_p)$ be the model's prediction. The model is *monotonically increasing* in feature $j$ if

$$
x_j \le x_j' \;\Rightarrow\; f(x_1, \dots, x_j, \dots, x_p) \le f(x_1, \dots, x_j', \dots, x_p)
$$

for every fixed setting of the other features. In words: change only $x_j$, keep everything else the same, and the prediction can never move in the wrong direction. Decreasing is the same with $\ge$.

Two details matter:

- This is a *ceteris paribus* statement ("other things equal"). It says nothing about the raw correlation between $x_j$ and the target, which can be distorted by other features.
- It must hold for **every** combination of the other features, not just on average. A partial dependence plot that rises on average does not prove monotonicity.

## Why unconstrained trees wiggle

A regression tree predicts a constant in each leaf. Boosting adds hundreds of such trees. Each split is chosen to reduce the training loss, so whenever noise makes a few points on the right of a threshold slightly lower than those on the left, a split appears that encodes that dip. With little data per region, these dips survive into the final model.

![Predicted y as x1 sweeps from 0 to 10 with the other features fixed. The dashed true curve rises smoothly. The unconstrained boosting curve follows it roughly but has many dips and spikes. The monotone-constrained curve is a rising staircase that never goes down.](fig-monotone-response.png)

The figure uses synthetic data where the true effect of $x_1$ is $3\sqrt{x_1}$, plus noise. The unconstrained model (orange) has dozens of reversals. The constrained model (blue) is a staircase that only climbs. It is still a tree model, so it is piecewise constant, but its shape now matches what we know.

## How the constraint is enforced

The constraint is applied while each tree grows, not as a post-processing step. The idea, as documented for XGBoost and LightGBM, has two parts.

**1. Reject splits in the wrong direction.** Suppose feature $j$ must be increasing and a candidate split sends $x_j < t$ to the left child and $x_j \ge t$ to the right. Each child has an optimal leaf value, for squared loss roughly the mean residual in that child, $w_L$ and $w_R$. The split is allowed only if $w_L \le w_R$. Splits that would make the prediction drop as $x_j$ increases are discarded (or the values are clipped).

**2. Pass bounds down the tree.** Checking one split is not enough. A later split deeper in the left subtree could create a leaf with a value higher than some leaf in the right subtree. To prevent this, after a valid split on $x_j$ the algorithm picks a bound $m$ between $w_L$ and $w_R$ (for example their midpoint). Every leaf below the left child is capped at $m$ from above; every leaf below the right child is limited to $m$ from below. Since the final model is a sum of trees, and a sum of monotone functions is monotone, the whole ensemble respects the constraint.

LightGBM offers stricter or looser variants of step 2 (`monotone_constraints_method` = `basic`, `intermediate`, `advanced`). The looser bounds let the trees fit more detail while keeping the guarantee.

## Using it

Constraints are given as one number per feature: $+1$ for increasing, $-1$ for decreasing, $0$ for no constraint. The names differ by library (`monotone_constraints` in XGBoost, LightGBM and CatBoost; `monotonic_cst` in scikit-learn), but the idea is the same. Here is a small experiment with scikit-learn's histogram gradient boosting on synthetic data with 300 noisy training rows:

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error as rmse

rng = np.random.default_rng(0)
def make_data(n):
    x1 = rng.uniform(0, 10, n)            # true effect: increasing
    x2 = rng.uniform(0, 10, n)            # true effect: decreasing
    x3 = rng.normal(0, 1, n)              # no fixed direction
    y = 3 * np.sqrt(x1) - 0.3 * x2 + np.sin(2 * x3) + rng.normal(0, 1.5, n)
    return np.column_stack([x1, x2, x3]), y

X_tr, y_tr = make_data(300)               # small, noisy training set
X_te, y_te = make_data(5000)

params = dict(max_iter=300, learning_rate=0.1, random_state=0)
free = HistGradientBoostingRegressor(**params).fit(X_tr, y_tr)
mono = HistGradientBoostingRegressor(monotonic_cst=[1, -1, 0], **params).fit(X_tr, y_tr)
print(f"unconstrained test RMSE {rmse(y_te, free.predict(X_te)):.3f}")   # 1.781
print(f"monotone      test RMSE {rmse(y_te, mono.predict(X_te)):.3f}")   # 1.666
```

And a direct check of the guarantee:

```python
# Sweep x1 with the other features fixed and count steps that go down
grid = np.linspace(0, 10, 200)
probe = np.column_stack([grid, np.full(200, 5.0), np.zeros(200)])
for name, m in [("unconstrained", free), ("monotone", mono)]:
    steps = np.diff(m.predict(probe))
    print(f"{name:13s} downward steps along x1: {(steps < -1e-9).sum()}")  # 44 vs 0
```

The constrained model is both more accurate on new data and free of reversals. The gain comes from regularization: the constraint removes a large set of noisy shapes the model could otherwise fit. With plenty of clean data the accuracy gain shrinks, but the guarantee remains.

## When it helps, and when it hurts

**Good reasons to constrain a feature:**

- Physics, biology or business logic fixes the direction, and you are confident it holds across the whole input range.
- Training data is small or noisy, so the model needs regularization.
- Users or regulators will inspect the model. "A larger value never lowers the score" is easy to explain and to test.
- You expect inputs near the edge of the training range, where unconstrained trees are least reliable.

**Ways it can go wrong:**

- **The direction is only true on average.** If the real effect rises and then falls (a hump), a monotone constraint forces the model to flatten one side. That is bias, not regularization.
- **Correlated features.** If two features carry the same information with opposite signs in the model, constraining one can push the model to use the other in a strange way. Look at all related features together.
- **Transformed inputs.** The constraint applies to the feature as the model sees it. If you feed $1/x$ instead of $x$, the direction flips. Ratios and differences of features need their own reasoning.
- **Interactions.** The constraint holds for every setting of the other features. If the true direction of $x_j$ depends on another feature, no global constraint is correct.

A practical habit: compare constrained and unconstrained models under the same honest validation (grouped folds if the data is grouped). If the constrained one is no worse, prefer it; it is the safer model to deploy.

## Key takeaways

- A monotonic constraint forces predictions to move in one direction along a feature, with all other features held fixed.
- Tree boosting libraries enforce it during training by rejecting wrong-direction splits and passing value bounds down each tree.
- It acts as strong, knowledge-based regularization: smoother, more trustworthy models, often with better accuracy on small or noisy data.
- Constrain only directions you are sure of across the whole range, and remember that transforms and correlated features change what "direction" means.

## Resources

- XGBoost documentation, "Monotonic Constraints": https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html
- LightGBM documentation, parameters `monotone_constraints` and `monotone_constraints_method`: https://lightgbm.readthedocs.io/en/latest/Parameters.html#monotone_constraints
- CatBoost documentation, training parameter `monotone_constraints`: https://catboost.ai/docs/en/references/training-parameters/common#monotone_constraints
- scikit-learn User Guide, "Monotonic Constraints" for histogram gradient boosting: https://scikit-learn.org/stable/modules/ensemble.html#monotonic-constraints
- Maya Gupta et al., "Monotonic Calibrated Interpolated Look-Up Tables", *JMLR* 17 (2016), on why monotonicity aids trust and generalization: https://jmlr.org/papers/v17/15-243.html
- Christoph Molnar, *Interpretable Machine Learning*, chapters on partial dependence and ICE plots: https://christophm.github.io/interpretable-ml-book/
- Jerome H. Friedman, "Greedy function approximation: A gradient boosting machine", *Annals of Statistics* 29(5) (2001): https://doi.org/10.1214/aos/1013203451
