---
title: "Grouped and spatial cross-validation: why random K-fold lies when samples cluster"
slug: grouped-cross-validation
date: 2026-09-24
topic: Model Evaluation
tags: [cross-validation, data-leakage, spatial-autocorrelation, model-evaluation, group-k-fold]
summary: "Why random K-fold gives scores that are too good when samples come in groups or sit close together in space, and how splitting by group or by spatial block gives an honest estimate."
sources:
  - title: "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"
    author: "David R. Roberts et al."
    url: "https://doi.org/10.1111/ecog.02881"
    type: paper
  - title: "Spatial validation reveals poor predictive performance of large-scale ecological mapping models"
    author: "Pierre Ploton et al."
    url: "https://doi.org/10.1038/s41467-020-18321-y"
    type: paper
  - title: "scikit-learn User Guide: Cross-validation iterators for grouped data"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold"
    type: docs
  - title: "The Elements of Statistical Learning, Section 7.10 (Cross-Validation)"
    author: "Trevor Hastie, Robert Tibshirani, Jerome Friedman"
    url: "https://hastie.su.domains/ElemStatLearn/"
    type: book


status: draft
---

The puzzle that got me here: a model scores great in cross-validation, then does clearly worse on new data. Cross-validation is how we guess a model's error on data it hasn't seen. But that guess is only honest if the test fold is as "new" to the model as real future data will be. When samples come in groups (many rows per patient, per field, per city, per sensor) or sit close together in space, a random split puts near-copies of each test row into the training set. So the score ends up measuring memory, not how well the model works on new cases, and it can be way too good.

## The assumption hiding in random K-fold

First, a quick reminder of what K-fold does. It shuffles the rows, cuts them into $K$ folds, trains on $K-1$ folds and tests on the one left over. The average test error is an estimate of the prediction error

$$
\text{Err} = \mathbb{E}\big[L(Y, \hat f(X))\big],
$$

where $L$ is the loss (for example squared error), $\hat f$ is the fitted model and $(X, Y)$ is a fresh sample. "Fresh" is the word that matters. The estimate is only trustworthy when test rows are independent of training rows, the same way future data will be.

Grouped data breaks this. Rows from the same group share things the features may not fully capture: the same instrument, the same weather, the same soil, the same person. So their errors are correlated. If one row of a group is in training and its sibling is in test, the model can do well on the sibling just by recognizing the group. This is the part that clicked for me: in real use you'll usually predict for a *new* group, and that shortcut isn't there.

Spatial data has the same problem, even without explicit groups. Tobler's "first law of geography" says near things are more related than distant things. A test point 5 m from a training point isn't really a fresh sample; it's almost a duplicate.

## A small synthetic test

I wanted to see the size of the effect myself, so I made some fake data. It has 30 groups of 40 rows. The target depends on one real feature plus an offset per group. Two extra features act as a group "signature": they're almost constant inside a group and don't cause anything, but they give away which group a row belongs to.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

rng = np.random.default_rng(0)
n_groups, per_group = 30, 40
groups = np.repeat(np.arange(n_groups), per_group)

# Each group has a shared offset in the target and a shared signature in two features
offset = rng.normal(0, 3, n_groups)[groups]
signature = rng.normal(0, 1, (n_groups, 2))[groups]

x_real = rng.uniform(0, 10, len(groups))           # the feature that truly matters
X = np.column_stack([x_real, signature + rng.normal(0, 0.05, signature.shape)])
y = 2.0 * x_real + offset + rng.normal(0, 1, len(groups))

model = RandomForestRegressor(n_estimators=200, random_state=0)
random_cv = KFold(n_splits=5, shuffle=True, random_state=0)
group_cv = GroupKFold(n_splits=5)

rmse = "neg_root_mean_squared_error"
r_random = -cross_val_score(model, X, y, cv=random_cv, scoring=rmse)
r_group = -cross_val_score(model, X, y, cv=group_cv, groups=groups, scoring=rmse)
print(f"random K-fold RMSE: {r_random.mean():.2f}")   # 1.63
print(f"group K-fold RMSE:  {r_group.mean():.2f}")    # 3.01
```

Random K-fold says the RMSE is 1.63. Group K-fold, which keeps each group completely in train or completely in test, says 3.01. The second number is what you'd actually see on a new group.

The sneakier problem shows up when you use cross-validation to *choose* features. Drop the signature features and run it again:

```python
X_real = X[:, :1]
print(-cross_val_score(model, X_real, y, cv=random_cv, scoring=rmse).mean())  # 3.13
print(-cross_val_score(model, X_real, y, cv=group_cv, groups=groups,
                       scoring=rmse).mean())                                 # 3.05
```

With random K-fold, the signature features look like a big win (3.13 down to 1.63). With group K-fold they do basically nothing (3.05 versus 3.01). So a feature search driven by random K-fold would happily keep features that only help the model memorize groups. Model selection and hyperparameter tuning get the same bias: leaky validation rewards whichever flexible model memorizes best.

## Split by group, not by row

Once you see the problem, the fix is simple to say: split by group, not by row. The common options:

- **Group K-fold**: put whole groups into $K$ folds, trying to keep the fold sizes balanced.
- **Leave-one-group-out**: one fold per group. Closest to the real situation when you have few groups, but slow and noisy when groups are small.
- **Stratified group K-fold**: whole groups per fold, while also trying to keep the target (or class) distribution similar across folds.
- **Nested grouped CV**: if you also tune hyperparameters, the inner tuning loop has to be grouped too. Otherwise the tuning leaks and the outer score is still too optimistic.

Pick the group based on how the model will be used. If it'll be used on new patients, group by patient. If it'll be used on new regions, group by region, even if the score looks worse. The question the score should answer is: "how well will this work where I'll actually use it?"

## Spatial blocks

With spatial data there's often no natural group label. Roberts et al. (2017) suggest *block cross-validation*: cut the map into blocks and treat each block as a group. The blocks should be bigger than the distance over which the residuals (the model's errors) are still correlated. You can estimate that distance from a variogram of the residuals, or just try a few block sizes and see when the error stops changing.

```python
import numpy as np
from sklearn.model_selection import GroupKFold

rng = np.random.default_rng(1)
xy = rng.uniform(0, 100, (600, 2))              # sample locations in metres

# Cut the map into 25 m x 25 m tiles and use the tile id as the group
block = 25.0
tile_id = (xy[:, 1] // block).astype(int) * 100 + (xy[:, 0] // block).astype(int)

for train_idx, test_idx in GroupKFold(n_splits=4).split(xy, groups=tile_id):
    print(len(train_idx), len(test_idx), np.unique(tile_id[test_idx]).size, "tiles held out")
```

![Two maps of 600 random points. Left: random K-fold, where the four test folds are mixed everywhere so every test point has training neighbours. Right: spatial block folds, where each 25 m tile belongs entirely to one fold.](fig-random-vs-blocked-folds.png)

In the left panel every test point has training neighbours a few metres away. In the right panel, a test point's nearest training neighbour is usually in another tile. There are variations too: a buffer zone (drop training points within some distance of the test block), or checkerboard and other regular block layouts so each fold covers the whole range of conditions.

Ploton et al. (2020) is a good example of why this matters. A large-scale biomass map looked accurate under random cross-validation, but spatial cross-validation showed it had almost no predictive skill far from the training data.

## The catch

Grouped and blocked CV have their own downsides:

- **Higher variance.** With few groups, each fold's score depends a lot on which groups ended up in it. Report the spread across folds, not just the mean, and maybe repeat with different group-to-fold assignments.
- **Possible pessimism.** If the blocks are much bigger than needed, test folds can land outside the range of the training data. Then you're measuring extrapolation instead of the interpolation you'll really be doing.
- **Unbalanced folds.** Groups have different sizes. Check how many rows and groups end up in each fold.
- **Preprocessing leaks too.** Scaling, imputation, target encoding and feature selection all have to be fitted inside each training fold. Put them in a pipeline so the grouped split covers them.

## Key takeaways

- Random K-fold assumes test rows are independent of training rows. Grouped or spatially clustered data breaks that.
- Leaky validation isn't just too optimistic; it also picks the wrong features and the wrong models.
- Split by the unit you'll meet in real use: group K-fold, leave-one-group-out, or spatial blocks.
- Make blocks bigger than the range where residuals are correlated, and report the spread across folds.
- Every step that learns from data (tuning, scaling, encoding) has to sit inside the grouped split.

## Resources

- David R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", *Ecography* 40 (2017): https://doi.org/10.1111/ecog.02881
- Pierre Ploton et al., "Spatial validation reveals poor predictive performance of large-scale ecological mapping models", *Nature Communications* 11 (2020): https://doi.org/10.1038/s41467-020-18321-y
- Hanna Meyer et al., "Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation", *Environmental Modelling & Software* 101 (2018): https://doi.org/10.1016/j.envsoft.2017.12.001
- Alexandre M. J.-C. Wadoux et al., "Spatial cross-validation is not the right way to evaluate map accuracy", *Ecological Modelling* 457 (2021), a useful counterpoint on when design-based validation is better: https://doi.org/10.1016/j.ecolmodel.2021.109692
- Shachar Kaufman et al., "Leakage in data mining: formulation, detection, and avoidance", *ACM TKDD* 6(4) (2012): https://doi.org/10.1145/2382577.2382579
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 7.10: https://hastie.su.domains/ElemStatLearn/
- scikit-learn User Guide, "Cross-validation iterators for grouped data": https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
