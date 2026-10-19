---
title: "Stacking regression models: combining predictions without fooling yourself"
slug: stacking-regression-models
date: 2025-11-07
topic: Machine Learning
tags: [stacking, ensembles, out-of-fold, data-leakage, regression]
summary: "How stacking learns to combine several regression models, and why the combiner has to be trained on out-of-fold predictions, not in-sample ones."
sources:
  - title: "Stacked generalization"
    author: "David H. Wolpert"
    url: "https://doi.org/10.1016/S0893-6080(05)80023-1"
    type: paper
  - title: "Stacked regressions"
    author: "Leo Breiman"
    url: "https://doi.org/10.1007/BF00117832"
    type: paper
  - title: "Super Learner"
    author: "Mark J. van der Laan, Eric C. Polley, Alan E. Hubbard"
    url: "https://doi.org/10.2202/1544-6115.1309"
    type: paper
  - title: "scikit-learn User Guide: Stacked generalization"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization"
    type: docs


status: draft
---

Here's the situation that got me curious. You've trained a few regression models and they each make different mistakes. A linear model misses curves, a tree ensemble is blocky between splits, and a nearest-neighbour model struggles where data is sparse. So why not let a second model learn how to combine them? That's stacking (or *stacked generalization*): one model's strengths can cover another's weaknesses. It often gives a small but real gain. But there's one classic trap, and it's easy to walk into. If the combiner sees predictions the base models made on their own training data, it learns to trust the wrong model.

## Two levels

So basically, there are two levels:

- **Level 0 (base models):** $M$ different models $f_1, \dots, f_M$, each trained to predict the target $y$ from the features $x$.
- **Level 1 (meta-model or combiner):** a model $g$ that takes the base predictions $\big(f_1(x), \dots, f_M(x)\big)$ as its input and gives the final prediction.

The most common combiner is just a weighted sum,

$$
\hat y(x) = w_0 + \sum_{m=1}^{M} w_m \, f_m(x),
$$

where $w_m$ is the weight for model $m$ and $w_0$ is an intercept. A plain average is the special case $w_m = 1/M$, $w_0 = 0$. Stacking *learns* the weights from data instead.

Wolpert (1992) came up with the general idea. Breiman (1996) looked at it for regression and found two practical rules that still hold: build the level-1 inputs from cross-validated predictions, and keep the weights non-negative ($w_m \ge 0$). Later, van der Laan et al. (2007) proved that the cross-validated version (the "Super Learner") does asymptotically as well as the best possible weighted combination.

## Why in-sample predictions fool the combiner

To fit $g$, you need a table with one row per training sample and one column per base model. The tempting shortcut is to fit each base model on the training set and then predict that same training set. The problem is that flexible models almost memorize their training data, so their in-sample predictions look nearly perfect. The combiner then puts almost all its weight on whichever model memorizes best.

![Two scatter plots of random-forest predictions against the true target on a synthetic training set. Left: in-sample predictions hug the diagonal with RMSE 0.73. Right: out-of-fold predictions scatter much more, with RMSE 2.01, which is close to the true test error.](fig-in-sample-vs-oof.png)

The left panel shows a random forest predicting its own training rows: RMSE 0.73. The right panel shows *out-of-fold* (OOF) predictions, where each row is predicted by a copy of the forest that never saw it. RMSE jumps to 2.01, which is close to the forest's real test error. Only the right-hand numbers tell you how the model behaves on new data, so only those should be used to learn the weights.

## Out-of-fold predictions

The recipe:

1. Split the training set into $K$ folds.
2. For each base model and each fold $k$, train on the other $K-1$ folds and predict fold $k$.
3. Stack the predictions: every training row now has one honest prediction per base model.
4. Fit the combiner $g$ on this OOF table.
5. Refit each base model on the full training set; at prediction time, feed their outputs to $g$.

Here's the comparison on synthetic data (Friedman's benchmark function with noise):

```python
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error as rmse

X, y = make_friedman1(n_samples=1500, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)

base = {
    "ridge": RidgeCV(),
    "forest": RandomForestRegressor(n_estimators=300, random_state=0),
    "knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
}
for name, m in base.items():
    m.fit(X_tr, y_tr)
    print(f"{name:7s} test RMSE {rmse(y_te, m.predict(X_te)):.3f}")
P_te = np.column_stack([m.predict(X_te) for m in base.values()])  # level-1 test inputs

# Wrong: combiner trained on in-sample predictions
P_in = np.column_stack([m.predict(X_tr) for m in base.values()])
naive = LinearRegression().fit(P_in, y_tr)

# Right: combiner trained on out-of-fold predictions
cv = KFold(n_splits=5, shuffle=True, random_state=0)
P_oof = np.column_stack([cross_val_predict(m, X_tr, y_tr, cv=cv) for m in base.values()])
honest = LinearRegression().fit(P_oof, y_tr)

print("naive weights ", naive.coef_.round(2), f"test RMSE {rmse(y_te, naive.predict(P_te)):.3f}")
print("honest weights", honest.coef_.round(2), f"test RMSE {rmse(y_te, honest.predict(P_te)):.3f}")
print("simple average", f"test RMSE {rmse(y_te, P_te.mean(axis=1)):.3f}")
```

The output:

| Model | Test RMSE | Weights (ridge, forest, kNN) |
|---|---|---|
| Ridge alone | 2.536 | |
| Forest alone | 1.934 | |
| kNN alone | 2.594 | |
| Simple average | 2.108 | 0.33, 0.33, 0.33 |
| Stack on in-sample predictions | 1.964 | −0.12, 1.19, −0.03 |
| Stack on OOF predictions | 1.881 | 0.04, 0.89, 0.26 |

I see three lessons in this table. First, the naive stack is *worse* than the forest on its own. It gives the forest a weight above 1 and negative weights to the others, because in-sample the forest looked almost perfect. Second, the OOF stack beats every single model, because it finds out that kNN adds information the forest doesn't have. Third, a plain average can be worse than the best single model when one member is clearly weaker; learned weights avoid that.

## Rules of thumb

**Keep the combiner simple.** The level-1 inputs are strongly correlated (all the models predict the same target), so a flexible combiner overfits easily. A linear model with non-negative weights, as Breiman suggested, is a good default. Non-negative weights are easy to read, and they stop the combiner from building fragile "A minus B" tricks.

```python
from sklearn.ensemble import StackingRegressor

stack = StackingRegressor(
    estimators=list(base.items()),
    final_estimator=LinearRegression(positive=True),  # non-negative weights
    cv=cv,                                             # out-of-fold level-1 inputs
)
stack.fit(X_tr, y_tr)
print("weights", stack.final_estimator_.coef_.round(2),
      f"test RMSE {rmse(y_te, stack.predict(X_te)):.3f}")  # 1.881
```

**Variety matters more than count.** Three models from different families usually help more than ten tuned versions of one family. If two base models make the same mistakes, the second one adds nothing.

**Use the right folds.** If your data is grouped or spatial, the OOF predictions have to come from grouped folds too. Otherwise the level-1 table leaks group identity and you're back to the in-sample problem. (See the note on [grouped cross-validation](/notes/grouped-cross-validation/).)

**Check the whole stack with an outer loop.** The OOF predictions were used to fit the weights, so their error isn't a fair estimate anymore. Keep a test set aside, or wrap the whole stacking procedure in an outer cross-validation.

**Blending is the cheap cousin.** Instead of $K$-fold OOF predictions, *blending* fits the base models on one part of the data and the combiner on a separate holdout part. It's simpler and can't leak across folds, but each level gets less data.

## Key takeaways

- Stacking uses a second-level model to learn how to weight several models' predictions.
- Train the combiner only on out-of-fold predictions. In-sample predictions make it trust the model that memorizes best.
- A linear combiner with non-negative weights is a good, easy-to-read default.
- Different kinds of base models help most, and the folds have to respect any grouping in the data.
- Measure the stack's error on data that wasn't used to fit either level.

## Resources

- David H. Wolpert, "Stacked generalization", *Neural Networks* 5(2) (1992): https://doi.org/10.1016/S0893-6080(05)80023-1
- Leo Breiman, "Stacked regressions", *Machine Learning* 24 (1996): https://doi.org/10.1007/BF00117832
- Mark J. van der Laan, Eric C. Polley, Alan E. Hubbard, "Super Learner", *Statistical Applications in Genetics and Molecular Biology* 6(1) (2007): https://doi.org/10.2202/1544-6115.1309
- Trevor Hastie, Robert Tibshirani, Jerome Friedman, *The Elements of Statistical Learning*, 2nd ed., Section 8.8 (Model Averaging and Stacking): https://hastie.su.domains/ElemStatLearn/
- Jerome H. Friedman, "Multivariate Adaptive Regression Splines", *Annals of Statistics* 19(1) (1991), source of the synthetic benchmark: https://doi.org/10.1214/aos/1176347963
- scikit-learn User Guide, "Stacked generalization": https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization
