---
title: "K-means: what the SSE tells you, why the seed matters, and how to pick k"
slug: kmeans-sse-seeds-and-choosing-k
date: 2020-02-25
topic: ML & Statistics
tags: [university-lab, k-means, clustering, model-selection, unsupervised-learning]
summary: "How k-means minimises the within-cluster sum of squared errors, why that number always drops when you add clusters, why different random starts give different answers, and what to use instead to choose k."
sources:
  - title: "732A75 Advanced Data Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "Data Mining: Concepts and Techniques, 3rd ed., Chapter 10 (Cluster Analysis: Basic Concepts and Methods)"
    author: "Jiawei Han, Micheline Kamber, Jian Pei"
    type: book
  - title: "k-means++: The Advantages of Careful Seeding"
    author: "David Arthur, Sergei Vassilvitskii"
    url: "https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf"
    type: paper
  - title: "scikit-learn User Guide: K-means"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/clustering.html#k-means"
    type: docs


status: draft
---

K-means is usually the first clustering method people try, and it's easy to run. The harder part is reading the output. The software prints a "within-cluster sum of squared errors", and it's tempting to treat that like a test score where lower is better. It isn't one, and the first time I ran k-means in Weka I read it the wrong way.

## What k-means minimises

You pick the number of clusters $k$. K-means then looks for $k$ centres and an assignment of every point to one centre so that this total is as small as possible:

$$
\text{SSE} = \sum_{j=1}^{k} \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2
$$

Here $C_j$ is the set of points in cluster $j$, $\mu_j$ is the mean (centroid) of those points, and $\lVert x_i - \mu_j \rVert^2$ is the squared Euclidean distance from point $x_i$ to its centroid. So SSE measures how tight the clusters are.

The algorithm (Lloyd's algorithm) repeats two steps until nothing changes:

1. Assign every point to its nearest centroid.
2. Move every centroid to the mean of the points assigned to it.

Each step can only lower the SSE or leave it the same, so it always stops. But it stops at a *local* minimum, which depends on where the centroids started.

## Trap 1: a lower SSE with more clusters doesn't mean "better"

In my lab report I ran the food data set with $k=2$ (SSE 5.069) and $k=5$ (SSE 2.7), and wrote that "the performance increased". That's not right. With more centres, each point can only get closer to its nearest one, so the best possible SSE never goes up as $k$ grows. With $k$ equal to the number of points, every point is its own cluster and SSE is zero. Comparing SSE across different $k$ tells you nothing about which $k$ is right.

What you can do:

- **Elbow plot.** Plot SSE against $k$ and look for the point where adding a cluster stops helping much. It's a rough visual guide, and often there's no clear elbow.
- **Silhouette score.** For each point, compare the mean distance to its own cluster ($a$) with the mean distance to the nearest other cluster ($b$): $s = (b - a)/\max(a, b)$. It's between -1 and 1, and unlike SSE it doesn't simply improve with $k$.
- **Usefulness.** Often the honest answer is "the $k$ whose clusters I can interpret and use".

## Trap 2: the seed changes the answer

The seed sets the random starting centroids. In the lab, $k=2$ with seed 10 gave SSE 5.069, and $k=2$ with seed 20 gave 6.45 and clusters that looked wrong on the scatter plot. Same data and same $k$, but a different local minimum.

The SSE comparison works here, because $k$ is the same: for a fixed $k$, the run with the lower SSE found a better solution to the same problem. That's why libraries run k-means several times from different starts and keep the best run (`n_init` in scikit-learn). Smarter starts help too. **k-means++** picks the first centre at random and then picks each new centre with probability proportional to its squared distance from the nearest centre already chosen, which spreads the starting points out.

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=300, centers=6, cluster_std=0.8, random_state=3)
X = StandardScaler().fit_transform(X)       # k-means uses distances: scale first

# SSE (inertia_) keeps dropping as k grows, even past the true k = 6
for k in range(1, 10):
    print(k, round(KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_, 1))

# One random start per run: some seeds land in a worse local minimum
for seed in range(6):
    km = KMeans(n_clusters=6, n_init=1, init="random", random_state=seed).fit(X)
    print("seed", seed, round(km.inertia_, 1))   # 36.3 for seeds 0 and 5, about 13.7 for the others
```

## Trap 3: scale matters

K-means uses Euclidean distance, so a feature with large numbers (calories, in the food data) would decide the clusters alone. Weka's `SimpleKMeans` normalises numeric attributes by default, which is why the SSE values in my report were small numbers like 5.069. When you use another library, scale the features yourself.

## Reading the output correctly

One more slip from my report: Weka prints the number of points next to each cluster, and I described "27, 9 and 18" as the centroids. Those are the cluster sizes (27 foods in total, split 9 and 18). The centroids are the per-attribute means listed below them.

## Key takeaways

- K-means minimises SSE, the sum of squared distances from each point to its cluster mean.
- SSE always drops as $k$ grows, so it can't pick $k$ by itself. Use an elbow plot, silhouette scores or how useful the clusters are.
- For a fixed $k$, different seeds give different local minima. Run several starts (or use k-means++) and keep the lowest SSE.
- Scale the features first, because distance-based methods are dominated by large-valued features.

## Resources

- 732A75 Advanced Data Mining, Linköping University (course lab on clustering in Weka).
- Han, Kamber and Pei, *Data Mining: Concepts and Techniques*, 3rd ed., Chapter 10.
- Arthur and Vassilvitskii, [k-means++: The Advantages of Careful Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf), 2007.
- [scikit-learn User Guide: K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means).
