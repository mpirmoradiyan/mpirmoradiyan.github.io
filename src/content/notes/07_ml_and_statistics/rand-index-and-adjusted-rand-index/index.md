---
title: "Rand index vs. adjusted Rand index: comparing a clustering with gold labels"
slug: rand-index-and-adjusted-rand-index
date: 2020-11-22
topic: ML & Statistics
tags: [university-lab, clustering, evaluation-metrics, rand-index, text-clustering, k-means]
summary: "How the Rand index scores a clustering as a classifier on pairs of documents, why it rises with the number of clusters even for random labels, and how the adjusted Rand index corrects for chance."
sources:
  - title: "732A92 Text Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "Objective Criteria for the Evaluation of Clustering Methods"
    author: "William M. Rand"
    url: "https://doi.org/10.1080/01621459.1971.10482356"
    type: paper
  - title: "Comparing partitions"
    author: "Lawrence Hubert, Phipps Arabie"
    url: "https://doi.org/10.1007/BF01908075"
    type: paper
  - title: "scikit-learn User Guide: Rand index"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/clustering.html#rand-index"
    type: docs


status: draft
---

After clustering product reviews with k-means, the text mining lab asked: if we have the true product categories for some reviews, how do we score the clustering? Cluster numbers are arbitrary ("cluster 2" isn't "books"), so you can't just compute accuracy. The Rand index gets around that by looking at **pairs** of documents. I used the adjusted version in the lab without thinking much about the difference, and the difference turns out to be the whole point.

## The Rand index: a classifier on pairs

Take every unordered pair of documents. For each pair, the clustering makes a yes/no claim: "these two are in the same cluster". The gold labels give the true answer: "these two have the same category". Count:

- $a$: pairs in the same cluster **and** the same category (true positives)
- $b$: pairs in different clusters **and** different categories (true negatives)

The Rand index is the accuracy of this pair classifier:

$$
\text{RI} = \frac{a + b}{\binom{n}{2}}
$$

where $n$ is the number of documents and $\binom{n}{2} = n(n-1)/2$ is the number of pairs. RI is between 0 and 1, and it doesn't care what the clusters are called. Renaming them changes nothing.

So the mapping I wrote in the lab (cluster 0 → "book", cluster 1 → "camera", and so on) before scoring wasn't needed. Any labels give the same score.

## The problem: RI rewards many clusters

With 6 categories, most pairs of documents have *different* categories. A clustering with many small clusters says "different" for almost every pair, and it's right most of the time. So RI climbs as $k$ grows, even when the clusters are pure noise. The same thing happens with accuracy on imbalanced classes: "different" is the majority class.

```python
import numpy as np
from sklearn.metrics import rand_score, adjusted_rand_score

rng = np.random.default_rng(0)
gold = rng.integers(0, 6, size=500)          # 6 categories, like the review data

for k in [1, 3, 5, 7, 20, 100]:
    pred = rng.integers(0, k, size=500)      # random clusters: no information at all
    print(k, round(rand_score(gold, pred), 3), round(adjusted_rand_score(gold, pred), 3))
# k=1: RI 0.167, ARI 0.0  ...  k=7: RI 0.739  ...  k=100: RI 0.826, ARI about 0
```

Random clusterings get an RI of 0.83 with 100 clusters. That's not a usable scale.

## The adjusted Rand index

The ARI (Hubert and Arabie) subtracts what you'd expect by chance and rescales:

$$
\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
$$

$\mathbb{E}[\text{RI}]$ is the expected RI if the documents were assigned to clusters at random while keeping the same cluster sizes and category sizes. So ARI is about 0 for random clusterings at any $k$, 1 for a perfect match, and can go slightly negative when a clustering is worse than chance. In the simulation above it stays at about 0 for every $k$, while RI goes from 0.17 to 0.83.

## What our numbers said

We fit k-means on all 11,914 reviews (tf-idf vectors) and scored the first 500 against their categories (music, books, dvd, camera, health, software). The ARI went:

| $k$ | 1 | 3 | 5 | 7 |
|---|---|---|---|---|
| ARI | 0.0 | 0.080 | 0.219 | 0.301 |

$k = 1$ gives exactly 0 by construction: one cluster carries no information. Since ARI corrects for chance, the rise from 3 to 7 means the clusters really matched the categories better, not just that there were more of them. That fits the cluster summaries (the top terms of each centroid). With $k=3$, books and cameras got clean clusters and everything else landed in one mixed cluster. With $k = 5$ to $7$, music, movies and software split off as well.

The lab asked for the plain (unadjusted) Rand index. It would probably have gone up with $k$ too, but on its own that rise wouldn't prove anything, because, as the simulation shows, RI rises with $k$ even for random clusters.

## Key takeaways

- The Rand index is the accuracy of "same cluster?" vs. "same class?" over all pairs. It ignores cluster names.
- RI rises with the number of clusters even for random clusterings, because most pairs are "different".
- The adjusted Rand index removes the chance level: about 0 for random, 1 for perfect. Use it to compare clusterings with different $k$.
- No need to map cluster IDs to category names before computing either index.

## Resources

- 732A92 Text Mining, Linköping University (text clustering lab).
- Rand, [Objective Criteria for the Evaluation of Clustering Methods](https://doi.org/10.1080/01621459.1971.10482356), JASA, 1971.
- Hubert and Arabie, [Comparing partitions](https://doi.org/10.1007/BF01908075), Journal of Classification, 1985.
- [scikit-learn: Rand index](https://scikit-learn.org/stable/modules/clustering.html#rand-index).
- Related note: [k-means: SSE, seeds and choosing k](/notes/kmeans-sse-seeds-and-choosing-k/).
