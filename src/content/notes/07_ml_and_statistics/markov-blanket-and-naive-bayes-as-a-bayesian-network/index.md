---
title: "Classifying with a Bayesian network: the Markov blanket, and naive Bayes as a tiny BN"
slug: markov-blanket-and-naive-bayes-as-a-bayesian-network
date: 2020-09-13
topic: ML & Statistics
tags: [university-lab, bayesian-networks, markov-blanket, naive-bayes, exact-inference, d-separation]
summary: "How a Bayesian network classifies by computing the posterior of one node given the others, why only that node's Markov blanket matters, and how naive Bayes is just one particular (usually wrong) network structure."
sources:
  - title: "732A96 Advanced Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Networks with Examples in R (2nd ed.)"
    author: "Marco Scutari, Jean-Baptiste Denis"
    url: "https://www.bnlearn.com/book-crc-2ed/"
    type: book
  - title: "Pattern Recognition and Machine Learning, Chapter 8 (graphical models)"
    author: "Christopher M. Bishop"
    url: "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/"
    type: book


status: draft
---

Once you have a Bayesian network, any node can be the "target". To classify it, you compute its posterior given whatever else you observed and pick the most probable value. In the first Advanced Machine Learning lab we did this for the "smoking" node of the Asia network with `bnlearn` and `gRain`. Two ideas from it are worth keeping: the Markov blanket, and seeing naive Bayes as just another graph. Plus a bug that made one of my conclusions wrong.

## Classification is inference

Write the target as $S$ and the observed variables as $e$ (the evidence). The classifier is

$$
\hat s = \arg\max_s\; p(S = s \mid e), \qquad p(s \mid e) = \frac{p(s, e)}{\sum_{s'} p(s', e)},
$$

and $p(s, e)$ comes from the network's factorisation $\prod_j p(x_j \mid \text{pa}(x_j))$, summing out any variables that weren't observed. `gRain` does this exactly with the junction-tree algorithm: it "compiles" the network (moralises it, triangulates it, builds a tree of cliques) so that queries are fast. For a small network you can also just enumerate the joint table, which is what the snippet below does.

## The Markov blanket

The **Markov blanket** of a node is its parents, its children, and its children's other parents. The key fact:

$$
p(S \mid \text{everything else}) = p(S \mid \text{MB}(S)).
$$

Once you know the blanket, no other variable adds information about $S$. The children's other parents are in there because of v-structures: if $S \to C \leftarrow P$ and you observe $C$, then $P$ becomes informative about $S$ ("explaining away").

In the Asia network, smoking has two children (lung cancer and bronchitis) and no parents, and those children have no other parents. So its Markov blanket is just those two nodes. That's why, in the lab, classifying smoking from its blanket alone gave exactly the same confusion matrix as using all seven other variables (error rate 0.284 for both, and the same for the true network).

But the blanket is "enough" only when it's *fully observed*. If one blanket variable is missing, other variables can carry information about $S$ through it. The snippet shows both cases:

```r
# Tiny BN: S -> L, S -> B, (L, B) -> D.  Markov blanket of S = {L, B}
grid <- expand.grid(S = 0:1, L = 0:1, B = 0:1, D = 0:1)
pS <- function(s) ifelse(s == 1, 0.5, 0.5)
pL <- function(l, s) { p <- ifelse(s == 1, 0.10, 0.01); ifelse(l == 1, p, 1 - p) }
pB <- function(b, s) { p <- ifelse(s == 1, 0.60, 0.30); ifelse(b == 1, p, 1 - p) }
pD <- function(d, l, b) { p <- 0.1 + 0.6 * l + 0.2 * b; ifelse(d == 1, p, 1 - p) }
grid$p <- with(grid, pS(S) * pL(L, S) * pB(B, S) * pD(D, L, B))   # joint table

posterior_S <- function(evidence) {
  keep <- Reduce(`&`, Map(function(v, val) grid[[v]] == val, names(evidence), evidence))
  rows <- grid[keep, ]
  tapply(rows$p, rows$S, sum) / sum(rows$p)                         # p(S | evidence)
}
posterior_S(list(L = 0, B = 1, D = 1))   # everything observed:     0.355 / 0.645
posterior_S(list(L = 0, B = 1))          # blanket only, same:      0.355 / 0.645
posterior_S(list(B = 1, D = 1))          # L unknown, D now matters: 0.298 / 0.702
```

My report explained the Markov blanket result with "S is a common parent of L and B, so given L and B all paths are blocked". That's the right idea (d-separation), but it's worth saying the general rule: the blanket includes co-parents of children too, and it's only sufficient when all of it is observed.

## Naive Bayes is a BN

Naive Bayes assumes all features are independent given the class. As a graph, that's the class node with an arrow to every feature and no other arrows:

$$
p(s, x_1, \dots, x_m) = p(s) \prod_{j=1}^m p(x_j \mid s).
$$

So you can build it by hand in `bnlearn` (`model2network("[S][A|S][T|S]...")`), fit its tables, and query it exactly like any other network. Seen this way, naive Bayes isn't a separate method. It's a BN with one fixed, very simple structure. When the true structure has dependencies among the features (in Asia, several features are chained through each other), naive Bayes counts correlated evidence more than once and its posteriors get overconfident. It also scored much worse (lower BIC) than the learned or the true network.

## The bug that broke my conclusion

In my naive Bayes loop I set the evidence on the naive Bayes network but then queried `dag_find_true`, a leftover variable from the previous section's loop. That variable held the true network with the evidence of the *last* test case. So every test case got the same answer, and the "naive Bayes" classifier predicted "no" for everyone. Its error rate of 0.466 was just the share of smokers in the test set. I then explained that number as the expected effect of the naive Bayes assumption. The explanation sounded reasonable, which is exactly why the bug went unnoticed.

Two lessons. A confusion matrix with an empty column (a class that's never predicted) is a red flag. And when copying a loop, rename every variable inside it, or better, wrap the loop in a function that takes the compiled network as an argument, so there's no leftover state to grab by mistake.

Also a wording fix: I wrote that naive Bayes gives a different "marginal distribution" of $S$. What differs is the *conditional* (posterior) distribution $p(S \mid e)$; the marginal $p(S)$ is basically the same in all the fitted networks, since it's estimated from the same data.

## Key takeaways

- Classifying with a BN = computing $p(\text{target} \mid \text{evidence})$ exactly and taking the arg max.
- Markov blanket = parents + children + children's other parents; given all of it, the rest is irrelevant.
- If part of the blanket is unobserved, other variables can matter again.
- Naive Bayes is a BN where the class points to every feature; it double-counts correlated features.
- A classifier that never predicts one class is a bug signal, not a finding.

## Resources

- Marco Scutari, Jean-Baptiste Denis, *Bayesian Networks with Examples in R*, 2nd ed., Chapter 1 (inference) and Section 2.4: https://www.bnlearn.com/book-crc-2ed/
- Christopher M. Bishop, *Pattern Recognition and Machine Learning*, Section 8.2 (d-separation, Markov blanket): https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
- Judea Pearl, *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann, 1988 (Markov blankets).
- gRain package documentation: https://cran.r-project.org/package=gRain
- 732A96 Advanced Machine Learning, Linköping University (course lab, autumn 2020).
