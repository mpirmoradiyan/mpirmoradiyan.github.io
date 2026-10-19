---
title: "Learning a Bayesian network's structure with hill climbing, and what 'equivalent' graphs are"
slug: bayesian-network-structure-learning
date: 2020-09-13
topic: Bayesian Methods
tags: [university-lab, bayesian-networks, structure-learning, graphical-models, hill-climbing, bnlearn]
summary: "How score-based structure learning searches over DAGs, why hill climbing can end in different graphs, and how to tell whether two learned graphs really say different things (Markov equivalence)."
sources:
  - title: "732A96 Advanced Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Bayesian Networks with Examples in R (2nd ed.)"
    author: "Marco Scutari, Jean-Baptiste Denis"
    url: "https://www.bnlearn.com/book-crc-2ed/"
    type: book
  - title: "Probabilistic Graphical Models: Principles and Techniques, Chapter 18 (structure learning)"
    author: "Daphne Koller, Nir Friedman"
    type: book
  - title: "bnlearn documentation"
    author: "Marco Scutari"
    url: "https://www.bnlearn.com/documentation/"
    type: docs


status: draft
---

A Bayesian network (BN) is a directed acyclic graph (DAG) where each node is a variable and the arrows say which variables directly influence which. Given data, you can try to learn the graph itself. The first Advanced Machine Learning lab used the classic eight-variable "Asia" medical example and the R package `bnlearn`. The question that tripped me up was simple: if two runs of the learning algorithm return different graphs, are they actually different models?

## What a BN encodes

A DAG over variables $X_1, \dots, X_d$ says the joint distribution factorises as

$$
p(x_1, \dots, x_d) = \prod_{j=1}^{d} p\big(x_j \mid \text{pa}(x_j)\big),
$$

where $\text{pa}(x_j)$ are the parents of node $j$ (the nodes with arrows into it). Missing arrows are claims of conditional independence. That's the real content of the graph.

## Score-based learning and hill climbing

The number of possible DAGs grows super-exponentially with the number of nodes, so you can't try them all. Score-based learning picks a score that rewards fit and penalises complexity, for example BIC:

$$
\text{BIC}(G) = \log p(D \mid \hat\theta_G, G) - \frac{k_G}{2}\log n,
$$

where $D$ is the data, $\hat\theta_G$ the maximum-likelihood parameters (conditional probability tables) for graph $G$, $k_G$ the number of free parameters and $n$ the number of rows. Another common choice is BDeu, a Bayesian marginal likelihood with a Dirichlet prior controlled by an "imaginary sample size".

Hill climbing then does greedy local search: start from some graph (by default the empty one), try every single-arc change (add, delete, reverse) that keeps the graph acyclic, apply the one that improves the score most, and repeat until no change helps. It stops at a **local** optimum.

## Why runs can end in different graphs

Plain hill climbing from the same start on the same data is deterministic. Different results come from:

- **different starting graphs**,
- **random restarts**: after converging, perturb the graph with a few random arc changes and climb again,
- **different scores or score settings** (BIC vs BDeu, or BDeu with different imaginary sample sizes),
- **different data**, for example a different subsample.

Because the search is greedy, these can end at different local optima with different scores. Those are genuinely different models.

## Different-looking isn't always different: Markov equivalence

Some different DAGs encode exactly the same independencies. $A \to B \to C$, $A \leftarrow B \leftarrow C$ and $A \leftarrow B \to C$ all say "$A$ and $C$ are independent given $B$", and nothing else. Data can't tell them apart. Two DAGs are **Markov equivalent** exactly when they have the same skeleton (the same arcs if you ignore direction) and the same *v-structures* (patterns $A \to C \leftarrow B$ where $A$ and $B$ aren't connected). A v-structure is different: it says $A$ and $B$ are independent, but become dependent once you know $C$.

An equivalence class is drawn as a CPDAG (completed partially directed graph): arcs that every member agrees on are directed, the rest are undirected. Scores like BIC and BDeu are *score equivalent*: equivalent DAGs get exactly the same score.

So the right way to compare two learned graphs is to compare their CPDAGs, not their arrows:

```r
library(bnlearn)
data(asia)
g1 <- hc(asia)                                     # from the empty graph
g2 <- hc(asia, start = random.graph(names(asia)), restart = 10, perturb = 3)
g3 <- hc(asia, score = "bde", iss = 50)            # different score settings

all.equal(cpdag(g1), cpdag(g2))    # TRUE only if they're Markov equivalent
all.equal(cpdag(g1), cpdag(g3))
sapply(list(g1, g2, g3), score, data = asia)       # BIC of each (default score)
vstructs(g1)                                        # the v-structures that fix directions
```

## What I got wrong

In my report I ran hill climbing six times (three from the empty graph, three from a hand-made graph), all with restarts, and all six got exactly the same score. I then wrote that the runs gave "different" graphs that nevertheless encode "the same independencies". That's self-contradictory as an answer to the question, which was about *non-equivalent* graphs. Identical scores under BIC strongly suggest my six graphs were the same or Markov equivalent, so the experiment didn't show what it was supposed to. To actually see non-equivalent results you need to vary something that changes where the greedy search ends: different scores or imaginary sample sizes, random starting graphs, or different subsamples. Then check with `cpdag()` and the scores.

I also wrote that hill climbing "initialises at a random state". It doesn't, by default: it starts from the empty graph, and randomness only enters through restarts or a random start you supply.

## A point about the Asia data

The Asia network contains a node ("either tuberculosis or lung cancer") that's a deterministic OR of two other nodes. Deterministic relationships break some of the assumptions that structure learning relies on, so it's normal for learned graphs to differ from the true one in that corner. Our learned graph and the true graph had almost the same BIC on the training data, and they gave identical classification results on the test set.

## Key takeaways

- A BN's content is its set of conditional independencies; missing arcs are the claims.
- Hill climbing is greedy and finds local optima; different starts, restarts, scores or data can give different graphs.
- Graphs with the same skeleton and the same v-structures are Markov equivalent and indistinguishable from data.
- Compare learned graphs via their CPDAGs and scores, not by looking at arrow directions.

## Resources

- Marco Scutari, Jean-Baptiste Denis, *Bayesian Networks with Examples in R*, 2nd ed., Chapters 1-2: https://www.bnlearn.com/book-crc-2ed/
- Daphne Koller, Nir Friedman, *Probabilistic Graphical Models*, Section 3.3.4 (I-equivalence) and Chapter 18 (structure learning), MIT Press, 2009.
- bnlearn documentation, `hc`, `cpdag`, `vstructs`: https://www.bnlearn.com/documentation/
- Steffen L. Lauritzen, David J. Spiegelhalter, "Local computations with probabilities on graphical structures and their application to expert systems", *JRSS B* 50(2) (1988), origin of the Asia example: https://doi.org/10.1111/j.2517-6161.1988.tb01721.x
- 732A96 Advanced Machine Learning, Linköping University (course lab, autumn 2020).
