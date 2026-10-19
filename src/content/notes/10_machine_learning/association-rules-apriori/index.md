---
title: "Association rules with Apriori: support, confidence, lift, and describing clusters"
slug: association-rules-apriori
date: 2020-03-14
topic: Machine Learning
tags: [university-lab, association-rules, apriori, frequent-itemsets, clustering, data-mining]
summary: "What support, confidence and lift measure, how Apriori finds frequent itemsets, and how rules can describe clusters or recover a class definition that distance-based clustering misses."
sources:
  - title: "732A75 Advanced Data Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "Fast Algorithms for Mining Association Rules"
    author: "Rakesh Agrawal, Ramakrishnan Srikant"
    url: "https://www.vldb.org/conf/1994/P487.PDF"
    type: paper
  - title: "Data Mining: Concepts and Techniques, 3rd ed., Chapter 6 (Mining Frequent Patterns, Associations, and Correlations)"
    author: "Jiawei Han, Micheline Kamber, Jian Pei"
    type: book
  - title: "The MONK's Problems: A Performance Comparison of Different Learning Algorithms"
    author: "Sebastian Thrun et al."
    url: "https://archive.ics.uci.edu/dataset/70/monk+s+problems"
    type: paper


status: draft
---

I first met association rules as a shopping-basket thing: "people who buy bread and butter often buy milk too". Then in two data mining labs we used them on ordinary tables, and that surprised me. The trick is simple: treat every attribute value as an "item". We used rules for two jobs. First, to describe clusters found by k-means in plain words. Second, to find the pattern behind a class that clustering couldn't find at all. That second one taught me the most.

## Items, itemsets and rules

Each row is a set of items, like `{petallength=low, petalwidth=low, cluster=3}`. A rule has the form $X \Rightarrow Y$, where $X$ and $Y$ are itemsets with no items in common. Three numbers describe a rule:

$$
\text{supp}(X \Rightarrow Y) = P(X \cup Y), \qquad \text{conf}(X \Rightarrow Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}, \qquad \text{lift}(X \Rightarrow Y) = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)}
$$

- $P(X \cup Y)$ is the fraction of rows that contain *all* items of $X$ and $Y$ (the union of the item sets, which means "both happen").
- **Support** says how common the pattern is.
- **Confidence** is the estimated $P(Y \mid X)$: when $X$ shows up, how often $Y$ comes with it.
- **Lift** compares that to how common $Y$ is anyway. Lift 1 means $X$ tells you nothing about $Y$. Lift above 1 means $X$ makes $Y$ more likely.

Confidence alone can mislead. If 90% of all rows have $Y$, then almost any $X \Rightarrow Y$ has confidence near 0.9. Lift catches that.

## How Apriori finds the rules

Checking every possible itemset is hopeless, because there are $2^m$ of them for $m$ items. Apriori uses one fact: **if an itemset is frequent, all its subsets are frequent too**. Turned around, if `{a, b}` is rare, then `{a, b, c}` must be rare as well, so you never need to count it. Apriori grows itemsets level by level (size 1, then 2, and so on), keeps only those above the minimum support, and builds the next level only from survivors. Then it splits each frequent itemset into rules and keeps those above the minimum confidence.

Weka's version adds a loop around this. It starts with minimum support 1.0 and lowers it by a step (delta, 0.05 in our runs) until it has found the requested number of rules or hits a lower bound. So the "minimum support" you set is really a floor.

## Using rules to describe clusters

In the Iris lab we discretised the four measurements into equal-width bins, ran k-means with 3 clusters, added the cluster label as a new attribute, and asked Apriori for rules with `cluster=...` on the right. With 3 bins and 3 clusters, rules like

`petallength='(-inf-2.966667]'  50 ==> cluster=cluster3  50   conf:(1)`

read like a plain-language description of each cluster: short petals means cluster 3, every time.

A correction to our group report: we explained the support ceiling of these rules with "because we only have 3 bins, the maximum number of each interval can't exceed 50". Equal-width bins don't hold equal counts, so that's not the reason. The real limit is the right-hand side. A rule $X \Rightarrow \{\text{cluster}=c\}$ can't have more support than the cluster itself, $|C_c|/n$. A cluster of 50 flowers caps it at $50/150$, and the 4-point cluster in our 4-cluster run caps it at $4/150$. (Our report does use this argument for the 4-cluster run and the 5-bin run.)

## When clustering can't find the classes

The MONK-1 data set has six small categorical attributes and a class. Its class is defined by a logical rule:

$$
\text{class} = 1 \iff (a_1 = a_2) \;\text{or}\; (a_5 = 1)
$$

We clustered it with k-means (2 and 3 clusters), EM and a density-based clusterer. None of them matched the classes. That was frustrating, because the rule is so simple. Then we tried Apriori. With Apriori (minimum support 0.05), the top rules for class 1 were

- `attribute#5=1  29 ==> class=1  29  conf:(1)`
- `attribute#1=3 attribute#2=3  17 ==> class=1  17  conf:(1)`
- `attribute#1=2 attribute#2=2  15 ==> class=1  15  conf:(1)`

which are exactly pieces of the true definition.

In our report we explained the clustering failure by saying the algorithms "use numeric-based measurements". That's not quite right. Weka's k-means handles nominal attributes directly: the distance for a nominal attribute is 0 if the values match and 1 if not, and the cluster "centroid" uses the most common value. The real reason is that clustering is unsupervised. It groups rows that are similar overall, and nothing tells it that the class depends on $a_1$ *equalling* $a_2$. Two rows with $a_1 = a_2 = 1$ and $a_1 = a_2 = 3$ are both class 1 but differ in two attributes. A row with $a_1 = 1, a_2 = 2$ is class 0 but is only one mismatch away from the first. The class is a logical rule, not a tight blob, so "similar" and "same class" don't line up. Rules work because they look for value combinations that always come with class 1.

Here's a small Python version that computes support, confidence and lift by hand on simulated MONK-1-style data:

```python
import random

random.seed(0)
rows = []
for _ in range(400):
    a1, a2, a5 = random.randint(1, 3), random.randint(1, 3), random.randint(1, 4)
    cls = int(a1 == a2 or a5 == 1)                  # the MONK-1 rule
    rows.append({f"a1={a1}", f"a2={a2}", f"a5={a5}", f"class={cls}"})

def support(items):
    return sum(items <= r for r in rows) / len(rows)  # share of rows containing all items

def rule(lhs, rhs):
    s_all = support(lhs | rhs)
    conf = s_all / support(lhs)
    return s_all, conf, conf / support(rhs)          # support, confidence, lift

for lhs in [{"a5=1"}, {"a1=3", "a2=3"}, {"a1=3"}]:
    s, c, l = rule(lhs, {"class=1"})
    print(sorted(lhs), f"supp={s:.2f} conf={c:.2f} lift={l:.2f}")
# a5=1: conf 1.00, lift 1.90 | a1=3 & a2=3: conf 1.00, lift 1.90 | a1=3 alone: conf 0.53, lift 1.01
```

`a1=3` alone has lift about 1. It says nothing until you pair it with `a2=3`.

## Key takeaways

- Support is how common a pattern is, confidence is $P(Y \mid X)$, and lift compares that confidence with how common $Y$ is anyway.
- Apriori prunes the search with one rule: every subset of a frequent itemset is frequent.
- Adding a cluster label as an item turns rules into readable cluster descriptions. A rule's support can't be bigger than the support of its right-hand side.
- Clustering finds overall similarity, not your classes. When a class follows a logical rule, association rules (or a classifier) can find it and clustering usually can't.

## Resources

- 732A75 Advanced Data Mining, Linköping University (course labs on association analysis in Weka; group work).
- Agrawal and Srikant, [Fast Algorithms for Mining Association Rules](https://www.vldb.org/conf/1994/P487.PDF), VLDB 1994.
- Han, Kamber and Pei, *Data Mining: Concepts and Techniques*, 3rd ed., Chapter 6.
- Thrun et al., The MONK's Problems, 1991. Data set: [UCI MONK's Problems](https://archive.ics.uci.edu/dataset/70/monk+s+problems).
