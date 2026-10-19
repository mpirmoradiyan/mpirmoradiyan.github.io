---
title: "The 0/1 knapsack problem three ways: brute force, dynamic programming, greedy"
slug: knapsack-brute-force-dp-greedy
date: 2019-10-13
topic: Programming & Data
tags: [university-lab, algorithms, dynamic-programming, complexity, r, profiling]
summary: "How the same optimization problem goes from exponential to pseudo-polynomial to n log n time, what you give up at each step, and how to read the chosen items back out of a dynamic programming table."
sources:
  - title: "732A94 Advanced R Programming (course lab)"
    author: "Linköping University"
    type: course
  - title: "Knapsack problem"
    author: "Wikipedia"
    url: "https://en.wikipedia.org/wiki/Knapsack_problem"
    type: docs
  - title: "Introduction to Algorithms, 3rd ed., Chapter 15 (Dynamic Programming) and Section 16.2 (greedy vs dynamic programming)"
    author: "Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein"
    type: book


status: draft
---

The knapsack problem is where "exponential time" stops being a phrase and turns into seconds on a clock. In my Advanced R course we wrote an R package (a pair project) with three solvers for the knapsack problem, and timed them. The problem is easy to say. You have $n$ items. Item $i$ has weight $w_i$ and value $v_i$. Your bag holds total weight $W$. Pick the set of items with the largest total value that still fits. The three solvers make a nice ladder: exact but slow, exact and much faster under one condition, and fast but only approximate.

Written out, with $x_i \in \{0, 1\}$ meaning "item $i$ is in the bag":

$$
\max_{x} \sum_{i=1}^{n} v_i x_i \quad \text{subject to} \quad \sum_{i=1}^{n} w_i x_i \le W.
$$

## 1. Brute force: try every subset

The simplest idea: there are $2^n$ possible subsets, so check each one and keep the best that fits. That's $O(2^n)$ subsets (times up to $n$ work each). It's always correct. A neat trick for listing all the subsets: the binary digits of the integers $1, \dots, 2^n - 1$ are exactly all the non-empty subsets (bit $j$ set = item $j$ included). In R, `intToBits(i)` gives you those bits.

The catch is how fast $2^n$ grows. With 16 items our R code already took about a second, and every extra item doubles that.

## 2. Dynamic programming: reuse answers to smaller problems

The idea here is to stop solving the same small problem again and again. If the weights are integers, let $m[i, c]$ be the best value you can get using only the first $i$ items with capacity $c$. Now think about item $i$. Either you leave it out, or you put it in:

$$
m[i, c] = \begin{cases} m[i-1, c] & \text{if } w_i > c,\\ \max\big(m[i-1, c],\ m[i-1, c - w_i] + v_i\big) & \text{otherwise,}\end{cases}
$$

with $m[0, c] = 0$ for all $c$. The answer is $m[n, W]$. The table has $(n+1)(W+1)$ cells, and each one takes constant time to fill, so the cost is $O(nW)$.

One subtle point that our report got slightly wrong: $O(nW)$ is called **pseudo-polynomial**, not polynomial. Input size is measured in bits, and $W$ only takes about $\log_2 W$ bits to write down. Add one more bit to $W$ and the table doubles. So the running time is exponential in the length of the input. In practice: DP is great when $W$ is small (like 2000 in our tests), and useless if, say, weights are in grams and $W$ is in the millions. Knapsack is still NP-hard.

**Reading back which items were chosen.** The table gives you the best value, but not the items. To get those, walk backwards from $(n, W)$. At row $i$ with remaining capacity $c$: if $m[i, c] \ne m[i-1, c]$, item $i$ was used, so write it down and set $c \leftarrow c - w_i$. Otherwise move up a row without taking it.

```r
knapsack_dp <- function(w, v, W) {
  n <- length(w)
  m <- matrix(0, n + 1, W + 1)          # row i+1 = first i items, col c+1 = capacity c
  for (i in 1:n) for (c in 0:W) {
    m[i + 1, c + 1] <- m[i, c + 1]
    if (w[i] <= c) m[i + 1, c + 1] <- max(m[i + 1, c + 1], m[i, c + 1 - w[i]] + v[i])
  }
  items <- c(); c <- W
  for (i in n:1) if (m[i + 1, c + 1] != m[i, c + 1]) { items <- c(i, items); c <- c - w[i] }
  list(value = m[n + 1, W + 1], elements = items)
}
knapsack_dp(w = c(3, 4, 2), v = c(4, 5, 3), W = 6)   # value 8, items 2 and 3
```

Our original version had two bugs worth remembering. The first was an off-by-one. R counts from 1 but capacity starts at 0, and our loop only filled the capacity columns up to `W` instead of `W + 1`. So the answer was really for capacity $W - 1$. Writing the "+1" out explicitly, as above, avoids that. The second bug was in the backtracking. It checked whether the current value appeared *anywhere* in the previous row (`currValue %in% m[N-1, ]`) instead of comparing with the cell directly above, at the current capacity. A value can show up in that row by coincidence, so items could be skipped or wrongly included. You have to keep track of the remaining capacity.

## 3. Greedy: sort by value per weight

The fast idea: sort items by $v_i / w_i$ (value per kilo, so to speak), highest first, and add them while they fit. The cost is $O(n \log n)$ for the sort. A million items took well under a second.

The catch: it's not exact, and the plain version can be **arbitrarily bad**. Here's a tiny example. Item A has $w = 1, v = 2$ (ratio 2). Item B has $w = W, v = W$ (ratio 1). Greedy takes A first, then B doesn't fit, so it returns 2. The best answer is $W$. Our report claimed greedy always gets at least 50% of the best value. That guarantee only holds for a small change to the algorithm: return the better of (a) the greedy solution, stopping at the first item that doesn't fit, and (b) the single most valuable item that fits on its own. That modified version is a 1/2-approximation.

## What the timing experiments taught

- **Data frames in hot loops are slow in R.** Profiling showed most of the time went into `x$w[j]` lookups inside loops. Converting to a matrix (or plain vectors) first cut the brute-force time from about 1.4 s to about 0.25 s for 16 items, and DP from about 2 s to about 0.34 s for 500 items.
- **Parallel isn't automatically faster.** Our parallel brute force with `parLapply` was slower (about 3.2 s vs 1.4 s). Starting worker processes and copying data to them cost more than the work itself at $n = 16$, and our parallel version built all $2^n$ combinations in memory. Also, we called `stopCluster()` after `return()`, so it never ran and the workers were left behind. `on.exit(stopCluster(cl))` right after creating the cluster fixes that.

## Key takeaways

- Brute force is $O(2^n)$: exact, only for tiny $n$.
- DP is $O(nW)$, pseudo-polynomial: exact and fast when integer capacities are small.
- Recover chosen items by walking the DP table backwards, comparing each cell with the one directly above and tracking the remaining capacity.
- Greedy by $v/w$ is $O(n \log n)$ but has no guarantee unless you also compare with the best single item.
- Profile before optimizing, and measure before assuming parallel helps.

## Resources

- 732A94 Advanced R Programming, Linköping University, lab 6 (2019).
- [Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem), Wikipedia: 0/1 DP and the greedy approximation.
- Cormen, Leiserson, Rivest, Stein, *Introduction to Algorithms*, 3rd ed., Chapter 15 (dynamic programming) and Section 16.2 (why greedy fails for 0/1 knapsack).
