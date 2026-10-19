---
title: "Tabular Q-learning: the update, epsilon-greedy exploration, discounting, and slippery moves"
slug: tabular-q-learning
date: 2020-10-04
topic: Machine Learning
tags: [university-lab, reinforcement-learning, q-learning, exploration, temporal-difference, reinforce]
summary: "How Q-learning learns action values from experience with one temporal-difference update, what epsilon and gamma change in the learned policy, why a slippery environment changes what 'optimal' means, and where a policy network generalises better than a table."
sources:
  - title: "732A96 Advanced Machine Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Reinforcement Learning: An Introduction (2nd ed.), Chapter 6 (temporal-difference learning) and Chapter 13 (policy gradient)"
    author: "Richard S. Sutton, Andrew G. Barto"
    url: "http://incompleteideas.net/book/the-book-2nd.html"
    type: book
  - title: "Q-learning"
    author: "Christopher J. C. H. Watkins, Peter Dayan"
    url: "https://doi.org/10.1007/BF00992698"
    type: paper


status: draft
---

Q-learning is the "hello world" of reinforcement learning. Picture an agent wandering around a grid. Some cells give rewards, and it slowly learns which way to go from each cell. In the third Advanced Machine Learning lab I coded the greedy and $\epsilon$-greedy policies and the Q-learning update myself. Then I played with three knobs: the exploration rate, the discount factor and a "slip" probability. The slip part tripped me up. I called the agent's behaviour a failure when it was actually the right answer, and that changed how I think about the word *optimal*.

## Action values and the update

$Q(s, a)$ is the expected discounted return from taking action $a$ in state $s$ and acting well afterwards. The discounted return is

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots,
$$

where $r$ are rewards and $\gamma \in [0, 1)$ is the discount factor. Q-learning keeps a table of $Q(s,a)$ and, after each step $s \xrightarrow{a} s'$ with reward $r$, nudges one entry:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\,\underbrace{\big[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\big]}_{\text{temporal-difference error}},
$$

where $\alpha$ is the learning rate. The bracket compares the current guess $Q(s,a)$ with a slightly better-informed guess: the reward you actually got plus the discounted value of the best action in the next state. At a terminal state there's no future, so the target is just $r$.

The $\max$ is what makes Q-learning *off-policy*: it learns the value of acting greedily even while the agent behaves more randomly. Adding up the sizes of the TD errors over an episode is a handy convergence signal; it shrinks as the table settles.

## Exploration: $\epsilon$-greedy

If the agent always picks the action with the highest current $Q$, it keeps doing whatever worked first and never finds better options. $\epsilon$-greedy fixes that: with probability $\epsilon$ pick a random action, otherwise the greedy one (breaking ties at random, which matters at the start when every $Q$ is 0).

- Large $\epsilon$: lots of exploration, slower to exploit, but more of the grid gets visited and its values learned.
- Small $\epsilon$: the agent sticks to the first decent route; some cells are never visited and their arrows in the policy plot are meaningless.

## Discounting: short-sighted vs far-sighted

$\gamma$ sets how much a reward $k$ steps away is worth: $\gamma^k$ of its face value. In the lab's grid with a reward of 5 nearby and 10 further away, a small $\gamma$ made the agent settle for the 5, and a large $\gamma$ made it go for the 10. A 1-D corridor shows the same thing in a few lines:

```r
reward <- c(0, 2, 0, 0, 0, 0, 10); terminal <- reward != 0   # start in cell 3
run_q <- function(gamma, epsilon = 0.3, alpha = 0.1, episodes = 5000) {
  Q <- matrix(0, 7, 2)                                      # actions: left, right
  for (ep in 1:episodes) {
    s <- 3
    repeat {
      best <- which(Q[s, ] == max(Q[s, ])); best <- best[sample(length(best), 1)]
      a  <- if (runif(1) < epsilon) sample(2, 1) else best  # epsilon-greedy
      s2 <- min(max(s + c(-1, 1)[a], 1), 7); r <- reward[s2]
      target <- r + if (terminal[s2]) 0 else gamma * max(Q[s2, ])
      Q[s, a] <- Q[s, a] + alpha * (target - Q[s, a])       # TD update
      s <- s2
      if (terminal[s]) break
    }
  }
  c("left", "right")[apply(Q[3:6, ], 1, which.max)]         # greedy action in cells 3-6
}
set.seed(9)
run_q(0.5)    # "left" "right" "right" "right": from cell 3 the nearby +2 wins
run_q(0.95)   # "right" "right" "right" "right": the +10 is worth the walk
```

From cell 3 with $\gamma = 0.5$: going left earns 2 after one step; going right earns $10 \times 0.5^3 = 1.25$ in present value. Left wins. With $\gamma = 0.95$ the far reward is worth $10 \times 0.95^3 \approx 8.6$.

## Slippery moves change what "optimal" means

In the last grid, the shortest route to the goal ran right next to a row of $-1$ cells. With a slip probability $\beta$, the agent moves in the intended direction with probability $1-\beta$ and sideways with probability $\beta/2$ each. As $\beta$ grew, the learned policy stopped hugging the dangerous row and took a longer detour.

In my report I wrote that with a large $\beta$ "the agent did not learn the optimum path". That's the wrong way round. The optimal policy is defined *for the environment as it is*, slips included. When moves are unreliable, the short route has a high chance of a $-1$, and with $\gamma < 1$ a detour can have higher expected return. The detour is the optimal policy for the slippery world, and Q-learning found it. What would be wrong is to judge it against the deterministic-world shortest path. (My intuition at the time, "first avoid hazards, then go to the goal", was pointing at the right thing, I just labelled the result as a failure.)

## When a table isn't enough

In the second part of the lab the goal moved: each episode had a random goal cell, and the agent was told where it was. A Q-table indexed only by position can't handle that; you'd need a separate table per goal, and nothing learned for one goal transfers to another. Instead we trained a small neural network policy with REINFORCE (a policy-gradient method): the network takes (position, goal) as input and outputs action probabilities, and its weights are pushed towards actions that led to reaching the goal.

Trained on goals spread over the whole grid, the network also did well on goals it had never seen. Trained only on goals in the top row, it did badly on goals lower down, especially in the far corner. A parameterised policy can generalise, but only within the range of situations it was trained on: the same train/test coverage issue as in supervised learning.

## Key takeaways

- Q-learning update: move $Q(s,a)$ towards $r + \gamma \max_{a'}Q(s',a')$ by a step $\alpha$.
- $\epsilon$ trades exploration for exploitation; too little exploration leaves parts of the table unlearned.
- $\gamma$ decides how far ahead the agent looks; small $\gamma$ prefers nearby rewards.
- With stochastic moves, the optimal policy may avoid risky shortcuts. That's optimal, not a failure.
- A tabular agent can't generalise across goals; a policy network can, if trained on representative goals.

## Resources

- Richard S. Sutton, Andrew G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., Section 6.5 (Q-learning) and Section 13.3 (REINFORCE): http://incompleteideas.net/book/the-book-2nd.html
- Christopher J. C. H. Watkins, Peter Dayan, "Q-learning", *Machine Learning* 8 (1992): https://doi.org/10.1007/BF00992698
- Ronald J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning", *Machine Learning* 8 (1992): https://doi.org/10.1007/BF00992696
- 732A96 Advanced Machine Learning, Linköping University (course lab, autumn 2020).
