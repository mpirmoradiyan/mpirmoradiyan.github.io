---
title: "Word embeddings: from co-occurrence counts to word2vec"
slug: word-embeddings
date: 2025-03-07
topic: NLP & LLMs
tags: [word-embeddings, word2vec, skip-gram, cbow, negative-sampling, svd, gensim]
summary: "Why short dense word vectors are better than one-hot codes, two ways to get them (counting plus SVD, or word2vec with negative sampling), and how to use pretrained vectors for similarity and analogies."
sources:
  - title: "Efficient Estimation of Word Representations in Vector Space"
    author: "Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013)"
    url: "https://arxiv.org/abs/1301.3781"
    type: paper
  - title: "Distributed Representations of Words and Phrases and their Compositionality"
    author: "Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean (2013)"
    url: "https://arxiv.org/abs/1310.4546"
    type: paper
  - title: "GloVe: Global Vectors for Word Representation"
    author: "Jeffrey Pennington, Richard Socher, Christopher Manning (2014)"
    url: "https://nlp.stanford.edu/projects/glove/"
    type: paper
  - title: "Gensim: Word2Vec model"
    author: "Radim Řehůřek"
    url: "https://radimrehurek.com/gensim/models/word2vec.html"
    type: docs


status: draft
---

Neural networks need numbers. The simplest way to turn a word into numbers is a one-hot vector: a vector as long as the vocabulary, all zeros except a single 1. The problem is that one-hot vectors are huge and say nothing about meaning. "Cat" is exactly as far from "dog" as it is from "carburettor". **Word embeddings** swap them for short, dense vectors (every number is used), learned so that words used in similar places end up close together. In these notes I build embeddings two ways, from counts and with word2vec, and then try pretrained vectors.

## The idea: you know a word by its neighbours

This idea has a fancy name, the distributional hypothesis, but it just says that words that show up in similar contexts usually mean similar things. "Cat" and "dog" both appear next to "the", "chased", "played" and "pet". So if you build each word's vector from its neighbours (or train it to predict them), words with similar neighbours get similar vectors.

To measure how similar two vectors are, people usually use **cosine similarity**:

$$
\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert},
$$

It's 1 when the vectors point the same way, 0 when they're at right angles and -1 when they point in opposite directions. It ignores how long the vectors are, which is good, because length often says more about how common a word is than what it means.

## Method 1: count which words appear together, then compress with SVD

Count how often each word shows up near each other word (within some window). With a window of 1 on three toy sentences ("the cat sat on the mat", "the dog chased the cat", "the cat and the dog played") you get a 9 x 9 symmetric matrix. Each row already describes a word as a vector, but it's as long as the vocabulary and mostly zeros.

The **singular value decomposition** $M = U \Sigma V^\top$ finds the directions that explain most of what's going on in the matrix. If you keep only the top $k$ singular values, you get the best rank-$k$ approximation of $M$, and the rows of $U_k \Sigma_k^{1/2}$ work as $k$-dimensional word vectors. So basically SVD squeezes the long sparse rows into short dense ones.

```python
import numpy as np

corpus = ["the cat sat on the mat", "the dog chased the cat",
          "the cat and the dog played"]
tokens = [s.split() for s in corpus]
vocab = sorted({w for s in tokens for w in s})
w2i = {w: i for i, w in enumerate(vocab)}

M = np.zeros((len(vocab), len(vocab)))
for s in tokens:
    for i, w in enumerate(s):
        for j in range(max(0, i - 1), min(len(s), i + 2)):   # window of 1
            if i != j:
                M[w2i[w], w2i[s[j]]] += 1

U, S, Vt = np.linalg.svd(M)
k = 2
emb = U[:, :k] * np.sqrt(S[:k])   # one k-dimensional vector per word
```

This is the heart of latent semantic analysis. In practice, raw counts are swamped by common words like "the", so the matrix is usually reweighted first, for example with a logarithm or with positive pointwise mutual information (a score for how much more often two words appear together than chance would predict). GloVe is a newer take on this counting idea: it fits word vectors so that their dot products match the log of the co-occurrence counts.

## Method 2: word2vec, learning by prediction

Word2vec trains a very small model on a guessing game, and then keeps the learned weights as the embeddings. The guessing game itself isn't the point. There are two versions:

- **Skip-gram**: given the centre word, guess each word around it.
- **CBOW** (continuous bag of words): given the surrounding words (averaged), guess the centre word.

Each word gets two vectors: $v_w$ when it's the centre (input) word and $u_c$ when it's a context (output) word. The full skip-gram model uses a softmax over the whole vocabulary,

$$
P(c \mid w) = \frac{\exp(u_c \cdot v_w)}{\sum_{c'=1}^{|V|} \exp(u_{c'} \cdot v_w)},
$$

where $|V|$ is the vocabulary size. The problem is the bottom of the fraction: it goes over every word in the vocabulary for every single training pair. With a million words that's way too slow.

### Negative sampling

**Negative sampling** turns it into a simple yes/no question. For each real (centre, context) pair, pick $K$ random "negative" words $n_1, \dots, n_K$, and train the model to say yes to the real pair and no to the fake ones:

$$
\mathcal{L} = -\log \sigma(u_c \cdot v_w) - \sum_{k=1}^{K} \log \sigma(-u_{n_k} \cdot v_w),
$$

where $\sigma$ is the sigmoid. The negatives are drawn from the word-frequency distribution raised to the power 0.75. That small tweak picks rare words a bit more often than their raw frequency would.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.inp = nn.Embedding(vocab_size, dim)    # centre-word vectors v
        self.out = nn.Embedding(vocab_size, dim)    # context-word vectors u
    def forward(self, centre, context, negatives):  # (B,), (B,), (B, K)
        v = self.inp(centre)                                       # (B, d)
        pos = (self.out(context) * v).sum(-1)                      # (B,)
        neg = torch.bmm(self.out(negatives), v.unsqueeze(2)).squeeze(2)  # (B, K)
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()
```

I use `F.logsigmoid` instead of `torch.log(torch.sigmoid(x))` because it stays accurate when the sigmoid is very close to 0 or 1.

**A mistake I made.** My first skip-gram and CBOW versions computed `sigmoid(u_c · v_w)` and trained it with binary cross-entropy against a target of 1, using only the real pairs. That has a lazy solution: make all vectors big and pointing the same way, so every dot product is huge and every answer is "yes". The loss drops to zero and the vectors mean nothing. It's the negative examples that force the model to tell real neighbours from random words. Without them (or a full softmax), there's nothing to learn.

On a corpus of four sentences nothing meaningful comes out anyway, since word2vec needs millions of words. The toy runs are only useful for checking shapes and the loss, not for reading the vectors.

## Using pretrained vectors

For anything real, it makes more sense to start from vectors someone already trained on a huge amount of text. Gensim can download Google's word2vec model, trained on about 100 billion words of news, with 300-dimensional vectors for 3 million words and phrases.

```python
import gensim.downloader as api
wv = api.load("word2vec-google-news-300")   # about 1.6 GB download

print(wv.most_similar("cat", topn=3))
# analogy: Paris - France + Italy ≈ ?
print(wv.most_similar(positive=["Paris", "Italy"], negative=["France"], topn=1))
```

This one tripped me up. The famous analogy "king - man + woman ≈ queen" is about **nearest neighbours**, not equality. My first test, `wv['Italy'] - wv['Rome'] == wv['France'] - wv['Paris']`, compared two float vectors number by number and gave me 300 `False` values. The right way is to compute the offset vector and ask which word is closest to it by cosine similarity. That's what `most_similar` does (and it also leaves the input words out of the answer).

## Beyond single words

Plain word vectors give one vector per word, so "bank" (river) and "bank" (money) get the same vector. Contextual models like BERT and GPT give a different vector each time a word appears, depending on the sentence. Sentence-embedding models (for example `all-MiniLM-L6-v2` in the sentence-transformers library, which returns 384-dimensional vectors) go one step further and turn whole sentences into vectors, for search and clustering. What I like is that inside every LLM, the first layer is still an embedding table just like the one here.

## Key takeaways

- One-hot vectors are long, mostly zeros and say nothing about similarity. Embeddings are short, dense and learned from context.
- Counting which words appear together, then compressing with SVD, gives embeddings directly. GloVe builds on this counting idea.
- Word2vec learns embeddings by guessing the neighbours (skip-gram) or the centre word (CBOW). Negative sampling makes training cheap.
- Training only on real pairs with a sigmoid doesn't work: the model learns to say "yes" to everything. You need negative examples or a full softmax.
- Analogies are nearest-neighbour searches on vector offsets with cosine similarity, not exact equalities.

## Resources

- Tomas Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", 2013: https://arxiv.org/abs/1301.3781
- Tomas Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (negative sampling), 2013: https://arxiv.org/abs/1310.4546
- Jeffrey Pennington, Richard Socher and Christopher Manning, "GloVe: Global Vectors for Word Representation", 2014: https://nlp.stanford.edu/projects/glove/
- Gensim documentation, Word2Vec and the downloader API: https://radimrehurek.com/gensim/models/word2vec.html
- Sentence-Transformers documentation: https://www.sbert.net
