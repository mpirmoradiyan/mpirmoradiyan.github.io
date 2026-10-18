---
title: "Word embeddings: from co-occurrence counts to word2vec"
slug: word-embeddings
date: 2025-03-07
topic: NLP & LLMs
tags: [word-embeddings, word2vec, skip-gram, cbow, negative-sampling, svd, gensim]
summary: "Why dense word vectors beat one-hot codes, how to get them from co-occurrence counts with SVD or by training skip-gram with negative sampling, and how to query pretrained vectors for similarity and analogies."
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

Neural networks need numbers, and the simplest way to turn a word into numbers is a one-hot vector: as long as the vocabulary, all zeros except a single 1. One-hot vectors are huge and say nothing about meaning; "cat" is exactly as far from "dog" as from "carburettor". **Word embeddings** replace them with short dense vectors, learned so that words used in similar contexts end up close together. This note builds embeddings two ways, from counts and with word2vec, and then uses pretrained vectors.

## The idea: you know a word by its neighbours

The distributional hypothesis says that words appearing in similar contexts tend to have similar meanings. "Cat" and "dog" both show up next to "the", "chased", "played" and "pet". If a vector for each word is built from, or trained to predict, its neighbours, words with similar neighbours get similar vectors.

Similarity between vectors is usually measured with **cosine similarity**:

$$
\cos(u, v) = \frac{u \cdot v}{\lVert u \rVert \, \lVert v \rVert},
$$

which is 1 for vectors pointing the same way, 0 for orthogonal ones and -1 for opposite ones. It ignores vector length, which often reflects word frequency more than meaning.

## Method 1: count co-occurrences, then compress with SVD

Count how often each word appears within a window of each other word. With a window of 1 on three toy sentences ("the cat sat on the mat", "the dog chased the cat", "the cat and the dog played") this gives a 9 x 9 symmetric matrix. Each row is already a vector description of a word, but it is as long as the vocabulary and very sparse.

The **singular value decomposition** $M = U \Sigma V^\top$ finds the directions that explain most of the structure in the matrix. Keeping the top $k$ singular values gives the best rank-$k$ approximation of $M$, and the rows of $U_k \Sigma_k^{1/2}$ serve as $k$-dimensional word vectors.

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

This is the core of latent semantic analysis. In practice raw counts are dominated by frequent words like "the", so the matrix is usually reweighted first, for example with the logarithm or positive pointwise mutual information. GloVe is a modern descendant of this counting approach: it fits word vectors so that their dot products match the log co-occurrence counts.

## Method 2: word2vec, learning by prediction

Word2vec trains a shallow model on a prediction task and keeps the learned weights as embeddings. It has two variants:

- **Skip-gram**: given the centre word, predict each word in its context window.
- **CBOW** (continuous bag of words): given the averaged context words, predict the centre word.

Each word gets two vectors: $v_w$ when it is the centre (input) word and $u_c$ when it is a context (output) word. The full skip-gram model uses a softmax over the whole vocabulary,

$$
P(c \mid w) = \frac{\exp(u_c \cdot v_w)}{\sum_{c'=1}^{|V|} \exp(u_{c'} \cdot v_w)},
$$

where $|V|$ is the vocabulary size. The denominator touches every word in the vocabulary for every training pair, which is far too slow for a vocabulary of a million words.

### Negative sampling

**Negative sampling** turns the problem into binary classification. For each real (centre, context) pair, draw $K$ random "negative" words $n_1, \dots, n_K$ and train the model to say yes to the real pair and no to the fake ones:

$$
\mathcal{L} = -\log \sigma(u_c \cdot v_w) - \sum_{k=1}^{K} \log \sigma(-u_{n_k} \cdot v_w),
$$

where $\sigma$ is the sigmoid. The negatives are drawn from the unigram distribution raised to the power 0.75, which samples rare words a bit more often than their raw frequency.

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

`F.logsigmoid` is used instead of `torch.log(torch.sigmoid(x))` because it stays accurate when the sigmoid is very close to 0 or 1.

**A mistake to avoid.** My first skip-gram and CBOW versions computed `sigmoid(u_c · v_w)` and trained it with binary cross-entropy against a target of 1, using only the real pairs. That objective has a trivial solution: make all vectors large and aligned, so every dot product is huge and every prediction says "yes". The loss goes to zero and the vectors mean nothing. The negative examples are what force the model to tell real neighbours apart from random words. Without them (or a full softmax), there is nothing to learn.

On a corpus of four sentences nothing meaningful emerges anyway; word2vec needs millions of words. The value of the toy runs is checking shapes and the loss, not reading the vectors.

## Using pretrained vectors

For real work, start from vectors trained on a large corpus. Gensim can download Google's word2vec model, trained on about 100 billion words of news, with 300-dimensional vectors for 3 million words and phrases.

```python
import gensim.downloader as api
wv = api.load("word2vec-google-news-300")   # about 1.6 GB download

print(wv.most_similar("cat", topn=3))
# analogy: Paris - France + Italy ≈ ?
print(wv.most_similar(positive=["Paris", "Italy"], negative=["France"], topn=1))
```

The famous analogy property, "king - man + woman ≈ queen", is a statement about **nearest neighbours**, not equality. My first test, `wv['Italy'] - wv['Rome'] == wv['France'] - wv['Paris']`, compared two float vectors element by element and returned 300 `False` values. The right test computes the offset vector and asks which word is closest to it by cosine similarity, which is what `most_similar` does (it also excludes the input words from the answer).

## Beyond single words

Static word vectors give one vector per word, so "bank" (river) and "bank" (money) share a vector. Contextual models such as BERT and GPT produce a different vector for each occurrence, depending on the sentence. Sentence-embedding models (for example `all-MiniLM-L6-v2` in the sentence-transformers library, which returns 384-dimensional vectors) go one step further and embed whole sentences for search and clustering. Inside every LLM, though, the first layer is still an embedding table of exactly the kind described here.

## Key takeaways

- One-hot vectors are long, sparse and carry no similarity; embeddings are short, dense and learned from context.
- Co-occurrence counts plus SVD give embeddings directly; GloVe refines this counting view.
- Word2vec learns embeddings by predicting neighbours (skip-gram) or the centre word (CBOW); negative sampling makes training cheap.
- Training only on positive pairs with a sigmoid is degenerate. Negative examples or a full softmax are required.
- Analogies are nearest-neighbour searches on vector offsets with cosine similarity, not exact equalities.

## Resources

- Tomas Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", 2013: https://arxiv.org/abs/1301.3781
- Tomas Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (negative sampling), 2013: https://arxiv.org/abs/1310.4546
- Jeffrey Pennington, Richard Socher and Christopher Manning, "GloVe: Global Vectors for Word Representation", 2014: https://nlp.stanford.edu/projects/glove/
- Gensim documentation, Word2Vec and the downloader API: https://radimrehurek.com/gensim/models/word2vec.html
- Sentence-Transformers documentation: https://www.sbert.net
