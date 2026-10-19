---
title: "Tf-idf: a tiny search engine and keyword extractor"
slug: tf-idf-search-and-keywords
date: 2020-11-08
topic: NLP & LLMs
tags: [university-lab, tf-idf, information-retrieval, vector-space-model, cosine-similarity, keyword-extraction]
summary: "How tf-idf weights words, why Euclidean nearest neighbours on tf-idf vectors give the same ranking as cosine similarity, and why keyword extraction needs idf from the whole collection, not from one document."
sources:
  - title: "732A92 Text Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "Introduction to Information Retrieval, Chapter 6 (Scoring, term weighting and the vector space model)"
    author: "Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze"
    url: "https://nlp.stanford.edu/IR-book/"
    type: book
  - title: "scikit-learn User Guide: Tf–idf term weighting"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting"
    type: docs


status: draft
---

The first text mining lab was to build a small search engine over about 1,600 app descriptions from the Google Play Store. You type "dodge trains" and get Subway Surfers. It sounds like a big thing, but it's built from just three ideas: tf-idf weights, vectors and cosine similarity. Two of my answers got small details wrong, and those details turned out to be the useful part.

## Tf-idf in words

The starting question: how do you turn a piece of text into numbers you can compare? Turn each document into a vector with one entry per word in the vocabulary. The entry should be high when a word is **frequent in this document** but **rare in the collection**. "Game" shows up in most app descriptions, so it says little about any one of them. "Railyard" is rare, so when it shows up it says a lot.

$$
\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t)
$$

- $\text{tf}(t, d)$ is how often term $t$ occurs in document $d$.
- $\text{idf}(t)$ is large for rare terms.

The textbook idf is $\log \frac{N}{\text{df}(t)}$, with $N$ the number of documents and $\text{df}(t)$ the number of documents containing $t$. That's what I wrote in my lab answer. But the code used scikit-learn's `TfidfVectorizer`, which by default uses a smoothed version:

$$
\text{idf}(t) = \ln \frac{1 + N}{1 + \text{df}(t)} + 1
$$

The $+1$ inside avoids dividing by zero, and the $+1$ outside means a word in every document gets idf 1, not 0, so it isn't dropped completely. So my statement "a term in all documents gets idf $\log 1 = 0$" was true for the textbook formula, not for the numbers my code produced. After this, each document vector is scaled to unit length (L2 norm 1).

The lowest-idf terms in the app data were *game, play, feature, free, new, world, time, app, fun, well*, which is what you'd expect from app-store marketing text. The highest-idf terms are words that appear in just one description, and they all tie for the top value.

## Searching: nearest neighbours by cosine

To search, vectorise the query with the same fitted vectorizer and return the documents whose vectors point in the most similar direction:

$$
\cos(q, d) = \frac{q \cdot d}{\lVert q \rVert\, \lVert d \rVert}
$$

My `search` used scikit-learn's `NearestNeighbors` with its default metric, Euclidean distance, not cosine. It still returned the right ranking, and the reason is worth knowing. When both vectors have length 1,

$$
\lVert q - d \rVert^2 = \lVert q \rVert^2 + \lVert d \rVert^2 - 2\, q \cdot d = 2 - 2\cos(q, d)
$$

so a smaller Euclidean distance means a larger cosine. The ranking is identical, but only because `TfidfVectorizer` normalises by default. With raw counts, Euclidean distance would favour short documents, and the two rankings would differ. Writing `metric="cosine"` makes the intent clear and doesn't depend on that.

## Keyword extraction: idf must come from the collection

A simple keyword extractor picks the $n$ terms with the highest tf-idf in one document. My version built a **new** `TfidfVectorizer` and fit it on that single document. With $N = 1$, every word has the same df, so every word gets the same idf, and the "tf-idf" ranking is just a word-count ranking. That's why my output contained generic words like *let*, *pick* and *manage* instead of the expected *overcast, locomotive, timetable, railyard*. The idf has to come from the whole collection, so you transform the text with the vectorizer that was already fitted on all descriptions. (My version also returned the top terms in increasing order. The highest score should come first.)

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

docs = ["dodge the trains and run on the tracks",
        "a train simulator with real railway timetables",
        "run from the trains in this endless runner game",
        "track your running and cycling"]
vec = TfidfVectorizer()                  # smoothed idf, rows L2-normalised
X = vec.fit_transform(docs)
q = vec.transform(["dodge trains"])

cos = cosine_similarity(q, X)[0]
print(np.round(cos, 3))                                  # [0.47  0.  0.174 0. ]
print(np.round(euclidean_distances(q, X)[0] ** 2, 3))    # [1.061 2. 1.652 2.] = 2 - 2*cos

def keywords(text, n=5):
    """Top-n terms of one text, using idf from the whole collection."""
    row = vec.transform([text]).toarray()[0]             # reuse the fitted vectorizer
    terms = vec.get_feature_names_out()
    return [terms[i] for i in row.argsort()[::-1][:n] if row[i] > 0]

print(keywords(docs[2]))
```

## A practical note on preprocessing

The lab plugged a spaCy function into the vectorizer as the tokenizer (lemmatise, drop stop words and non-alphabetic tokens). My function called `spacy.load(...)` *inside* the tokenizer, so the language model was reloaded for every one of the 1,614 documents and for every query. Load it once, outside the function. The results are the same, and it's much faster.

Word-count vectors like these are also what a simple text classifier eats. The next lab used them to guess a speaker's party, in [Multinomial Naive Bayes for text](/notes/multinomial-naive-bayes-for-text/).

## Key takeaways

- Tf-idf = term frequency × inverse document frequency. scikit-learn uses a smoothed idf, $\ln\frac{1+N}{1+\text{df}} + 1$, not plain $\log\frac{N}{\text{df}}$.
- On unit-length vectors, squared Euclidean distance equals $2 - 2\cos$, so the two give the same ranking. On unnormalised vectors they don't.
- Keyword extraction needs idf fitted on the collection. Fitting on one document gives every word the same idf.
- Load heavy models (spaCy pipelines) once, not inside a function called per document.

## Resources

- 732A92 Text Mining, Linköping University (information retrieval lab).
- Manning, Raghavan and Schütze, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), Chapter 6.
- [scikit-learn: Tf–idf term weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting).
