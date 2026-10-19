---
title: "Multinomial Naive Bayes for text, and tuning it without fooling yourself"
slug: multinomial-naive-bayes-for-text
date: 2020-11-13
topic: NLP & LLMs
tags: [university-lab, naive-bayes, text-classification, bag-of-words, grid-search, smoothing]
summary: "How Multinomial Naive Bayes scores a document from word counts, what additive smoothing and the bag-of-words vs. set-of-words choice do, and why a cross-validation score on resampled training data can't be compared with a test score."
sources:
  - title: "732A92 Text Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "Speech and Language Processing (3rd ed. draft), Chapter 4 (Naive Bayes, Text Classification, and Sentiment)"
    author: "Daniel Jurafsky, James H. Martin"
    url: "https://web.stanford.edu/~jurafsky/slp3/"
    type: book
  - title: "Introduction to Information Retrieval, Chapter 13 (Text classification and Naive Bayes)"
    author: "Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze"
    url: "https://nlp.stanford.edu/IR-book/"
    type: book
  - title: "scikit-learn User Guide: Naive Bayes"
    author: "scikit-learn developers"
    url: "https://scikit-learn.org/stable/modules/naive_bayes.html"
    type: docs


status: draft
---

Naive Bayes is the classic first model for text: fast, no tuning to speak of, and often a hard baseline to beat. In the text classification lab we used it to guess a speaker's party from speeches in the Swedish Riksdag, with the 2017/18 session for training and 2018/19 for testing. The model was easy. Tuning it properly and reading the scores was where I slipped.

## The model

Treat a document as a bag of words: keep the counts, forget the order. For a class $c$ (a party) and a document with word counts $x_1, \dots, x_V$ over a vocabulary of $V$ words, Multinomial Naive Bayes picks

$$
\hat c = \arg\max_c \Big[\log P(c) + \sum_{w=1}^{V} x_w \log P(w \mid c)\Big]
$$

- $P(c)$ is the prior: the share of training documents in class $c$.
- $P(w \mid c)$ is the probability that a random word from a class-$c$ document is $w$.
- $x_w$ is how many times word $w$ appears in the document.

The "naive" part is assuming the words are independent given the class. That's clearly false, but the ranking of classes often comes out fine anyway. Logs turn a long product of small probabilities into a sum, which avoids numerical underflow.

## Smoothing

$P(w \mid c)$ is estimated from counts:

$$
P(w \mid c) = \frac{N_{cw} + \alpha}{N_c + \alpha V}
$$

$N_{cw}$ is how often $w$ appears in class $c$'s training documents, $N_c$ is the total word count for class $c$, and $\alpha$ is the smoothing constant. Without $\alpha$, a word never seen with a class gets probability 0, its log is $-\infty$, and one unseen word vetoes the class. $\alpha = 1$ is Laplace smoothing. Smaller values like $0.1$ (Lidstone) trust the counts more. In our grid search, $\alpha = 0.1$ won.

## Bag-of-words vs. set-of-words

- **Bag-of-words**: $x_w$ is the count. A speech that says "skatt" (tax) ten times gets ten times the evidence.
- **Set-of-words**: $x_w$ is 1 if the word occurs at all, else 0. Repetition doesn't count.

In scikit-learn the switch is `CountVectorizer(binary=True)`. The lab asked us to grid-search exactly this, plus unigrams vs. unigrams+bigrams and $\alpha \in \{1, 0.1\}$. My grid had a tf-idf on/off switch (`TfidfTransformer(use_idf=...)`) where the binary switch should have been, so I never tested set-of-words. The grid I should have run:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([("vec", CountVectorizer()), ("nb", MultinomialNB())])
grid = {
    "vec__binary": [False, True],          # bag-of-words (counts) vs set-of-words (0/1)
    "vec__ngram_range": [(1, 1), (1, 2)],  # unigrams vs unigrams + bigrams
    "nb__alpha": [1.0, 0.1],               # additive smoothing
}
search = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro")
# search.fit(train_texts, train_labels); search.best_params_
```

Multinomial NB is defined for counts, but it also runs on tf-idf values in practice, so my version wasn't broken, just a different question.

## Reading the scores: don't compare a CV score with a test score

My best grid-search configuration had a cross-validation accuracy of 0.62. On the test set the same pipeline got 0.45. That gap isn't mainly overfitting. The two numbers were measured on different things:

1. **Different class balance.** I ran the grid search on the *undersampled* training set, where all 8 parties have the same number of speeches. A random guess there is right 1/8 of the time. The test set keeps the real imbalance (the biggest party has 30% of the speeches). Accuracy on a balanced set and on an imbalanced set aren't on the same scale.
2. **Different time period.** Test speeches come from the next parliamentary session, after the 2018 election. Topics and wording shift, so even a perfect CV estimate on 2017/18 overestimates performance on 2018/19.

So the CV score is useful for **choosing** between configurations, all measured the same way. To report performance, use the test set, and make sure the CV folds look like the test data as much as possible. Choosing a scoring metric that doesn't depend on class balance, like macro-F1, also makes the two numbers easier to compare.

The same lab compared Naive Bayes trained on all speeches with Naive Bayes trained on the undersampled set. That trade-off (accuracy down, macro-F1 up) is in [class imbalance: baselines, weights and undersampling](/notes/class-imbalance-baselines-and-weights/).

## What the confusion matrix showed

Using the confusion matrix of the undersampled model, I listed for each party the party it was most often confused with. Most parties were most often mistaken for M (Moderaterna), and M itself for C (Centerpartiet). Some of that may be real similarity in what these parties talk about. But a column can also collect confusions just because the model predicts that class a lot. Looking at both views helps: the raw row (where do this party's speeches end up?) and the column divided by its total (when the model says M, how often is it some other party?).

## Key takeaways

- Multinomial NB adds up $\log P(w \mid c)$ for each word occurrence, plus the class log prior.
- Smoothing $\alpha$ keeps unseen words from zeroing out a class. Smaller $\alpha$ trusts the counts more.
- Set-of-words is `CountVectorizer(binary=True)`. It ignores repetition.
- A CV score on resampled or older data isn't comparable with a test score on real, newer data. Use CV to choose, the test set to report.

## Resources

- 732A92 Text Mining, Linköping University (text classification lab).
- Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), Chapter 4.
- Manning, Raghavan and Schütze, [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), Chapter 13.
- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html).
