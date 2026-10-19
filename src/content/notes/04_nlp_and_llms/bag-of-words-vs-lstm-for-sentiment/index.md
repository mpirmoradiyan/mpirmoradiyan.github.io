---
title: "Bag-of-words vs. LSTM for tweet sentiment: what each model can actually see"
slug: bag-of-words-vs-lstm-for-sentiment
date: 2020-12-17
topic: NLP & LLMs
tags: [university-lab, sentiment-analysis, text-classification, bag-of-words, lstm, stop-words, model-selection]
summary: "What information a bag-of-words model, a tree ensemble on n-grams and an LSTM each have access to, why default stop-word lists delete negations, and the evaluation mistakes that make a model comparison look cleaner than it is."
sources:
  - title: "732A92 Text Mining (course project)"
    author: "Linköping University"
    type: course
  - title: "Coronavirus tweets NLP - Text Classification (Kaggle dataset)"
    author: "Aman Miglani"
    url: "https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification"
    type: docs
  - title: "Speech and Language Processing (3rd ed. draft), Chapter 4 (Naive Bayes, Text Classification, and Sentiment)"
    author: "Daniel Jurafsky, James H. Martin"
    url: "https://web.stanford.edu/~jurafsky/slp3/"
    type: book
  - title: "Deep Learning with Python, Chapter 6 (Deep learning for text and sequences)"
    author: "François Chollet"
    type: book
  - title: "Understanding LSTM Networks"
    author: "Christopher Olah"
    url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
    type: docs


status: draft
---

For the text mining project I classified about 45,000 coronavirus tweets as negative, neutral or positive with three kinds of models: Multinomial Naive Bayes, tree ensembles (Random Forest and AdaBoost) on n-gram counts, and a bidirectional LSTM. The ranking came out as you'd guess: Naive Bayes at about 0.68 test accuracy, the ensembles at about 0.79 to 0.81, and the LSTMs around 0.87. My report explained the whole ranking with one story: "word order and dependencies matter". Looking back, that story fits the LSTM but not the ensembles, and a few evaluation details make the comparison look cleaner than it really was.

## What each model sees

**Bag-of-words Naive Bayes** sees word counts and nothing else. "Not good, bad" and "good, not bad" are the same input. On top of that, it assumes every word is independent of the others given the class.

**Tree ensembles on the same counts** see exactly the same input. A Random Forest or AdaBoost on unigram counts gets no word-order information at all. What trees *can* do that Naive Bayes can't is model **interactions**: a split on "hand" followed by a split on "sanitizer" can treat that pair differently from either word alone, and a tree can ignore words that don't help. So when my report said ensembles did better because they don't assume independence and therefore capture "the dependency between the words", the first half is right and the second half isn't. They capture co-occurrence, not order.

**N-grams** add *local* order back into a bag-of-words model. With bigrams, "not good" becomes its own feature. For AdaBoost the best features were unigrams to trigrams combined. For Random Forest, unigrams worked best.

**An LSTM** reads the tokens in sequence (in both directions for a bidirectional LSTM), so it can in principle use order and long-range context. It also learns its own word embeddings, so similar words get similar vectors and rare words can borrow strength from common ones. My model was embedding (100) → bidirectional LSTM (100 units) → global max pooling over time → softmax over 3 classes. Its gain over the ensembles probably comes from both order and learned embeddings. The experiment can't separate the two.

## The stop-word trap

The report describes stop-word removal as part of preprocessing, and in the code the Naive Bayes and ensemble pipelines did it inside the vectorizer (`stop_words="english"`). For topic classification that's usually harmless. For sentiment it's a real problem, because the standard lists contain negations. scikit-learn's `stop_words="english"` includes *not, no, nor, never, nothing, none, cannot* and *against*. NLTK's and spaCy's lists also include *not*, *no* and *nor*.

```python
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

tweets = ["the store is good", "the store is not good", "no masks left, never again"]

vec = CountVectorizer(stop_words="english", ngram_range=(1, 2))
print(vec.fit(tweets).get_feature_names_out())
# ['good' 'left' 'masks' 'masks left' 'store' 'store good']  -> "not good" is gone

keep = {"not", "no", "nor", "never", "nothing", "none", "cannot", "against"}
vec = CountVectorizer(stop_words=sorted(ENGLISH_STOP_WORDS - keep), ngram_range=(1, 2))
print(vec.fit(tweets).get_feature_names_out())
# now 'not', 'not good', 'no' and 'no masks' survive
```

After stop-word removal, "the store is good" and "the store is not good" give the same features, so even bigrams can't help. For sentiment, keep negations, or skip stop-word removal and let the model (or idf weighting) decide.

There's a second problem here. The LSTM was fed the cleaned tweets *without* stop-word removal, so it still saw every *not* and *no*, while the bag-of-words models didn't. Part of the LSTM's lead may come from that difference in input, not from the architecture. A fair comparison gives every model the same preprocessed text.

## Evaluation details that matter

**Choosing on the test set.** I trained about ten LSTM variants and compared them all on the same test set, then reported the best. Choosing among many models on the test set makes the winning score optimistic. The test set has effectively become a validation set. The clean way: tune on a validation split (Keras' `validation_split` was already there), choose once, and touch the test set only for the final model.

**Fit preprocessing on training data only.** The Keras `Tokenizer` was fit on all tweets before the train/test split, so the vocabulary (and the size of the embedding table) included words that only appear in test tweets. It's a small leak, but the rule is simple: fit anything that learns from data after the split, on the training part only.

**Softmax output plus `from_logits=True`.** The output layer had a softmax, and the loss was `SparseCategoricalCrossentropy(from_logits=True)`. `from_logits=True` tells the loss that the inputs are raw scores, so it applies softmax again, to values that are already probabilities. Training still works more or less, but the gradients are flattened. Use a softmax output with `from_logits=False`, or a linear output with `from_logits=True`.

**Per-class metrics.** My report said Naive Bayes had its lowest *precision* on the neutral class. Its real weakness was neutral *recall*: 0.23, while neutral precision (0.68) was similar to the other classes. The model rarely predicted neutral, which is typical when a class is small (neutral was about 19% of the tweets) and shares its vocabulary with the others. AdaBoost reached 0.86 recall on neutral.

**A Python trap.** The dense-layer size was written `n_nodes/(2^i)`. In Python `^` is bitwise XOR, not a power, so `2^0` is 2 and `2^1` is 3. Use `2**i`.

## Key takeaways

- Bag-of-words models (Naive Bayes, and trees on counts) see no word order. Trees add feature interactions, n-grams add local order, and sequence models add order plus learned embeddings.
- Default stop-word lists remove negations, which wrecks sentiment features. Keep *not*, *no*, *never* and friends, and give every model you compare the same input text.
- Picking the best of many models on the test set inflates the reported score. Use a validation split to choose.
- Softmax output → `from_logits=False`. And in Python, `**` is power and `^` is XOR.

## Resources

- 732A92 Text Mining, Linköping University (individual course project).
- [Coronavirus tweets NLP dataset](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification), Kaggle.
- Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), Chapter 4 (sentiment and negation).
- Chollet, *Deep Learning with Python*, Chapter 6.
- Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
- Related notes: [LSTMs](/notes/lstm/), [Multinomial Naive Bayes for text](/notes/multinomial-naive-bayes-for-text/).
