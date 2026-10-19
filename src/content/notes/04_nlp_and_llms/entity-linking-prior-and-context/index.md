---
title: "Entity linking: from spaCy spans to Wikipedia pages with a prior and a context model"
slug: entity-linking-prior-and-context
date: 2020-12-22
topic: NLP & LLMs
tags: [university-lab, entity-linking, named-entity-recognition, naive-bayes, information-extraction, evaluation]
summary: "How to score mention spans with set-based precision and recall, why filtering spaCy's numeric entity types lifts precision, and how a P(entity | mention) prior plus a per-mention Naive Bayes context model links mentions to Wikipedia pages."
sources:
  - title: "732A92 Text Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "Robust Disambiguation of Named Entities in Text (AIDA / CoNLL entity annotations)"
    author: "Johannes Hoffart et al."
    url: "https://aclanthology.org/D11-1072/"
    type: paper
  - title: "Speech and Language Processing (3rd ed. draft), Chapter on Information Extraction and Entity Linking"
    author: "Daniel Jurafsky, James H. Martin"
    url: "https://web.stanford.edu/~jurafsky/slp3/"
    type: book
  - title: "spaCy: Named Entity Recognition"
    author: "Explosion"
    url: "https://spacy.io/usage/linguistic-features#named-entities"
    type: docs


status: draft
---

Read *Lincoln beat Swansea*. You know right away that "Lincoln" is a football club, not the president or the city in Nebraska. A computer doesn't. Entity linking is the task of answering "which real thing does this name refer to?". In the information extraction lab we built a small pipeline for it on news text annotated with Wikipedia page names: find the names (mention spans) with spaCy, then link each one to a page. What I liked is that each step had a simple idea behind it, and each one gave a clear jump in the scores.

## Scoring with sets

First, how do you even score this? Both steps were scored the same way. Put the gold items and predicted items in two sets (a span is `(sentence_id, start, end)`, a link adds the page name) and compute

$$
P = \frac{|G \cap \hat{S}|}{|\hat{S}|}, \qquad R = \frac{|G \cap \hat{S}|}{|G|}, \qquad F_1 = \frac{2PR}{P + R}
$$

where $G$ is the set of gold items and $\hat S$ the set of predictions. Precision asks "how many of my predictions are right?", recall asks "how many of the true items did I find?". It's an exact match: a span that's off by one token counts as a false positive **and** a false negative.

## Step 1: spans from spaCy, then clean them up

spaCy's NER out of the box got P = 54%, R = 68%, F1 = 60% on the dev spans. Error analysis on training sentences showed two main kinds of false positives:

1. **Entity types the data doesn't annotate.** spaCy tags dates, times, numbers, money and percentages (*Thursday*, *1996-08-22*, *two*). The gold data only marks names. Dropping the types `DATE, TIME, CARDINAL, ORDINAL, QUANTITY, PERCENT, MONEY` raised precision to 83% with recall unchanged at 68%, so F1 went to 75%.
2. **Boundary mismatches.** spaCy often included a leading *the* or a trailing *'s*: it predicted *The European Commission* and *the European Union 's*, while the gold spans were *European Commission* and *European Union*. Each of these counts twice against you, as a false positive and a false negative.

My error summary only mentioned the first kind. The second was right there in the error report, and trimming a leading determiner and a trailing possessive from each span is an easy post-processing step that would have fixed both precision and recall for those cases.

## Step 2: naive linking

The simplest link: join the span's tokens with underscores and call that the page name (*Jimi Hendrix* → `Jimi_Hendrix`). It got F1 = 28% (P 31%, R 26%). It fails for adjectives (*German* → `Germany`), abbreviations, and every ambiguous name.

## Step 3: the most probable entity

A knowledge base harvested from Wikipedia anchor texts gives, for each mention string $m$, a distribution over pages $e$:

$$
P(e \mid m) = \frac{\text{count}(m \text{ links to } e)}{\text{count}(m \text{ used as a link})}
$$

For "Sweden" that's 0.986 for the country and 0.014 for the national football team. Linking every mention to its most probable page (and to `--NME--`, "not in the knowledge base", when the mention is unknown) raised F1 to 58%. Most mentions have one dominant meaning, so this prior alone does most of the work.

## Step 4: context-sensitive disambiguation

The prior always picks the same page for "Lincoln", whatever the sentence says. For the ambiguous mentions, the obvious next step is to look at the words around them. For each mention with several candidate pages, train a small Multinomial Naive Bayes classifier on contexts (5 tokens left and right) where the mention linked to each page. Use $P(e \mid m)$ as the class prior, so the classifier picks

$$
\hat e = \arg\max_e \Big[\log P(e \mid m) + \sum_{w \in \text{context}} \log P(w \mid e)\Big]
$$

With weak context, the prior wins. With strong context ("president", "1865"), the context overrides it. This added about one point: P 65%, R 53%, F1 59%. The gain is small because most mentions aren't ambiguous, and spans that are already wrong can't be fixed by linking.

One detail I got wrong: in the training contexts, the mention itself is replaced by `@`. At prediction time I built the context window from the sentence tokens *including* the mention words. So the test input had features (the mention's own words) that never appeared in training, and the `@` that always appeared in training was missing. It's the usual rule: build features at prediction time exactly as at training time.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def train_linker(contexts, entities, prior):
    """One classifier per ambiguous mention. prior: {entity: P(entity | mention)}."""
    classes = sorted(set(entities))                 # sklearn orders classes alphabetically
    nb = MultinomialNB(class_prior=[prior[e] for e in classes])
    vec = CountVectorizer(token_pattern=r"\S+")    # keep '@' (the default pattern drops it)
    return make_pipeline(vec, nb).fit(contexts, entities)

def link(tokens, beg, end, clf, window=5):
    # Same format as the training contexts: mention replaced by '@'
    ctx = tokens[max(0, beg - window):beg] + ["@"] + tokens[end:end + window]
    return clf.predict([" ".join(ctx)])[0]

# With priors Nebraska 0.7 / Abraham_Lincoln 0.2 / Lincoln_City_F.C. 0.1 and a few contexts each,
# "Swansea beat Lincoln on goals ..." links to the club, "speech by president Lincoln" to the
# president, and "they drove to Lincoln last week" (no useful context) to the prior's choice.
```

Two details here are easy to get silently wrong. The order of `class_prior` has to match the classifier's sorted class names. And scikit-learn's default token pattern only keeps words of two or more letters or digits, so it would quietly throw the `@` away. That's why the vectorizer splits on whitespace instead.

## Key takeaways

- Set-based P/R/F1 with exact span matching: a boundary error costs both a false positive and a false negative.
- Off-the-shelf NER needs post-processing to match a dataset's annotation guidelines: drop entity types it doesn't annotate and fix span boundaries.
- A $P(e \mid m)$ prior from anchor texts is a strong linker on its own. Context models help with the ambiguous minority.
- Build test-time features exactly like training features (here: the `@` placeholder for the mention).

## Resources

- 732A92 Text Mining, Linköping University (information extraction lab).
- Hoffart et al., [Robust Disambiguation of Named Entities in Text](https://aclanthology.org/D11-1072/), EMNLP 2011 (source of the CoNLL entity annotations).
- Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), information extraction chapters.
- [spaCy: Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities).
- Related note: [Multinomial Naive Bayes for text](/notes/multinomial-naive-bayes-for-text/).
