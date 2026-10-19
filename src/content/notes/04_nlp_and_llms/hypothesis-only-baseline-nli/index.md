---
title: "Natural language inference: the hypothesis-only baseline and cross-unigram features"
slug: hypothesis-only-baseline-nli
date: 2020-12-16
topic: NLP & LLMs
tags: [university-lab, natural-language-inference, snli, baselines, annotation-artifacts, feature-engineering]
summary: "Why a classifier that only reads the hypothesis gets about 60% on SNLI while one that only reads the premise gets chance, what that says about annotation artifacts, and how to build cross-unigram features so that each word pair really is one feature."
sources:
  - title: "732A92 Text Mining (course lab)"
    author: "Linköping University"
    type: course
  - title: "A large annotated corpus for learning natural language inference"
    author: "Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning"
    url: "https://aclanthology.org/D15-1075/"
    type: paper
  - title: "Annotation Artifacts in Natural Language Inference Data"
    author: "Suchin Gururangan et al."
    url: "https://aclanthology.org/N18-2017/"
    type: paper
  - title: "Hypothesis Only Baselines in Natural Language Inference"
    author: "Adam Poliak et al."
    url: "https://aclanthology.org/S18-2023/"
    type: paper


status: draft
---

Natural language inference (NLI) gives you two sentences, a premise and a hypothesis, and asks whether the premise **entails** the hypothesis, **contradicts** it, or is **neutral**. The task is about the relation between the two sentences, so a model that reads only one of them shouldn't do better than guessing. In the word embeddings lab, one of them did much better than guessing, and that was the most interesting result of the lab.

## The data

SNLI (Bowman et al., 2015) has about 570k sentence pairs. The premises are image captions. Crowd workers saw a caption and wrote three hypotheses: one definitely true, one maybe true, one definitely false. For example:

- Premise: *A soccer game with multiple males playing.*
- Hypothesis: *Some men are playing a sport.* → entailment

## Baselines

We trained a bag-of-words logistic regression on just one sentence of each pair and evaluated on the dev set (accuracy):

| Model | 25k training pairs | All 549k pairs |
|---|---|---|
| Random (stratified) | 0.339 | 0.340 |
| Premise only | 0.331 | 0.337 |
| Hypothesis only | 0.602 | 0.650 |
| Hypothesis + cross-unigrams | 0.611 | 0.677 |
| Summed word vectors + MLP | 0.649 | 0.761 |

**Premise only is at chance, and that's by design.** Every premise was written up with one hypothesis of each label, so the same premise appears once as entailment, once as neutral and once as contradiction. The premise alone carries no information about the label.

**Hypothesis only gets 60 to 65%.** In our lab answer we explained it as "the gold label depends on the state of sentence 2 following sentence 1". That's too vague, and it misses what's going on. The label is supposed to depend on *both* sentences. The hypothesis-only model does well because of **annotation artifacts**: habits in how people wrote each kind of hypothesis. Gururangan et al. and Poliak et al. found, for example:

- Contradictions often contain negation words (*nobody, no, never, nothing*) or unrelated activities (*sleeping*, *a cat*).
- Entailments tend to be shorter and more generic (*a person*, *outdoors*, *some men*), because the easy way to write something definitely true is to say less.
- Neutral hypotheses often add details or purposes the premise doesn't support (*for a competition*, *his first*, *tall*).

So a word like "nobody" predicts contradiction without reading the premise at all. That's a shortcut, not inference. It matters because a dataset with strong artifacts can make a model look like it understands sentence relations when it mostly learned word statistics of one sentence. The hypothesis-only score is the bar a real NLI model has to clear.

## Cross-unigram features

Bowman et al. used **cross-unigrams**: pairs $(w_1, w_2)$ where $w_1$ is in the premise, $w_2$ is in the hypothesis, and both have the same part-of-speech tag. A pair like `males__men` (two nouns) or `playing__sleeping` (two verbs) lets a linear model learn things a single sentence can't show, such as "males ↔ men supports entailment" and "playing ↔ sleeping suggests contradiction".

The key is that each pair has to become **one** feature. Our transformer did this:

```python
result += ' '.join((word2, word1))
```

That appends "men males" with a space in the middle (and no space before, so it glues onto the previous word). The vectorizer then splits it back into the two single words. So our "cross-unigram" model mostly just counted premise and hypothesis words again, and never saw a pair. That explains why it barely beat the hypothesis-only baseline. A correct version joins each pair into one token:

```python
def cross_unigrams(s1, t1, s2, t2):
    """Hypothesis unigrams plus premise/hypothesis word pairs with the same POS tag."""
    feats = s2.split()
    for w2, tag2 in zip(s2.split(), t2.split()):
        for w1, tag1 in zip(s1.split(), t1.split()):
            if tag1 == tag2:
                feats.append(f"{w1}__{w2}")      # one token per pair, e.g. "males__men"
    return " ".join(feats)

print(cross_unigrams("multiple males playing", "ADJ NOUN VERB",
                     "some men sleep", "DET NOUN VERB"))
# some men sleep males__men playing__sleep
```

scikit-learn's default token pattern treats `_` as a word character, so `males__men` stays one feature. The number of possible pairs is large, so this needs a lot of data. Bowman et al.'s full lexicalized classifier, which also used cross-bigrams and other features, got about 78% on SNLI.

## Dense vectors: summing word embeddings

The last model represented each sentence as the **sum of its word vectors** (spaCy's 300-dimensional vectors), concatenated the two sentence vectors into 600 numbers, and fed them to an MLP with three hidden layers of 300 units. It went from 0.649 with 25k training pairs to 0.761 with all of them. That's the biggest gain from more data of any model in the table, which fits the intuition that a flexible model needs more examples before it stops overfitting. Word vectors also help with words the model saw rarely, because similar words have similar vectors.

## Key takeaways

- Always run one-sided baselines for a two-input task. Premise-only at chance and hypothesis-only well above it means annotation artifacts are present.
- SNLI hypotheses contain label cues (negation for contradiction, generic wording for entailment, added details for neutral). A model can exploit them without reading the premise.
- Pair features must be joined into single tokens, or the vectorizer turns them back into unigrams.
- Summed word vectors + an MLP beat sparse features here, and gained the most from more data.

## Resources

- 732A92 Text Mining, Linköping University (word embeddings lab, done in a pair).
- Bowman et al., [A large annotated corpus for learning natural language inference](https://aclanthology.org/D15-1075/), EMNLP 2015.
- Gururangan et al., [Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/), NAACL 2018.
- Poliak et al., [Hypothesis Only Baselines in Natural Language Inference](https://aclanthology.org/S18-2023/), *SEM 2018.
- Related note: [word embeddings](/notes/word-embeddings/).
