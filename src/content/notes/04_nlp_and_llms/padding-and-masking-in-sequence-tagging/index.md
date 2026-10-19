---
title: "Padding and masking in sequence tagging: how a 99.97% accuracy can be fake"
slug: padding-and-masking-in-sequence-tagging
date: 2020-05-11
topic: NLP & LLMs
tags: [university-lab, sequence-labeling, pos-tagging, padding, masking, lstm, evaluation]
summary: "How padding sentences to one length inflates per-token accuracy, how masking fixes it, and how to count the parameters of an embedding plus bidirectional LSTM tagger."
sources:
  - title: "732A78 Deep Learning (course lab)"
    author: "Linköping University"
    type: course
  - title: "Universal Dependencies: English Web Treebank (UD_English-EWT)"
    author: "Universal Dependencies project"
    url: "https://universaldependencies.org/treebanks/en_ewt/index.html"
    type: docs
  - title: "Keras guide: Understanding masking & padding"
    author: "Keras team"
    url: "https://keras.io/guides/understanding_masking_and_padding/"
    type: docs
  - title: "Speech and Language Processing (3rd ed. draft), Chapter 8 (Sequence Labeling for Parts of Speech and Named Entities)"
    author: "Daniel Jurafsky, James H. Martin"
    url: "https://web.stanford.edu/~jurafsky/slp3/"
    type: book


status: draft
---

Our part-of-speech tagger (embedding, bidirectional LSTM, softmax at every position) reported a test accuracy of 0.9997. That number should have worried us: good POS taggers on this data get roughly 95 to 97% of tokens right, so 99.97% was a red flag. When I looked back at it, I found two separate mistakes stacked on top of each other. Both are common enough to be worth writing down.

## The setup

A tagger reads a sentence and predicts one tag per word: *There/PRON has/AUX been/VERB talk/NOUN ...*. Sentences have different lengths, but Keras wants one fixed-size tensor. So every sentence gets padded to the length of the longest one with a special index 0 (`<pad>`), and the tag sequence gets padded the same way with a `<pad>` tag. The model outputs, for each sentence $s$ and position $t$, a probability vector over $T$ tags (18 here, counting the pad tag).

## Mistake 1: accuracy counted the padding

Per-token accuracy should be

$$
\text{acc} = \frac{\sum_{s}\sum_{t} m_{s,t}\, \mathbb{1}[\hat y_{s,t} = y_{s,t}]}{\sum_s \sum_t m_{s,t}}
$$

where $y_{s,t}$ is the gold tag at position $t$ of sentence $s$, $\hat y_{s,t}$ is the prediction, $\mathbb{1}[\cdot]$ is 1 when the condition holds and 0 otherwise, and $m_{s,t}$ is 1 for a real token and 0 for padding. Keras' built-in `accuracy` metric leaves $m$ out and counts every position.

Does that really matter much? I did the math. The training data has 12,543 sentences and the longest has 159 tokens, so the padded tensor has about 2.0 million positions. Only 204,585 of them are real tokens. That's about 10%. The other 90% is padding, and predicting `<pad>` after a run of padding is trivial. So a tagger could get every real word wrong and still score about 90%.

```python
import numpy as np

PAD = 0
y_true = np.array([[0, 0, 0, 0, 0, 3, 5, 2],     # front padding, like pad_sequences
                   [0, 0, 0, 4, 4, 1, 3, 2]])
y_pred = np.array([[0, 0, 0, 0, 0, 3, 1, 1],     # 1 of 3 real tags right
                   [0, 0, 0, 4, 2, 2, 2, 2]])    # 2 of 5 real tags right

mask = y_true != PAD
print((y_pred == y_true).mean())         # 0.6875: padding counted as "correct"
print((y_pred == y_true)[mask].mean())   # 0.375: real tokens only
```

The fix in Keras is `Embedding(..., mask_zero=True)`. The mask then flows through the LSTM, so padding doesn't affect the hidden state, and the loss and metrics skip the padded positions. Index 0 must then be reserved for padding only, which is why the vocabulary starts with `<pad>` and then `<unk>`.

## Mistake 2: the "test set" was the training set

The second problem was a plain bug. Our `encode()` function took a list of sentences as its argument but looped over the global `train_data` inside. So the "validation" and "test" tensors were copies of the training data. The training log gives it away: "Train on 12543 samples, validate on 12543 samples", even though the dev set has 2,002 sentences and the test set 2,077. Our 0.9997 was a training-set score with padding included.

Two cheap checks would have caught it: print the shapes of every encoded split, and be suspicious when validation loss is *lower* than training loss from the first epoch on.

## Counting the parameters

The lab also asked for the parameter counts, and my answers were off.

**Embedding layer.** It's a lookup table with one row per vocabulary entry:

$$
\text{params} = |V| \cdot d
$$

with $|V|$ the vocabulary size and $d$ the embedding size. With $|V| = 19{,}674$ (including `<pad>` and `<unk>`) and $d = 100$ that's 1,967,400, which is what `model.summary()` printed. My answer used 19,672 and $d = 50$, which matched neither the model nor the summary.

**LSTM layer.** An LSTM has four gate blocks (input, forget, output, and the candidate cell). Each has a weight matrix on the input $x_t$, a weight matrix on the previous hidden state $h_{t-1}$, and a bias:

$$
\text{params}_\text{LSTM} = 4\,\big(d_\text{in}\, h + h^2 + h\big) = 4\,\big((d_\text{in} + 1)\, h + h^2\big)
$$

where $d_\text{in}$ is the input size and $h$ the number of units. I wrote this formula down correctly, but stopped there. A **bidirectional** LSTM runs two independent LSTMs, one in each direction, so the count doubles:

$$
2 \cdot 4 \cdot \big((100 + 1) \cdot 100 + 100^2\big) = 160{,}800
$$

That also matches the summary. The output size is $2h = 200$ because the two directions are concatenated, so the final dense layer has $200 \cdot 18 + 18 = 3{,}618$ parameters.

The embedding has 92% of the model's 2.1 million parameters. That's typical for small taggers, and it's one reason people start from pretrained word vectors.

## Key takeaways

- Padding can be most of a batch. Unmasked per-token accuracy rewards the model for predicting padding.
- Use `mask_zero=True` (or compute metrics with a mask) and reserve index 0 for padding.
- Check split sizes after preprocessing. A validation set the same size as the training set is a bug.
- Embedding params: $|V| \cdot d$. LSTM params: $4((d_\text{in}+1)h + h^2)$, times 2 if bidirectional.

## Resources

- 732A78 Deep Learning, Linköping University (RNN lab, done in a pair).
- [UD English Web Treebank](https://universaldependencies.org/treebanks/en_ewt/index.html).
- [Keras guide: Understanding masking & padding](https://keras.io/guides/understanding_masking_and_padding/).
- Jurafsky and Martin, [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/), Chapter 8.
- Related notes: [LSTMs](/notes/lstm/), [recurrent neural networks](/notes/recurrent-neural-networks/).
