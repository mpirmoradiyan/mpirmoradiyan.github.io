---
title: "Sequence-to-sequence models: an encoder, a decoder and teacher forcing"
slug: seq2seq-encoder-decoder
date: 2025-04-15
topic: NLP & LLMs
tags: [seq2seq, encoder-decoder, gru, teacher-forcing, machine-translation, attention, pytorch]
summary: "How an RNN encoder compresses a source sentence into a vector and an RNN decoder generates the target sentence from it, what teacher forcing and start/end tokens are for, and why the single-vector bottleneck led to attention."
sources:
  - title: "Sequence to Sequence Learning with Neural Networks"
    author: "Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)"
    url: "https://arxiv.org/abs/1409.3215"
    type: paper
  - title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
    author: "Kyunghyun Cho et al. (2014)"
    url: "https://arxiv.org/abs/1406.1078"
    type: paper
  - title: "Neural Machine Translation by Jointly Learning to Align and Translate"
    author: "Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014)"
    url: "https://arxiv.org/abs/1409.0473"
    type: paper
  - title: "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention"
    author: "PyTorch Tutorials (Sean Robertson)"
    url: "https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html"
    type: docs


status: draft
---

Many language tasks map one sequence to another sequence of a different length: translation, summarization, question answering. A language model alone does not fit, because the output is conditioned on a separate input. The sequence-to-sequence (seq2seq) architecture from 2014 solves this with two networks: an **encoder** that reads the input and a **decoder** that writes the output. It was the state of the art for machine translation before transformers, and the transformer itself kept the encoder-decoder layout.

## The architecture

1. The **encoder** is an RNN (here a GRU) that reads the source tokens one by one. Its final hidden state is a fixed-size vector that should summarize the whole source sentence. It is often called the **context vector**.
2. The **decoder** is a second RNN whose initial hidden state is that context vector. It generates the target sentence one token at a time, like a language model, feeding each produced token back in as the next input.

Two special tokens make this work:

- `<sos>` (start of sequence) is the decoder's first input. It has to predict the first word from the context vector alone.
- `<eos>` (end of sequence) is appended to every target sentence. When the decoder produces it, generation stops. Without it, the model has no way to decide how long the output should be.

My first notebook skipped both: it looked up `french_vocab["SOS"]`, which did not exist, and it had no end token, so it could never have stopped on its own. Both tokens have to be in the vocabulary from the start.

## A minimal implementation

The toy task translates three English phrases into French. It is far too small to generalize, but it exercises every part of the pipeline.

```python
import torch, torch.nn as nn

pairs = [("hello world", "bonjour le monde"),
         ("how are you", "comment allez vous"),
         ("goodbye", "au revoir")]
SOS, EOS = 0, 1

def build_vocab(sentences):
    vocab = {"<sos>": SOS, "<eos>": EOS}
    for s in sentences:
        for w in s.split():
            vocab.setdefault(w, len(vocab))
    return vocab

src_vocab = build_vocab(p[0] for p in pairs)
tgt_vocab = build_vocab(p[1] for p in pairs)
tgt_itos = {i: w for w, i in tgt_vocab.items()}

def to_ids(sentence, vocab):
    return torch.tensor([vocab[w] for w in sentence.split()] + [EOS])
```

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, src):                       # src: (1, S)
        outputs, h = self.gru(self.embed(src))    # outputs: (1, S, H), h: (1, 1, H)
        return outputs, h

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, token, h):                  # token: (1, 1)
        o, h = self.gru(self.embed(token), h)
        return self.out(o[:, -1]), h              # logits: (1, vocab)
```

The GRU processes the whole source sentence in one call. The decoder is called once per output token because each step depends on the previous output.

## Training with teacher forcing

During training we know the correct target sentence. **Teacher forcing** feeds the decoder the *correct* previous token at each step, not the token it just predicted. This keeps early mistakes from derailing the rest of the sequence and makes training much faster and more stable.

```python
enc, dec = Encoder(len(src_vocab), 32), Decoder(len(tgt_vocab), 32)
params = list(enc.parameters()) + list(dec.parameters())
opt = torch.optim.Adam(params, lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    for src_s, tgt_s in pairs:
        src, tgt = to_ids(src_s, src_vocab)[None], to_ids(tgt_s, tgt_vocab)
        _, h = enc(src)                           # context vector
        token, loss = torch.tensor([[SOS]]), 0.0
        for t in range(len(tgt)):
            logits, h = dec(token, h)
            loss = loss + loss_fn(logits, tgt[t:t + 1])
            token = tgt[t].view(1, 1)             # teacher forcing
        opt.zero_grad()
        (loss / len(tgt)).backward()              # gradients flow into the encoder too
        opt.step()
```

The single loss trains both networks: the gradient flows from the decoder's predictions back through the context vector into the encoder. My original version used `LogSoftmax` plus `NLLLoss`; that is mathematically the same as `CrossEntropyLoss` on raw logits.

Teacher forcing has a cost called **exposure bias**: at test time the decoder must consume its own predictions, which it never did during training. A common compromise, scheduled sampling, mixes in the model's own predictions with growing probability as training goes on.

## Decoding

At inference time there is no target, so the decoder feeds its own output back in. The simplest strategy is **greedy decoding**: always take the most likely token, stop at `<eos>` or a length limit.

```python
@torch.no_grad()
def translate(sentence, max_len=10):
    _, h = enc(to_ids(sentence, src_vocab)[None])
    token, words = torch.tensor([[SOS]]), []
    for _ in range(max_len):
        logits, h = dec(token, h)
        nxt = logits.argmax(-1).item()
        if nxt == EOS:
            break
        words.append(tgt_itos[nxt])
        token = torch.tensor([[nxt]])
    return " ".join(words)

print(translate("how are you"))   # comment allez vous
```

Greedy decoding can commit early to a word that leads nowhere. **Beam search** keeps the $k$ best partial translations at every step and usually finds better outputs.

## The bottleneck, and the step to attention

The whole source sentence has to fit into one hidden vector of fixed size. For a three-word phrase that is fine. For a 50-word sentence, the decoder must recover every detail from the same small vector, and translation quality falls off sharply as sentences get longer.

Bahdanau, Cho and Bengio (2014) removed the bottleneck by keeping **all** encoder hidden states $h_1, \dots, h_S$ (the `outputs` tensor the encoder above already returns) and letting the decoder look back at them at every step. At step $t$ the decoder scores how relevant each encoder state is to its current state $s_{t-1}$, turns the scores into weights with a softmax, and builds its own context vector as a weighted sum:

$$
\alpha_{t,i} = \frac{\exp(\operatorname{score}(s_{t-1}, h_i))}{\sum_{j=1}^{S} \exp(\operatorname{score}(s_{t-1}, h_j))}, \qquad
c_t = \sum_{i=1}^{S} \alpha_{t,i}\, h_i .
$$

The weights $\alpha_{t,i}$ act like a soft alignment between target and source words. This "look back and take a weighted average" operation is attention. The transformer took the next step: drop the RNNs entirely and build the whole model from attention. That story continues in the [self-attention note](/notes/self-attention/).

## Key takeaways

- A seq2seq model has an encoder RNN that summarizes the source and a decoder RNN that generates the target, conditioned on that summary.
- `<sos>` starts decoding and `<eos>` ends it; both must be in the target vocabulary.
- Teacher forcing feeds the true previous token during training. It speeds up learning but creates a train/test mismatch (exposure bias).
- Greedy decoding is simple; beam search explores several candidates and usually translates better.
- Squeezing the source into one vector is a bottleneck for long inputs. Attention fixes it by letting the decoder look at all encoder states.

## Resources

- Ilya Sutskever, Oriol Vinyals and Quoc V. Le, "Sequence to Sequence Learning with Neural Networks", 2014: https://arxiv.org/abs/1409.3215
- Kyunghyun Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (introduced the GRU), 2014: https://arxiv.org/abs/1406.1078
- Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate", 2014: https://arxiv.org/abs/1409.0473
- PyTorch tutorial, "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention": https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
