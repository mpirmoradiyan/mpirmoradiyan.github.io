---
title: "Sequence-to-sequence models: an encoder, a decoder and teacher forcing"
slug: seq2seq-encoder-decoder
date: 2025-04-15
topic: NLP & LLMs
tags: [seq2seq, encoder-decoder, gru, teacher-forcing, machine-translation, attention, pytorch]
summary: "How an RNN encoder squeezes a sentence into one vector and an RNN decoder writes the translation from it, what teacher forcing and start/end tokens are for, and why that one-vector bottleneck led to attention."
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

Think about translation. "Goodbye" is one word in English and two in French ("au revoir"). The output has a different length from the input, and it has to depend on that input, not just on the words written so far. A plain language model doesn't quite fit. The sequence-to-sequence (seq2seq) setup from 2014 handles this with two networks: an **encoder** that reads the input and a **decoder** that writes the output. The same goes for summarization and question answering. Seq2seq (later with attention added) was the leading approach to machine translation before transformers, and the original transformer kept the same encoder-decoder layout.

## How it's put together

1. The **encoder** is a [recurrent neural network](/notes/recurrent-neural-networks/) (here a GRU, a simpler cousin of the [LSTM](/notes/lstm/)) that reads the source tokens one by one. Its last hidden state is a fixed-size vector that's supposed to sum up the whole source sentence. People often call it the **context vector**.
2. The **decoder** is a second RNN that starts from that context vector as its hidden state. It writes the target sentence one token at a time, like a language model, feeding each token it produces back in as the next input.

Two special tokens make this work:

- `<sos>` (start of sequence) is the decoder's first input. From it, the decoder has to guess the first word using only the context vector.
- `<eos>` (end of sequence) is added to the end of every target sentence. When the decoder outputs it, generation stops. Without it, the model has no way to know how long the output should be.

My first notebook got both wrong: it looked up `french_vocab["SOS"]`, which didn't exist, and it had no end token, so it could never have stopped on its own. Both tokens need to be in the vocabulary from the start.

## A minimal implementation

The toy task translates three English phrases into French. It's way too small to learn anything general, but it touches every part of the pipeline.

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

The encoder's GRU reads the whole source sentence in one call. The decoder has to be called once per output token, because each step depends on what came out the step before.

## Training with teacher forcing

Here's the problem. Early in training the decoder guesses wrong almost every time. If each wrong guess becomes the next input, the rest of the sentence is built on garbage. But during training we know the right target sentence. **Teacher forcing** means feeding the decoder the *correct* previous token at each step, not the token it just guessed. That way one early mistake doesn't throw off the rest of the sentence, and training gets much faster and more stable.

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

What clicked for me here is that one loss trains both networks. The gradient flows from the decoder's predictions back through the context vector into the encoder. My original version used `LogSoftmax` plus `NLLLoss`, which is mathematically the same as `CrossEntropyLoss` on raw logits.

Teacher forcing has a downside called **exposure bias**: at test time the decoder has to work from its own guesses, which it never had to do during training. A common middle ground, scheduled sampling, mixes in the model's own guesses more and more often as training goes on.

## Decoding

When actually using the model there's no target, so the decoder feeds its own output back in. The simplest way is **greedy decoding**: always take the most likely token, and stop at `<eos>` or at a length limit.

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

Greedy decoding can lock in a word early that leads nowhere. **Beam search** keeps the $k$ best partial translations at every step and usually ends up with better results.

## The bottleneck, and how it led to attention

There's one weak spot in all this. The whole source sentence has to fit into one hidden vector of fixed size. For a three-word phrase that's fine. For a 50-word sentence, the decoder has to dig every detail out of the same small vector, and translation quality drops fast as sentences get longer.

Bahdanau, Cho and Bengio (2014) got rid of the bottleneck. They kept **all** the encoder hidden states $h_1, \dots, h_S$ (the `outputs` tensor the encoder above already returns) and let the decoder look back at them at every step. At step $t$, the decoder scores how relevant each encoder state is to its current state $s_{t-1}$, turns the scores into weights with a softmax, and builds its own context vector as a weighted sum:

$$
\alpha_{t,i} = \frac{\exp(\operatorname{score}(s_{t-1}, h_i))}{\sum_{j=1}^{S} \exp(\operatorname{score}(s_{t-1}, h_j))}, \qquad
c_t = \sum_{i=1}^{S} \alpha_{t,i}\, h_i .
$$

The way I think about it: the weights $\alpha_{t,i}$ are a soft matching between target words and source words. This "look back and take a weighted average" step is attention. The transformer went one step further: drop the RNNs completely and build the whole model out of attention. That story continues in the [self-attention note](/notes/self-attention/).

## Key takeaways

- A seq2seq model has an encoder RNN that sums up the source and a decoder RNN that writes the target based on that summary.
- `<sos>` starts decoding and `<eos>` ends it. Both have to be in the target vocabulary.
- Teacher forcing feeds the true previous token during training. It speeds up learning but means training and testing don't match (exposure bias).
- Greedy decoding is simple. Beam search tries several candidates and usually translates better.
- Squeezing the source into one vector is a bottleneck for long inputs. Attention fixes it by letting the decoder look at all the encoder states.

## Resources

- Ilya Sutskever, Oriol Vinyals and Quoc V. Le, "Sequence to Sequence Learning with Neural Networks", 2014: https://arxiv.org/abs/1409.3215
- Kyunghyun Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (introduced the GRU), 2014: https://arxiv.org/abs/1406.1078
- Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate", 2014: https://arxiv.org/abs/1409.0473
- PyTorch tutorial, "NLP From Scratch: Translation with a Sequence to Sequence Network and Attention": https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
