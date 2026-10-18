---
title: "Building a small GPT from scratch, one improvement at a time"
slug: gpt-from-scratch
date: 2025-01-09
topic: NLP & LLMs
tags: [gpt, transformer, decoder-only, layer-normalization, residual-connections, dropout, pytorch]
summary: "How a small character-level GPT grows out of a bigram model by adding position embeddings, self-attention, several heads, feed-forward layers, residual connections and layer norm, checking the validation loss after each step."
sources:
  - title: "Let's build GPT: from scratch, in code, spelled out"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=kCc8FmEb1nY"
    type: video
  - title: "ng-video-lecture (code for the lecture)"
    author: "Andrej Karpathy"
    url: "https://github.com/karpathy/ng-video-lecture"
    type: repo
  - title: "Attention Is All You Need"
    author: "Ashish Vaswani et al. (2017)"
    url: "https://arxiv.org/abs/1706.03762"
    type: paper
  - title: "Layer Normalization"
    author: "Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016)"
    url: "https://arxiv.org/abs/1607.06450"
    type: paper
  - title: "Deep Residual Learning for Image Recognition"
    author: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)"
    url: "https://arxiv.org/abs/1512.03385"
    type: paper


status: draft
---

A GPT is a stack of identical transformer blocks on top of an embedding layer, trained to guess the next token. That one sentence hides a handful of design choices, and I wanted to see what each one actually does. So I started from the simplest possible language model and added the pieces one at a time, checking the validation loss after each change. The data is Tiny Shakespeare, about 1.1 million characters of plays, handled one character at a time with a vocabulary of 65 symbols.

## The setup

- **Tokens**: single characters, turned into integers 0-64.
- **Split**: the first 90% of the text for training, the last 10% for validation.
- **Batches**: random chunks of `block_size` characters. The targets are the same chunks shifted by one (see the [text data note](/notes/text-data-for-llms/)).
- **Loss**: cross-entropy between the logits at every position and the next character. Pure guessing gives $\ln 65 \approx 4.17$.
- **Evaluation**: the average loss over 200 random batches from each split, in `model.eval()` mode and under `torch.no_grad()`. One batch is way too noisy to compare models with.

## Step 0: the bigram baseline

The simplest model is one embedding table of size (65, 65). Row $i$ holds the logits (raw scores) for the character that comes after character $i$. It ignores everything except the current character.

```python
class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)
    def forward(self, idx):                  # (B, T)
        return self.table(idx)               # (B, T, vocab) logits
```

Trained with AdamW it gets stuck at a validation loss of about **2.48** and writes gibberish, though with roughly the right letter frequencies.

Generating text works the same way for every model here: run the model on the context, take the logits at the last position, sample a character, append it, repeat. Once the model has position embeddings, you have to cut the context down to the last `block_size` tokens, because there's no position embedding past that.

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size):
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])          # crop the context
        probs = F.softmax(logits[:, -1, :], dim=-1)   # last position only
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx
```

## The path from bigram to GPT

Steps 1 to 4 below all use `block_size = 8`, embedding size 32, batch size 32, learning rate $10^{-3}$ and 5,000 steps, so the only thing that changes is the architecture. (The bigram baseline was trained with a learning rate of $10^{-2}$ for 3,000 steps, and it had already stopped improving.) The losses are the last values logged, around step 4,500 to 5,000.

| step | change | val loss |
|---|---|---|
| 0 | bigram lookup table | 2.47-2.49 |
| 1 | token + position embeddings, one self-attention head (size 32) | 2.40 |
| 2 | four heads of size 8 instead of one of size 32 | 2.29 |
| 3 | 3 transformer blocks: attention + feed-forward, with residual connections | 2.11 |
| 4 | layer normalization before each sub-layer, and a final layer norm | 2.06 |

**Step 1: let the tokens talk.** Each position gets a token embedding plus a learned position embedding, then one head of [causal self-attention](/notes/causal-attention/), then a linear layer to get logits. Now a character can use the seven characters before it, not just one.

**Step 2: several heads.** [Multi-head attention](/notes/multi-head-attention/) lets different heads keep track of different kinds of relationships. The total width stays 32.

**Step 3: blocks that think, plus shortcuts.** The way I think about it: attention is *communication*, where tokens collect information from each other. After that, each token needs to *think* about what it collected. That's the job of a small **feed-forward network** applied to each position on its own, with a hidden layer four times wider than the model:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)
```

Stacking several blocks makes the network deep, and deep networks are hard to train. **Residual connections** fix this. Each sub-layer adds its output to its input, `x = x + sublayer(x)`, instead of replacing it. In backprop, the addition passes the gradient straight through to earlier layers, so there's always a clean path from the loss back to the first block. At the start the sub-layers add very little, so the network acts almost like a shallow one, and then the blocks slowly learn to add useful corrections. This step gave the biggest single jump.

**Step 4: layer normalization.** [Batch normalization](/notes/batch-normalization/) normalizes each feature across the batch. **Layer normalization** flips that around: it normalizes each token's vector across its own features:

$$
\operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\qquad \mu = \frac{1}{C}\sum_{j=1}^{C} x_j,\quad \sigma^2 = \frac{1}{C}\sum_{j=1}^{C}(x_j - \mu)^2
$$

Here $x$ is one token's vector of $C$ features, $\gamma$ and $\beta$ are a learned gain and shift, and $\epsilon$ is a small constant. Since the statistics come from a single token, it doesn't depend on the other examples in the batch, there are no running averages, and it behaves the same in training and inference. That fits sequences of different lengths nicely. I checked this on a toy (32, 100) tensor: after layer norm, each row had mean 0 and standard deviation 1, but the columns didn't.

The original transformer put layer norm *after* each residual addition. GPT-2 and later models put it *before* each sub-layer ("pre-norm"), which trains more smoothly:

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # communicate
        x = x + self.ffwd(self.ln2(x))   # compute
        return x
```

So the full model is: token embedding + position embedding, a stack of blocks, a final layer norm, and a linear "language-model head" that gives the logits.

## Scaling up, and a lesson about overfitting

Then I scaled the same code up: a context of 256 characters, embedding size 384, 6 heads, 6 blocks, dropout 0.2, learning rate $3 \times 10^{-4}$, batch size 64. All linear and embedding weights started from $\mathcal{N}(0, 0.02^2)$. That's 10.8 million parameters, and you need a GPU for it.

| step | train loss | val loss |
|---|---|---|
| 0 | 4.22 | 4.23 |
| 1,000 | 1.39 | 1.60 |
| 3,000 | 1.07 | **1.48** |
| 5,000 | 0.86 | 1.56 |

The validation loss hit its lowest point, about **1.48**, around step 3,000. After that the training loss kept dropping while the validation loss went up: the model had started memorizing the training plays. Even with dropout, 10.8M parameters are enough to memorize 1M characters. The fix is to keep the checkpoint with the best validation loss (early stopping), use more data, or add more regularization.

The samples changed a lot along the way, and that was the fun part. The bigram wrote `xiv&dbHU?YhSqNY!`. The final model wrote text that looks like a play: speaker names in capitals, line breaks, mostly real words and something like Shakespeare's rhythm ("To prevent it, as I love this country's cause."), even though the sentences make no sense.

## What's still missing compared to ChatGPT

This model is a **decoder-only** transformer, like GPT. It has no encoder and no cross-attention, because all it does is continue text. It's basically a tiny pretrained model. A chat assistant adds a much bigger pretraining run with a subword tokenizer, then fine-tuning on conversations and training on human preferences. Those later steps are what turn a "finish this document" model into an assistant.

## Key takeaways

- A GPT is token + position embeddings, a stack of blocks (masked multi-head attention and a feed-forward MLP), a final layer norm and a linear head.
- Attention moves information between positions. The feed-forward layer works on it inside each position.
- Residual connections give gradients a direct path through all the layers, and they were the biggest single gain here.
- Layer norm normalizes each token over its own features, so it doesn't depend on the batch. GPT-style models put it before each sub-layer.
- Keep an eye on the validation loss: the 10.8M-parameter model was best at step 3,000 (1.48) and overfit after that.

## Resources

- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY. The progression and the model code are adapted from this lecture.
- Andrej Karpathy, ng-video-lecture (lecture code and the Tiny Shakespeare dataset): https://github.com/karpathy/ng-video-lecture
- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Jimmy Lei Ba, Jamie Ryan Kiros and Geoffrey Hinton, "Layer Normalization", 2016: https://arxiv.org/abs/1607.06450
- Kaiming He et al., "Deep Residual Learning for Image Recognition", 2015: https://arxiv.org/abs/1512.03385
- Tom Brown et al., "Language Models are Few-Shot Learners" (GPT-3), 2020: https://arxiv.org/abs/2005.14165
