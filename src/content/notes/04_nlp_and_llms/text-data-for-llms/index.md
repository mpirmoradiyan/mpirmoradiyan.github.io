---
title: "Preparing text for LLM training: sliding windows, batches and input embeddings"
slug: text-data-for-llms
date: 2025-01-09
topic: NLP & LLMs
tags: [llm, data-loader, next-token-prediction, sliding-window, embeddings, pytorch, tiktoken]
summary: "How a long text becomes (input, target) pairs for next-token prediction, why one chunk of T tokens is really T examples, and how token ids turn into the vectors a GPT reads."
sources:
  - title: "Build a Large Language Model (From Scratch), Chapter 2: Working with text data"
    author: "Sebastian Raschka (Manning, 2024)"
    url: "https://www.manning.com/books/build-a-large-language-model-from-scratch"
    type: book
  - title: "LLMs-from-scratch"
    author: "Sebastian Raschka"
    url: "https://github.com/rasbt/LLMs-from-scratch"
    type: repo
  - title: "Let's build GPT: from scratch, in code, spelled out"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=kCc8FmEb1nY"
    type: video


status: draft
---

A GPT learns just one thing: guess the next token. Sounds simple. But before writing any model code, I got stuck on a more basic question: how does a plain text file turn into training examples of the right shape? These are my notes on that path: text file, then batches of token ids, then the vectors the first transformer layer actually gets. How text becomes ids in the first place is in my [tokenization note](/notes/tokenization-bpe/).

## Step 1: the whole text becomes one long list of ids

The example text is *The Verdict*, a short story with 20,479 characters. The GPT-2 tokenizer turns it into 5,145 token ids.

```python
import tiktoken
raw_text = open("the-verdict.txt", encoding="utf-8").read()
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))   # 5145
```

With a big collection of texts you do the same thing one document at a time, and put an `<|endoftext|>` token between documents. That way the model can tell where one stops and the next starts.

## Inputs and targets: just shift by one

So what's the input and what's the answer? Take `context_length` tokens in a row as the input `x`. The target `y` is the same window moved one token to the right. That's it.

```python
context_length = 4
x = enc_text[50:54]   # [290, 4920, 2241, 287]
y = enc_text[51:55]   # [4920, 2241, 287, 257]
```

The part that surprised me: this one pair is really four training examples, one per position:

| context | target |
|---|---|
| " and" | " established" |
| " and established" | " himself" |
| " and established himself" | " in" |
| " and established himself in" | " a" |

A causal transformer (one that can only look backwards) makes a prediction at every position at the same time. The attention mask makes sure position $t$ only sees tokens up to $t$. So one forward pass over a chunk of length $T$ gives $T$ predictions and $T$ loss terms. That's cheap, and it also teaches the model to handle any context length from 1 up to $T$. It needs that later, when it starts generating from a short prompt.

## Sliding a window over the text

One window isn't much. To use the whole text, you slide the window along the list of ids. The **stride** is how far the window jumps each time.

```python
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.inputs, self.targets = [], []
        for i in range(0, len(token_ids) - max_length, stride):
            self.inputs.append(torch.tensor(token_ids[i:i + max_length]))
            self.targets.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

ds = GPTDataset(enc_text, max_length=4, stride=4)
loader = DataLoader(ds, batch_size=8, shuffle=False, drop_last=True)
inputs, targets = next(iter(loader))
print(inputs.shape, targets.shape)   # torch.Size([8, 4]) torch.Size([8, 4])
```

- `stride = 1` means the windows overlap as much as possible: the second window is just the first one moved by one token. You get lots of samples, but they're almost copies of each other, so on a small text the model overfits more easily.
- `stride = max_length` means the windows don't overlap, so each token is seen exactly once per epoch.
- `drop_last=True` throws away the last batch if it's shorter than the others. Otherwise that small batch can cause a jumpy loss value.

A batch is a tensor of shape (B, T): B separate chunks, each T tokens long. B is the **batch size** and T is the **context length** (Karpathy's code calls it `block_size`). Keeping these two letters straight helped me a lot when reading the shapes later.

Karpathy's GPT notebook does it another way. There's no dataset class; it just picks random start positions at every step:

```python
def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
```

Both give the same (B, T) shapes. The random version is handy when the whole text is already one big tensor.

## From ids to vectors: the token embedding

Now the batch is a grid of integers. But token ids are just labels. Id 4920 isn't "bigger" or "more" than id 290 in any useful sense. So the first layer of the model is an **embedding table**: a matrix with one learnable row per token. Looking up a row is the same as multiplying a one-hot vector (all zeros except a single 1) by the matrix, just faster.

```python
vocab_size, emb_dim = 50257, 256
tok_emb = torch.nn.Embedding(vocab_size, emb_dim)
token_embeddings = tok_emb(inputs)
print(token_embeddings.shape)        # torch.Size([8, 4, 256])
```

The rows start out random and get trained along with the rest of the model. Over time, tokens that are used in similar ways end up with similar vectors. GPT-2 small uses 768 dimensions and the biggest GPT-3 model uses 12,288. The 256 here is just a small size to play with.

## Telling the model where each token is

This tripped me up at first: self-attention has no idea about order. It treats its input like a bag of tokens. If you shuffle the tokens (with no causal mask), each output gets shuffled the same way but doesn't otherwise change. So word order has to be added by hand. GPT-2 does this with **learned absolute position embeddings**: a second table with one row per position, added on top of the token embeddings.

```python
pos_emb = torch.nn.Embedding(context_length, emb_dim)
pos_embeddings = pos_emb(torch.arange(context_length))   # (4, 256)
input_embeddings = token_embeddings + pos_embeddings     # broadcasts to (8, 4, 256)
```

The (T, 256) position tensor is broadcast (copied automatically) across the batch, so every sequence gets the same position vectors. The result has shape (B, T, emb_dim), and that's what goes into the first transformer block. There are other ways to encode position, like the fixed sine waves of the original transformer or rotary embeddings, but GPT-2 uses the learned table.

## Key takeaways

- Next-token training data is just the token list, with the targets shifted one step to the right.
- A chunk of T tokens is really T training examples, because a causal model predicts at every position at once.
- Sliding a window with some stride turns a long text into (B, T) batches. The stride decides how much windows overlap.
- An embedding table turns each token id into a learnable vector of size `emb_dim`.
- Position has to be added separately. GPT-2 adds a learned vector for each position to each token embedding.

## Resources

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 2 "Working with text data": https://www.manning.com/books/build-a-large-language-model-from-scratch. The dataset class and the use of *The Verdict* follow this chapter.
- Sebastian Raschka, LLMs-from-scratch (code repository): https://github.com/rasbt/LLMs-from-scratch
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY. Source of the `get_batch` approach.
- PyTorch documentation, `torch.utils.data.DataLoader`: https://pytorch.org/docs/stable/data.html
