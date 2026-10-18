---
title: "Preparing text for LLM training: sliding windows, batches and input embeddings"
slug: text-data-for-llms
date: 2025-01-09
topic: NLP & LLMs
tags: [llm, data-loader, next-token-prediction, sliding-window, embeddings, pytorch, tiktoken]
summary: "How a long text is turned into (input, target) pairs for next-token prediction, why one chunk of length T holds T training examples, and how token ids become the vectors a GPT actually reads."
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

A GPT is trained on one task only: predict the next token. Before any model code, the raw text has to be turned into a stream of training examples in exactly the right shape. This note goes from a text file to batches of token ids, and from token ids to the input vectors the first transformer layer receives. Tokenization itself (how text becomes ids) is covered in the [tokenization note](/notes/tokenization-bpe/).

## From text to one long list of token ids

The example corpus is *The Verdict*, a short story of 20,479 characters. The GPT-2 tokenizer turns it into 5,145 token ids.

```python
import tiktoken
raw_text = open("the-verdict.txt", encoding="utf-8").read()
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))   # 5145
```

For a large corpus the same happens document by document, with `<|endoftext|>` inserted between documents so the model can tell where one ends and the next begins.

## Inputs and targets: shift by one

Take a window of `context_length` tokens as the input `x`. The target `y` is the same window shifted one token to the right.

```python
context_length = 4
x = enc_text[50:54]   # [290, 4920, 2241, 287]
y = enc_text[51:55]   # [4920, 2241, 287, 257]
```

This one pair actually contains four training examples, one per position:

| context | target |
|---|---|
| " and" | " established" |
| " and established" | " himself" |
| " and established himself" | " in" |
| " and established himself in" | " a" |

A causal transformer predicts at every position in parallel, and the attention mask makes sure position $t$ only sees tokens up to $t$. So one forward pass over a chunk of length $T$ produces $T$ predictions and $T$ loss terms. This is efficient, and it also trains the model to work with any context length from 1 up to $T$, which is what it needs when it generates text starting from a short prompt.

## A sliding-window dataset

To cover the whole text, slide the window along the token list. The **stride** is how far the window moves each time.

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

- `stride = 1` gives maximally overlapping windows: the second window is the first one shifted by a single token. It produces many samples but they are highly redundant, which encourages overfitting on a small text.
- `stride = max_length` gives non-overlapping windows that cover each token exactly once per epoch.
- `drop_last=True` discards a final short batch, which would otherwise give a noisy loss spike.

A batch is a tensor of shape (B, T): B independent chunks, each T tokens long. B is the **batch size** and T is the **context length** (Karpathy's code calls it `block_size`).

An alternative, used in Karpathy's GPT notebook, skips the dataset class and samples random start positions on every step:

```python
def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
```

Both give the same (B, T) shapes; the random version is convenient when the corpus is one big tensor.

## From ids to vectors: the token embedding

Token ids are arbitrary labels; id 4920 is not "bigger" than id 290. The first layer of the model is an **embedding table**: a matrix with one learnable row per token. Looking up a row is the same as multiplying a one-hot vector by the matrix, just faster.

```python
vocab_size, emb_dim = 50257, 256
tok_emb = torch.nn.Embedding(vocab_size, emb_dim)
token_embeddings = tok_emb(inputs)
print(token_embeddings.shape)        # torch.Size([8, 4, 256])
```

The rows start random and are trained with the rest of the model, so tokens used in similar ways end up with similar vectors. GPT-2 small uses 768 dimensions and the largest GPT-3 model 12,288; 256 is just a small size for experiments.

## Telling the model where each token is

Self-attention treats its input as a set: if you shuffle the tokens, each token's output is shuffled the same way but otherwise unchanged. Word order has to be injected explicitly. GPT-2 uses **learned absolute position embeddings**: a second table with one row per position, added to the token embeddings.

```python
pos_emb = torch.nn.Embedding(context_length, emb_dim)
pos_embeddings = pos_emb(torch.arange(context_length))   # (4, 256)
input_embeddings = token_embeddings + pos_embeddings     # broadcasts to (8, 4, 256)
```

The (T, 256) position tensor broadcasts across the batch dimension, so every sequence gets the same position vectors. The result, of shape (B, T, emb_dim), is what the first transformer block receives. Other ways to encode position (fixed sinusoids, rotary embeddings) are covered in a separate note on positional encoding.

## Key takeaways

- Next-token training data is simply the token stream with targets shifted one position to the right.
- A chunk of length T holds T training examples, because a causal model predicts at every position at once.
- A sliding window with a chosen stride turns a long text into (B, T) batches of inputs and targets; the stride controls overlap.
- An embedding table turns each token id into a learnable vector of size `emb_dim`.
- Position information is added separately; GPT-2 adds a learned vector per position to each token embedding.

## Resources

- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 2 "Working with text data": https://www.manning.com/books/build-a-large-language-model-from-scratch. The dataset class and the use of *The Verdict* follow this chapter.
- Sebastian Raschka, LLMs-from-scratch (code repository): https://github.com/rasbt/LLMs-from-scratch
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video): https://www.youtube.com/watch?v=kCc8FmEb1nY. Source of the `get_batch` approach.
- PyTorch documentation, `torch.utils.data.DataLoader`: https://pytorch.org/docs/stable/data.html
