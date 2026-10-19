---
title: "Tokenization and byte-pair encoding: how text becomes integers for an LLM"
slug: tokenization-bpe
date: 2025-02-28
topic: NLP & LLMs
tags: [tokenization, bpe, byte-pair-encoding, utf-8, tiktoken, gpt-2, special-tokens]
summary: "Why LLMs read neither single characters nor whole words, how byte-pair encoding builds word pieces out of UTF-8 bytes, and what extra tricks GPT tokenizers add on top."
sources:
  - title: "Let's build the GPT Tokenizer"
    author: "Andrej Karpathy (Neural Networks: Zero to Hero)"
    url: "https://www.youtube.com/watch?v=zduSFxRajkE"
    type: video
  - title: "minbpe"
    author: "Andrej Karpathy"
    url: "https://github.com/karpathy/minbpe"
    type: repo
  - title: "Build a Large Language Model (From Scratch), Chapter 2: Working with text data"
    author: "Sebastian Raschka (Manning, 2024)"
    url: "https://www.manning.com/books/build-a-large-language-model-from-scratch"
    type: book
  - title: "Neural Machine Translation of Rare Words with Subword Units"
    author: "Rico Sennrich, Barry Haddow, Alexandra Birch (2016)"
    url: "https://arxiv.org/abs/1508.07909"
    type: paper
  - title: "tiktoken"
    author: "OpenAI"
    url: "https://github.com/openai/tiktoken"
    type: repo


status: draft
---

A language model never sees text. It only sees integers. The tokenizer is the thing that chops text into pieces and gives each piece a number. I used to think of it as a boring preprocessing step. It isn't. It decides how long sequences get, how much text fits in the context window, and how well the model copes with rare words, code, numbers and non-English text. A lot of weird LLM behaviour, like being bad at spelling or arithmetic, comes back to tokenization. So here's the basic trade-off, the byte-pair encoding (BPE) algorithm the GPT models use, and the extra pieces around it.

## Characters, words, or something in between?

The first question is what a "piece" should be. There are two obvious choices, and both have problems.

- **Characters.** The vocabulary is tiny (65 symbols for the Shakespeare text in the GPT note), but sequences get long. 1,024 characters is only about a paragraph, and attention gets more expensive as sequences grow.
- **Words.** Sequences are short, but the vocabulary is huge and still never complete. A simple word-level tokenizer built from the short story *The Verdict* (split on spaces and punctuation with a regex) had 1,130 entries. It broke on the very first word of "Hello, do you like tea?", because "Hello" never shows up in the story. The usual fix is an `<|unk|>` ("unknown") token, but then the model has no idea what the word actually was.

**Subword tokenization** is the middle ground. Common words get their own token, and rare words get split into a few known pieces. Byte-pair encoding is the most common way to learn those pieces.

## Start from bytes, not characters

Unicode has well over 100,000 characters. Way too many for a starting vocabulary. A Python string is a list of Unicode code points (one number per character), and `str.encode("utf-8")` turns it into bytes. UTF-8 uses 1 byte for plain ASCII and 2 to 4 bytes for everything else. So if you work with bytes, you start with exactly 256 symbols and can still write any text at all, emoji included. The catch is that non-ASCII text gets long. The merges below fix that.

```python
text = "Ｕｎｉｃｏｄｅ! 😄 The very name strikes fear..."
ids = list(text.encode("utf-8"))   # integers in 0..255
```

## How BPE works

BPE is basically a compression trick, run in a loop:

1. Count every pair of tokens that sit next to each other.
2. Take the most common pair and give it a new token id (256, 257, ...).
3. Replace every place that pair shows up with the new id.
4. Repeat until the vocabulary is as big as you want.

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges = {}                               # (int, int) -> new id
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        merges[pair] = 256 + i
        ids = merge(ids, pair, 256 + i)
    return merges
```

On a short paragraph about Unicode (616 bytes), the first merge was `(101, 32)`, which is `"e "`: the letter e followed by a space, the most common word ending in English. After 20 merges the paragraph was 451 tokens long. On a longer text, 20 merges shrank 24,532 bytes to 19,419 tokens, a compression ratio of 1.26. The fun part is that merged tokens can be merged again: `(116, 104)` = `th`, and later `(th, "e ")` became a single token for `"the "`.

My first version of `merge` used `enumerate` and `list.pop` inside the loop. It passed my unit tests, but changing a list while you loop over it is asking for trouble. The `while` loop with an explicit index is clearer and safe.

## Encoding and decoding

**Decoding** is the easy direction. Turn each id back into its bytes and glue them together. The bytes for a merged id are just the bytes of its two parents, built up in the order the merges were made:

```python
def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():        # dicts keep insertion order
        vocab[idx] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

Don't skip `errors="replace"`. Not every byte sequence is valid UTF-8 (a lone byte 128 isn't, for example). If the model spits out something like that, strict decoding crashes. Real tokenizers just swap the bad bytes for `�`.

**Encoding** new text is the trickier direction. You have to apply the merges in the same order they were learned, because later merges are built on top of earlier ones. So at each step, find the neighbouring pair with the lowest merge index and merge it. Stop when no pair is in the merge table.

```python
def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                             # nothing left to merge
        ids = merge(ids, pair, merges[pair])
    return ids

assert decode(encode("hello world!", merges), merges) == "hello world!"
```

Something I hadn't realised before: the tokenizer is a separate thing from the LLM. It's trained once, on its own data (which can be different from the model's training data), and then frozen. The LLM only ever sees the token ids.

## GPT-style splitting with a regex first

Plain BPE doesn't care about word boundaries or punctuation. It will happily make `"dog."`, `"dog!"` and `"dog?"` into three separate tokens. GPT-2 avoids this by first cutting the text into chunks with a regular expression, and then running BPE only inside each chunk:

```python
import regex as re   # the third-party 'regex' module supports \p{L}
gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
print(re.findall(gpt2pat, "Hello world! you're better"))
# ['Hello', ' world', '!', ' you', "'re", ' better']
```

In plain words, the pattern keeps common English contractions separate, groups letters together (with an optional space in front), keeps numbers apart from letters, and deals with runs of spaces. Two quirks I noticed: the contraction rules only match lowercase forms, so `'S` in all-caps text gets split differently, and the whole thing is built around English.

GPT-4's tokenizer (`cl100k_base`) uses an improved pattern and about 100,000 tokens. One easy-to-see difference is spaces. GPT-2 encodes `"   hello world !"` as `[220, 220, 23748, 995, 5145]`, with a token for each leading space, while `cl100k_base` gives `[256, 24748, 1917, 758]` with the spaces grouped. For indented code that's a big deal.

## Special tokens and the GPT-2 vocabulary

GPT-2 has 50,257 tokens: 256 byte tokens, 50,000 learned merges, and one **special token**, `<|endoftext|>` (id 50256). Special tokens never come out of BPE. They're added to the vocabulary by hand and handled outside the merge loop. `<|endoftext|>` goes between unrelated documents in the training data, so the model learns that what comes next has nothing to do with what came before. Chat models add more special tokens to mark who's speaking and where each turn starts and ends.

In `tiktoken` you have to allow special tokens explicitly. That way a user can't sneak one in just by typing the string:

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello, do you like tea? <|endoftext|> In the sunlit terraces",
                 allowed_special={"<|endoftext|>"})
print(enc.decode(ids))
for t in enc.encode("Akwirw ier"):
    print(t, repr(enc.decode([t])))   # 'Ak', 'w', 'ir', 'w', ' ', 'ier'
```

There's no `<|unk|>` token at all. A made-up word like "Akwirw" just gets broken into smaller known pieces, down to single bytes if needed. For me that's the nicest thing about byte-level BPE: nothing is ever "unknown".

What happens next? The ids get cut into training windows, which is in [Preparing text for LLM training](/notes/text-data-for-llms/), and the 50,257-token vocabulary sets the size of the embedding and output layers in [the GPT-2 architecture](/notes/gpt2-architecture/).

## Key takeaways

- Tokenization is a trade-off between vocabulary size and sequence length. Subwords sit between characters and words.
- Byte-level BPE starts from the 256 UTF-8 bytes and keeps merging the most common neighbouring pair into a new token.
- Encoding has to apply merges in the order they were learned. Decoding just joins bytes, but it has to cope with invalid UTF-8.
- GPT tokenizers split text with a regex first, so merges don't cross words, numbers and punctuation.
- Special tokens like `<|endoftext|>` are added by hand, outside BPE. Byte-level BPE never needs an "unknown word" token.

## Resources

- Andrej Karpathy, "Let's build the GPT Tokenizer", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=zduSFxRajkE. The BPE functions here are adapted from this lecture.
- Andrej Karpathy, minbpe (repository and exercise): https://github.com/karpathy/minbpe
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 2 (simple tokenizers, special tokens, tiktoken): https://www.manning.com/books/build-a-large-language-model-from-scratch; code: https://github.com/rasbt/LLMs-from-scratch
- Rico Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", 2016: https://arxiv.org/abs/1508.07909
- OpenAI, tiktoken: https://github.com/openai/tiktoken
- Nathan Reed, "A Programmer's Introduction to Unicode" (source of the sample text): https://www.reedbeta.com/blog/programmers-intro-to-unicode/
- Tiktokenizer (interactive token visualizer): https://tiktokenizer.vercel.app
