---
title: "Tokenization and byte-pair encoding: how text becomes integers for an LLM"
slug: tokenization-bpe
date: 2025-02-28
topic: NLP & LLMs
tags: [tokenization, bpe, byte-pair-encoding, utf-8, tiktoken, gpt-2, special-tokens]
summary: "Why LLMs neither read characters nor whole words, how byte-pair encoding builds a subword vocabulary from UTF-8 bytes, and how GPT-style tokenizers add regex pre-splitting and special tokens."
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

A language model only sees integers. The tokenizer decides how text is cut into pieces and which integer each piece gets. That choice affects almost everything downstream: how long sequences are, how much text fits in the context window, how well the model handles rare words, code, numbers and non-English text. Many odd LLM behaviours, such as trouble spelling words or doing arithmetic, trace back to tokenization. This note covers the trade-off, the byte-pair encoding (BPE) algorithm used by the GPT family, and the extra machinery around it.

## Characters, words or something in between?

There are two obvious options, and both have problems.

- **Characters.** The vocabulary is tiny (65 symbols for the Shakespeare text in the GPT note), but sequences are long. A context of 1,024 characters holds only a paragraph, and attention cost grows with sequence length.
- **Words.** Sequences are short, but the vocabulary is huge and still incomplete. A simple word-level tokenizer built from the short story *The Verdict* (splitting on spaces and punctuation with a regex) had 1,130 entries. It failed on the very first word of "Hello, do you like tea?" because "Hello" never appears in the story. The usual patch is an `<|unk|>` token for unknown words, but then the model cannot see what the word was.

**Subword tokenization** sits in between. Frequent words become single tokens, and rare words are split into a few known pieces. Byte-pair encoding is the most widely used way to learn those pieces.

## Start from bytes, not characters

Unicode has well over 100,000 characters, far too many for a base vocabulary. Python strings are sequences of Unicode code points, and `str.encode("utf-8")` turns them into bytes. UTF-8 uses 1 byte for ASCII and 2 to 4 bytes for other characters. Working on bytes gives a fixed base vocabulary of exactly 256 symbols that can represent any text, emoji and all. The downside is that non-ASCII text becomes long, which is what the merges will fix.

```python
text = "Ｕｎｉｃｏｄｅ! 😄 The very name strikes fear..."
ids = list(text.encode("utf-8"))   # integers in 0..255
```

## The BPE algorithm

BPE is a compression algorithm run in a loop:

1. Count every pair of adjacent tokens.
2. Take the most frequent pair and give it a new token id (256, 257, ...).
3. Replace every occurrence of that pair with the new id.
4. Repeat until the vocabulary has the desired size.

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

On a short paragraph about Unicode (616 bytes), the first merge was `(101, 32)`, which is `"e "`: the letter e followed by a space, the most common ending in English. After 20 merges the paragraph was 451 tokens long. On a longer text, 20 merges shrank 24,532 bytes to 19,419 tokens, a compression ratio of 1.26. Merged tokens can be merged again: `(116, 104)` = `th`, and later `(th, "e ")` became one token for `"the "`.

I first wrote `merge` with `enumerate` and `list.pop` inside the loop. It passed my unit tests, but changing a list while iterating over it is fragile. The `while` loop with an explicit index is clearer and safe.

## Encoding and decoding

**Decoding** maps each id back to its bytes and joins them. Build the byte string for each merged id from its two parents, in the order the merges were created:

```python
def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():        # dicts keep insertion order
        vocab[idx] = vocab[a] + vocab[b]
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
```

`errors="replace"` matters. Not every byte sequence is valid UTF-8; a lone byte 128, for example, is not. If the model emits such a sequence, strict decoding raises an error. Replacing invalid bytes with `�` is what real tokenizers do.

**Encoding** new text must apply the merges in the order they were learned, because later merges are built on earlier ones. At each step, find the adjacent pair with the lowest merge index and merge it; stop when no pair is in the merge table.

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

The tokenizer is a separate component from the LLM. It is trained once, on its own dataset (which may differ from the model's training data), and then frozen. The LLM only ever sees the token ids.

## GPT-style pre-splitting with a regex

Plain BPE will happily merge across word boundaries and punctuation, producing tokens like `"dog."`, `"dog!"` and `"dog?"` as separate entries. GPT-2 first splits the text into chunks with a regular expression and runs BPE inside each chunk only:

```python
import regex as re   # the third-party 'regex' module supports \p{L}
gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
print(re.findall(gpt2pat, "Hello world! you're better"))
# ['Hello', ' world', '!', ' you', "'re", ' better']
```

The pattern keeps common English contractions apart, groups letters with an optional leading space, keeps numbers separate from letters, and handles runs of whitespace. Two quirks are worth knowing: the contraction rules only match lowercase apostrophe forms, so `'S` in all-caps text is split differently, and the rules are English-centric.

GPT-4's tokenizer (`cl100k_base`) uses a refined pattern and about 100,000 tokens. One visible difference is whitespace: GPT-2 encodes `"   hello world !"` as `[220, 220, 23748, 995, 5145]`, spending a token on each leading space, while `cl100k_base` gives `[256, 24748, 1917, 758]` with the spaces grouped. That matters a lot for indented code.

## Special tokens and the GPT-2 vocabulary

GPT-2 has 50,257 tokens: 256 byte tokens, 50,000 learned merges, and one **special token**, `<|endoftext|>` (id 50256). Special tokens never come out of BPE; they are added to the vocabulary by hand and handled outside the merge loop. `<|endoftext|>` is placed between independent documents in the training data, so the model learns that what follows is unrelated to what came before. Chat models add more special tokens to mark the roles and turns of a conversation.

With `tiktoken`, special tokens must be allowed explicitly so that a user cannot inject them by typing the string:

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Hello, do you like tea? <|endoftext|> In the sunlit terraces",
                 allowed_special={"<|endoftext|>"})
print(enc.decode(ids))
for t in enc.encode("Akwirw ier"):
    print(t, repr(enc.decode([t])))   # 'Ak', 'w', 'ir', 'w', ' ', 'ier'
```

There is no `<|unk|>` token: a made-up word like "Akwirw" is simply broken into smaller known pieces, down to single bytes if necessary. That is the main practical advantage of byte-level BPE.

## Key takeaways

- Tokenization trades vocabulary size against sequence length. Subwords sit between characters and words.
- Byte-level BPE starts from the 256 UTF-8 bytes and repeatedly merges the most frequent adjacent pair into a new token.
- Encoding must apply merges in the order they were learned; decoding concatenates bytes and must tolerate invalid UTF-8.
- GPT tokenizers pre-split text with a regex so merges do not cross words, numbers and punctuation.
- Special tokens such as `<|endoftext|>` are added outside BPE. Byte-level BPE never needs an unknown-word token.

## Resources

- Andrej Karpathy, "Let's build the GPT Tokenizer", *Neural Networks: Zero to Hero* (video): https://www.youtube.com/watch?v=zduSFxRajkE. The BPE functions here are adapted from this lecture.
- Andrej Karpathy, minbpe (repository and exercise): https://github.com/karpathy/minbpe
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Manning 2024, Chapter 2 (simple tokenizers, special tokens, tiktoken): https://www.manning.com/books/build-a-large-language-model-from-scratch; code: https://github.com/rasbt/LLMs-from-scratch
- Rico Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", 2016: https://arxiv.org/abs/1508.07909
- OpenAI, tiktoken: https://github.com/openai/tiktoken
- Nathan Reed, "A Programmer's Introduction to Unicode" (source of the sample text): https://www.reedbeta.com/blog/programmers-intro-to-unicode/
- Tiktokenizer (interactive token visualizer): https://tiktokenizer.vercel.app
