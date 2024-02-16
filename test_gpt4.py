"""
Verifies that our implementation agrees with that from tiktoken,
and that we can encode and decode text exactly as GPT-4 would.
"""

import tiktoken
from bpe_regex import Tokenizer as RegexTokenizer

# get the official tokenizer and its merges
enc = tiktoken.get_encoding("cl100k_base")
# mergeable_ranks is the variable thing we need from the official tokenizer
mergeable_ranks = enc._mergeable_ranks

# -----------------------------------------------------------------------------
"""
now comes a bit tricky part.
- the `merges` that tiktoken has above contain first the 255 raw bytes, but
  for some reason these bytes are permuted in a different order. This is
  non-sensical, and I think historical, but for that reason we have to here
  use that custom byte order manually and it looks weird but it's ok.
- second, the `merges` are already the byte sequences in their merged state.
  so we have to recover the original pairings. We can do this by doing
  a small BPE training run on all the tokens, in their order.
also see https://github.com/openai/tiktoken/issues/60
"""

def bpe(mergeable_ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

merges = {}
for token, rank in mergeable_ranks.items():
    if len(token) == 1:
        continue # skip raw bytes
    pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
    assert len(pair) == 2
    # recover the integer ranks of the pair
    ix0 = mergeable_ranks[pair[0]]
    ix1 = mergeable_ranks[pair[1]]
    merges[(ix0, ix1)] = rank

# -----------------------------------------------------------------------------
# now create our own tokenizer. bit hacky
tokenizer = RegexTokenizer()
# override the merges
tokenizer.merges = merges
# and finally keep in mind we have to shuffle the bytes
tokenizer.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
tokenizer.inverse_byte_shuffle = {v: k for k, v in tokenizer.byte_shuffle.items()}
# re-construct the vocab
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
tokenizer.vocab = vocab

# -----------------------------------------------------------------------------
# let's take it for a spin!
text = "hello world!!!? (안녕하세요!) lol123 😉"
print(text)
print(enc.encode(text)) # tiktoken
print(tokenizer.encode(text)) # ours
print(tokenizer.decode(tokenizer.encode(text))) # ours back to text

# two quick tests: equality (to tiktoken) and identity
print("OK" if enc.encode(text) == tokenizer.encode(text) else "FAIL")
print("OK" if text == tokenizer.decode(tokenizer.encode(text)) else "FAIL")

# let's also tokenize all of taylor swift