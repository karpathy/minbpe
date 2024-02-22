"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

import torch
from torch import nn
from .base import Tokenizer, get_stats, merge


class OptimizedTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        ids = torch.tensor(ids, dtype=torch.int64, device="cpu")
        merge_pairs = torch.zeros((num_merges, 2), dtype=torch.int64)

        for i in range(num_merges):
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            pair_index = torch.argmax(counts)
            pair = unique[pair_index]
            count = counts[pair_index]

            mask = torch.all(pairs == pair, dim=1)
            mask = torch.cat((mask, torch.tensor([False])))
            ids[mask] = i + 256
            ids = ids[~torch.roll(mask, 1, 0)]

            merge_pairs[i] = pair
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {i + 256}")

        self.merges = {
            tuple(pair.tolist()): j + 256
            for j, pair in enumerate(merge_pairs)
        }

        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            pair = merge_pairs[i]
            vocab[i + 256] = vocab[pair[0].item()] + vocab[pair[1].item()]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {i + 256} ({vocab[i + 256]}) had {count} occurrences")
        self.vocab = vocab


    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids