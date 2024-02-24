"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

import torch
from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def train_pytorch(self, text: str, vocab_size: int, verbose=False, device='cpu'):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        int_type = torch.int16 if vocab_size <= 2**15 else torch.int32
        ids = torch.tensor(ids, dtype=int_type, device=device)
        merges = torch.zeros((num_merges, 2), dtype=int_type, device=device)
        false_tensor = torch.tensor([False], dtype=torch.bool, device=device)

        for i in range(num_merges):
            # determine the most common pair to merge next
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            pair_index = torch.argmax(counts)
            pair, count = unique[pair_index], counts[pair_index]
            merges[i] = pair

            # merge the pair
            # create a mask for the first element of every matching pair
            is_first_in_pair = torch.all(pairs == pair, axis=1)
            is_first_in_pair = torch.cat((is_first_in_pair, false_tensor))
            # create a mask for the second element of every matching pair
            is_second_in_pair = torch.roll(is_first_in_pair, 1, 0)
            # each token can only belong to one pair
            is_first_in_pair &= ~is_second_in_pair
            is_second_in_pair = torch.roll(is_first_in_pair, 1, 0)
            # change the first element of every occurrence of the pair to the new id
            ids[is_first_in_pair] = i + 256
            # remove the second element of every occurrence of the pair
            ids = ids[~is_second_in_pair]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {tuple(pair.tolist())} -> {i + 256} had {count} occurrences")

        merges = merges.cpu().numpy()
        merges = [tuple(pair) for pair in merges]

        self.merges = {pair: j + 256 for j, pair in enumerate(merges)}

        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            idx = 256 + i
            pair = merges[i]
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]})")
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
