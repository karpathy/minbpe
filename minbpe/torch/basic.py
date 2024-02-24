"""
Minimal (byte-level) Byte Pair Encoding tokenizer with PyTorch.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

import torch
from torch import Tensor
from .base import merge_torch
from ..basic import BasicTokenizer


class BasicTokenizerTorch(BasicTokenizer):

    def train(self, text: str, vocab_size: int, verbose=False, device='cpu'):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        int_type = torch.int16 if vocab_size <= 2**15 else torch.int32
        ids = torch.tensor(ids, dtype=int_type, device=device)

        for i in range(num_merges):
            # determine the most common pair to merge next
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            pair_index = torch.argmax(counts)
            pair, count = unique[pair_index], counts[pair_index]

            idx = i + 256
            ids = merge_torch(ids, pair, idx)

            pair = tuple(pair.tolist())

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {count} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def encode(self, text: str):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        if len(self.merges) == 0:
            return ids

        int_type = torch.int16 if len(self.merges) <= 2**15 else torch.int32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ids = torch.tensor(ids, dtype=int_type, device=device)

        merges = sorted(list(self.merges), key=lambda p: self.merges[p])
        merges = torch.tensor(merges, dtype=int_type, device=device)

        while len(ids) >= 2:
            # find the pair with the lowest merge index
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique: Tensor = torch.unique(pairs, dim=0)

            is_present = (merges[:, None] == unique[None]).all(-1).any(-1)
            if not is_present.any():
                break # nothing else can be merged anymore

            # otherwise let's merge the best pair (lowest merge index)
            pair_index = is_present.nonzero()[0]
            pair = merges[pair_index]
            idx = pair_index.to(ids.dtype) + 256
            ids = merge_torch(ids, pair, idx)

        return ids.cpu().tolist()
