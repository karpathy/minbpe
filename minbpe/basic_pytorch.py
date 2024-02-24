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
from .base import Tokenizer

def merge(ids: Tensor, pair: Tensor, idx: int):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    # create a mask for the first element of every matching pair
    pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
    is_first_in_pair = (pairs == pair).all(axis=1)
    false_tensor = torch.tensor([False], dtype=torch.bool, device=ids.device)
    is_first_in_pair = torch.cat((is_first_in_pair, false_tensor))
    # create a mask for the second element of every matching pair
    is_second_in_pair = is_first_in_pair.roll(1)
    # each token can only belong to one pair
    is_first_in_pair &= ~is_second_in_pair
    is_second_in_pair = is_first_in_pair.roll(1)
    # change the first element of every matching pair to the new token
    ids[is_first_in_pair] = idx
    # remove the second element of every matching pair
    ids = ids[~is_second_in_pair]
    return ids

class BasicPyTorchTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose=False, device='cuda'):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        int_type = torch.int16 if vocab_size <= 2**15 else torch.int32
        ids = torch.tensor(ids, dtype=int_type, device=device)
        merges = torch.zeros((num_merges, 2), dtype=int_type, device=device)

        for i in range(num_merges):
            # determine the most common pair to merge next
            pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
            unique, counts = torch.unique(pairs, return_counts=True, dim=0)
            pair_index = torch.argmax(counts)
            pair, count = unique[pair_index], counts[pair_index]

            ids = merge(ids, pair, i + 256)
            merges[i] = pair

            if verbose:
                print(f"merge {i+1}/{num_merges}: {tuple(pair.tolist())} -> {i + 256} had {count} occurrences")

        merges = merges.cpu().numpy()
        merges = [tuple(pair) for pair in merges]

        self.merges = {pair: i + 256 for i, pair in enumerate(merges)}

        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            idx = i + 256
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

    def encode(self, text: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        int_type = torch.int16 if len(self.merges) <= 2**15 else torch.int32
        ids = torch.tensor(ids, dtype=int_type, device=device)

        merges = list(self.merges.keys())
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
            ids = merge(ids, pair, idx)

        return ids.cpu().tolist()
