"""
Overrides the train and encode methods of BasicTokenizer with PyTorch implementations.
"""

import torch
from torch import Tensor
from .basic import BasicTokenizer

def merge_torch(ids: Tensor, pair: Tensor, idx: int):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """

    # create a mask for the first element i of every matching pair (i, j)
    pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
    is_pair = (pairs == pair).all(axis=1)
    false_tensor = torch.tensor([False], dtype=torch.bool, device=ids.device)
    is_pair_i = torch.cat((is_pair, false_tensor))

    # create a mask for the second element j of every matching pair (i, j)
    is_pair_j = is_pair_i.roll(1)

    # handle overlapping pairs for repeated tokens
    while True:
        is_overlap = (is_pair_i & is_pair_j).any()
        if not is_overlap:
            break # no overlapping pairs

        # remove first overlapping pairs in repeated sequences
        is_first = (is_pair_i & is_pair_j).int().diff() == 1
        is_first = torch.cat((false_tensor, is_first))
        is_pair_i &= ~is_first
        is_pair_j = is_pair_i.roll(1)

    # change the first element i of every matching pair (i, j) to the new token
    ids[is_pair_i] = idx

    # remove the second element j of every matching pair (i, j)
    ids = ids[~is_pair_j]
    return ids


class BasicTorchTokenizer(BasicTokenizer):

    def train(self, text: str, vocab_size: int, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        int_type = torch.int16 if vocab_size <= 2**15 else torch.int32
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
