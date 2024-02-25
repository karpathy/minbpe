"""
Overrides the encode_ordinary method of RegexTokenizer with a PyTorch implementation.
TODO: override train method with a PyTorch implementation.
"""

import regex as re
import torch
from torch import Tensor
from .base import merge_torch
from ..regex import RegexTokenizer


class RegexTokenizerTorch(RegexTokenizer):

    def train(self, text, vocab_size, verbose=False):
        # TODO
        super().train(text, vocab_size, verbose)

    def pre_encode(self, text):
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        chunks = [chunk.encode("utf-8") for chunk in text_chunks]
        return chunks

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        chunks = self.pre_encode(text)
        # all chunks of text are encoded separately, then results are joined
        int_type = torch.int16 if len(self.merges) <= 2**15 else torch.int32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ids = [list(chunk_bytes) for chunk_bytes in chunks]
        if len(self.merges) == 0:
            return sum(ids, [])
        merges = sorted(list(self.merges), key=lambda p: self.merges[p])
        merges = torch.tensor(merges, dtype=int_type, device=device)

        for i, chunk_ids in enumerate(ids):
            chunk_ids = torch.tensor(chunk_ids, dtype=int_type, device=device)
            while len(chunk_ids) >= 2:
                # find the pair with the lowest merge index
                pairs = torch.stack((chunk_ids[:-1], chunk_ids[1:]), dim=1)
                unique: Tensor = torch.unique(pairs, dim=0)

                is_present = (merges[:, None] == unique[None]).all(-1).any(-1)
                if not is_present.any():
                    break # nothing else can be merged anymore

                # otherwise let's merge the best pair (lowest merge index)
                pair_index = is_present.nonzero()[0]
                pair = merges[pair_index]
                idx = pair_index.to(chunk_ids.dtype) + 256
                chunk_ids = merge_torch(chunk_ids, pair, idx)

            ids[i] = chunk_ids.cpu().tolist()
        return sum(ids, [])
