"""
Minimal (byte-level) Byte Pair Encoding tokenizer with regex splitting and parallel processing.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Parallelized version of RegexTokenizer:
- Parallel counting with multi CPU cores during training
- Maintains regex splitting and special token handling
- Does not parallel merging to avoid cross-chunk issues
"""

import regex as re
import multiprocessing
from collections import Counter
from .base import Tokenizer, get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# Helper function for parallel processing
def chunk_stats_worker(chunk_ids):
    """Worker function to calculate stats for a chunk of text."""
    stats = {}
    get_stats(chunk_ids, stats)
    return stats


class ParallelRegexTokenizer(Tokenizer):

    def __init__(self, pattern=None, num_workers=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - num_workers: number of worker processes (default: CPU count)
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        
        # Set number of workers
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks using regex
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        
        for i in range(num_merges):
            # Parallel counting of byte pair frequencies
            if self.num_workers > 1 and len(ids) > 100:  # Only parallelize for larger datasets
                with multiprocessing.Pool(self.num_workers) as pool:
                    stats_list = pool.map(chunk_stats_worker, ids)
                
                # Combine stats from all chunks
                stats = Counter()
                for chunk_stats in stats_list:
                    stats.update(chunk_stats)
            else:
                # Sequential counting for small datasets
                stats = {}
                for chunk_ids in ids:
                    get_stats(chunk_ids, stats)
            
            if not stats:
                break  # No more pairs to merge
                
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            
            # mint a new token: assign it the next available id
            idx = 256 + i
            
            # replace all occurrences of pair in all chunks with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    # The following methods are identical to RegexTokenizer
    # as they don't benefit from parallelization or require
    # special handling for parallel execution
    
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
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

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
