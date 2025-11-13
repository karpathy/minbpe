"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""
from __future__ import annotations

from collections import Counter
from enum import StrEnum, auto

import regex
from minbpe.base import MIN_VOCAB_SIZE, Tokenizer, get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class SpecialTokenMode(StrEnum):
    ALL = auto()
    NONE = auto()
    NONE_RAISE = auto()

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = pattern or GPT4_SPLIT_PATTERN
        self.compiled_pattern = regex.compile(self.pattern)
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    def train(
        self, text: str, vocab_size: int, num_merges: int = None, verbose: bool = False
    ) -> None:
        assert vocab_size >= MIN_VOCAB_SIZE, "vocab_size can't be less than 256"
        if num_merges is None:
            num_merges = self._simple_num_merges(vocab_size)
        assert num_merges > 0, "num_merges can't be less than 0"

        # split the text up into text chunks
        text_chunks = regex.findall(self.compiled_pattern, text)
        
         # input text preprocessing
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(MIN_VOCAB_SIZE)}
        for i in range(num_merges):
            stats = Counter()
            for chunk_ids in ids:
                # passing stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            most_common_element = stats.most_common(1)[0]
            most_common_pair = most_common_element[0]
            occurrences = most_common_element[1]
            # Create a new token: assign it to next aviliable id
            new_token = MIN_VOCAB_SIZE + i
            # replace all occurance of pair in ids with idx
            ids = [merge(chunk_ids, most_common_pair, new_token) for chunk_ids in ids]
            # save the merge
            merges[most_common_pair] = new_token
            vocab[new_token] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {most_common_pair} -> {new_token} ({vocab[new_token]}) had {occurrences} occurrences")
            
            self.merges = merges
            self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}


    # list[bytes] -> token_ids(list[int]) -> text(string)    
    def decode(self, ids: list[int]) -> str:
        """ Given ids (list of integers), return Python string"""
        part_bytes = []
        for idx in ids:
            if idx in ids:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid tokens {idx = }")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # text(string) -> token_ids(list[int]) -> list[bytes]
    # ab -> [97, 98] -> b"ab"
    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        # need to merge all mergable sequence
        while len(ids) > 2:
            # find the pair with lowest index because this index can be reuse by latter index
            stats = get_stats(ids)
            lowest_index_pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if lowest_index_pair not in self.merges:
                # Nothing else can be merge
                break
            # otherwise merge the pair
            idx = self.merges[lowest_index_pair]
            ids = merge(ids, lowest_index_pair, idx)
        return ids
        

    def encode_ordinary(self, text: str) -> list[int]:
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = regex.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
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
        special_chunks = regex.split(special_pattern, text)
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