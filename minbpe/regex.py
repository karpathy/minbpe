"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

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
    def __init__(self, pattern: str | None = None) -> None:
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = regex.compile(self.pattern)
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= MIN_VOCAB_SIZE
        num_merges = vocab_size - MIN_VOCAB_SIZE

        # split the text up into text chunks
        text_chunks = regex.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(MIN_VOCAB_SIZE)}  # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = Counter()
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            most_common_element = stats.most_common(1)
            most_common_pair = most_common_element[0][0]
            occurance = most_common_element[0][1]
            # pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = MIN_VOCAB_SIZE + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, most_common_pair, idx) for chunk_ids in ids]
            # save the merge
            merges[most_common_pair] = idx
            vocab[idx] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
            # prints
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {most_common_pair} -> {idx} ({vocab[idx]}) had {occurance} occurrences"
                )

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids: list[int]) -> str:
        # given ids (list of integers), return Python string
        part_bytes: list[bytes] = []
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

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
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
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = regex.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids: list[int] = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")  # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(
        self, text: str, allowed_special: str | set[str] = "none_raise"
    ) -> list[int]:
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        # support both SpecialTokenMode enum and legacy string inputs
        mode = None
        if isinstance(allowed_special, SpecialTokenMode):
            mode = allowed_special
        elif isinstance(allowed_special, str):
            try:
                mode = SpecialTokenMode(allowed_special)
            except ValueError:
                mode = None

        if mode == SpecialTokenMode.ALL:
            special = self.special_tokens
        elif mode == SpecialTokenMode.NONE:
            special = {}
        elif mode == SpecialTokenMode.NONE_RAISE:
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
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
        special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
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
