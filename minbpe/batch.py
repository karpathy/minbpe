"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Optimized version of RegexTokenizer

Unlike RegexTokenizer:
- Non-overlapping merges are made in batches, significantly reducing runtime.
- Code is optimized for speed, not readability (but still pretty readable).
"""

import regex as re
from .base import Tokenizer
from collections import defaultdict
from heapq import nlargest
from functools import lru_cache

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def get_stats(ids):
    """
    Given a list of bytes objects (iterables of ints), return a defaultdict with
    the counts of occurrences of all the consecutive pairs of integers.
    This doesn't make pairs between the last element of one bytes object, and 
    the first element of the next bytes object.
    Example: get_stats([b'abc', b'bcd'])
    -> defaultdict(<class 'int'>, {(97, 98): 1, (98, 99): 2, (99, 100): 1})
    """
    counts = defaultdict(int)
    for chunk_ids in ids:
        last_index = len(chunk_ids) - 1
        i = 0
        while i < last_index:
            j = i + 1
            counts[(chunk_ids[i], chunk_ids[j])] += 1
            i = j
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    i = 0
    last_index = len(ids) - 1
    while i < last_index:
        j = i + 1
        if ids[i] == pair[0] and ids[j] == pair[1]:
            ids[i] = idx
            del ids[j]
            last_index -= 1
        i = j
    return ids

def multi_merge(ids, pairs):
    """
    In the list of lists of integers (ids), replace all consecutive occurrences
    of any pair of ints from an internal list in the pairs dictionary with the
    value of the pair. Don't make merges between internal lists. Changes the ids
    list in place.
    Example: ids=[[1, 2, 4, 5, 4], [5, 3, 1, 2]], pairs={(1, 2): 6, (4, 5): 7}
    -> [[6, 7, 4], [5, 3, 6]]
    """
    for chunk_ids in ids:
        i = 0
        last_index = len(chunk_ids) - 1
        while i < last_index:
            j = i + 1
            token = pairs.get((chunk_ids[i], chunk_ids[j]))
            if token is not None:
                chunk_ids[i] = token
                del chunk_ids[j]
                last_index -= 1
            i = j

@lru_cache(maxsize=8192)
def _memoized_encode_to_list(text):
    return [*text.encode("utf-8")]


class BatchTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        merges_remaining = num_merges
        curr_vocab_size = 256
        max_batch_size = 512 # magic number, can be tuned, must be > 0

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [*map(_memoized_encode_to_list, text_chunks)]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = self.vocab # idx -> bytes
        while merges_remaining > 0:
            seen_first = set() # set of ints that were seen in the first position in pairs
            seen_last = set() # set of ints that were seen in the last position in pairs
            pairs_to_merge = {}
            # count the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pairs with the highest counts
            # use min to make sure you don't consider pairs beyond the vocab size
            top_pairs = nlargest(min(max_batch_size, merges_remaining), stats, key=stats.get)
            for first, last in top_pairs:  # pairs are (first, last) tuples
                if first in seen_last or last in seen_first:
                    # skip this pair because it overlaps with a previous pair
                    # still add it to the seen sets to avoid future overlaps
                    seen_first.add(first)
                    seen_last.add(last)
                    continue # skip this pair but keep looking for mergeable top_pairs
                seen_first.add(first)
                seen_last.add(last)
                pairs_to_merge[(first, last)] = curr_vocab_size
                vocab[curr_vocab_size] = vocab[first] + vocab[last]
                curr_vocab_size += 1
            merges_remaining -= len(pairs_to_merge)
            # replace all occurrences of pair in ids with idx
            multi_merge(ids, pairs_to_merge)
            # save the merges
            merges.update(pairs_to_merge)
            # prints
            if verbose:
                print(f"merges {curr_vocab_size-len(pairs_to_merge)}-{curr_vocab_size}: {pairs_to_merge}")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = [self.vocab[idx] if idx in self.vocab
            else self.inverse_special_tokens[idx].encode("utf-8")
            for idx in ids] # raises KeyError if any idx is not a valid token
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    @lru_cache(maxsize=8192)
    def _encode_chunk(self, ids):
        # return the token ids as bytes obj or list of ints
        # the ids parameter starts as a bytes object, but for our purposes this
        # functions as a list of integers.
        ids = [*ids]
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            low = 987654321
            for i in range(len(ids) - 1):
                current_pair = (ids[i], ids[i+1])
                new_val = self.merges.get(current_pair, 987654321)
                if new_val < low:
                    pair = current_pair
                    low = new_val
            # subtle: if there are no more merges available, the key will be the
            # above .get call's default value for every single pair,
            # meaning none of the pairs were in the merges dictionary
            # we can detect this terminating case by checking if that's the low
            if low == 987654321:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids # bytes obj or list of ints. NB: ids = b'abc'; ly = []; ly.extend(ids) -> [97, 98, 99]

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
        special_pattern = f"({'|'.join([re.escape(k) for k in special])})"
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
