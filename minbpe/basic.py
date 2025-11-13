"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge

MIN_VOCAB_SIZE = 256


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _simple_num_merges(vocab_size: int) -> int:
        return vocab_size - MIN_VOCAB_SIZE

    def train(
        self, text: str, vocab_size: int, num_merges: int = None, verbose: bool = False
    ) -> None:
        assert vocab_size >= MIN_VOCAB_SIZE, "vocab_size must be greater than 256"
        if num_merges is None:
            num_merges = self._simple_num_merges(vocab_size)
        assert num_merges > 0, "num_merges must be greater than 0"

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        # each train is a fresh start
        token_ids = list(text_bytes)  # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        # int to bytes mapping
        vocab = {idx: bytes([idx]) for idx in range(MIN_VOCAB_SIZE)}  # int -> bytes

        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(token_ids)
            # find the pair with the highest count
            pair = stats.most_common(1)[0][0]
            # mint a new token: assign it the next available id
            new_token = MIN_VOCAB_SIZE + i
            # replace all occurrences of pair in ids with idx
            token_ids = merge(token_ids, pair, new_token)
            # save the merge
            merges[pair] = new_token
            vocab[new_token] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {new_token} ({vocab[new_token]}) had {stats[pair]} occurrences"
                )

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def decode(self, ids: list[int]) -> str:
        """given ids (list of integers), return Python string"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> list[int]:
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        # this prevents merges less than 2 tokens
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
