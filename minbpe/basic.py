"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        raise NotImplementedError()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError()

    def decode(self, ids):
        raise NotImplementedError()

    def encode(self, text):
        raise NotImplementedError()
