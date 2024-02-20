"""
Implements the LlamaTokenizer as a lightweight wrapper around SentencePiece for tokenization tasks.
Note that this is a pretrained tokenizer. By default and inside __init__(), it loads the pretrained tokenizer from a specified source.
"""
import sentencepiece as spm
from .regex import RegexTokenizer

class LlamaTokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that attempts to replicate the behavior of SentencePiece."""

    def __init__(self, model_file):
        super().__init__()
        self.sp = spm.SentencePieceProcessor(model_file)

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def save(self, file_prefix):
        raise NotImplementedError("LlamaTokenizer does not support saving.")

    def load(self, model_file):
        raise NotImplementedError("LlamaTokenizer does not support loading.")
