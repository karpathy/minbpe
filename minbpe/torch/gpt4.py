"""
Inherits from GPT4Tokenizer and overrides the pre_encode method of
RegexTokenizerTorch to permute the bytes
"""

from .regex import RegexTokenizerTorch
from ..gpt4 import GPT4Tokenizer

class GPT4TokenizerTorch(GPT4Tokenizer, RegexTokenizerTorch):
    def pre_encode(self, text):
        chunks_bytes = super().pre_encode(text)
        chunks_bytes = [bytes(self.byte_shuffle[b] for b in chunk) for chunk in chunks_bytes]
        return chunks_bytes