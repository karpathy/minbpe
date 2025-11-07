"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    raise NotImplementedError()


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    raise NotImplementedError()

# first two helper functions...
def replace_control_characters(s: str) -> str:
    raise NotImplementedError()

def render_token(t: bytes) -> str:
    raise NotImplementedError()

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        raise NotImplementedError()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError()

    def encode(self, text):
        raise NotImplementedError()

    def decode(self, ids):
        raise NotImplementedError()

    def _build_vocab(self):
        raise NotImplementedError()

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        raise NotImplementedError()

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        raise NotImplementedError()
