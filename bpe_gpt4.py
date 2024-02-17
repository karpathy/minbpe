"""
Implements the GPT-4 Tokenizer with a light wrapper around the RegexTokenizer.
"""

import tiktoken
from bpe_regex import RegexTokenizer
import os
import json
from transformers.utils import PushToHubMixin, cached_file

def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

class GPT4Tokenizer(RegexTokenizer, PushToHubMixin):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self, vocab_file=None, merges=None):
        super().__init__()
        if merges is None:
            # get the official tokenizer and its merges
            enc = tiktoken.get_encoding("cl100k_base")
            self.mergeable_ranks = enc._mergeable_ranks
            # the merges are those of gpt4, but we have to recover them
        else:
            with open(vocab_file,  encoding="utf-8") as vocab_handle:
                self.mergeable_ranks = json.load(vocab_handle)

        self.merges = recover_merges(self.mergeable_ranks)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        # now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.
        self.byte_shuffle = {i: self.mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

    def _encode_chunk(self, text_bytes):
        # before we start processing bytes, we have to permute them
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        # we have to un-permute the bytes before we decode
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, "vocab.txt")
        # we usually save the merges in a string format with token.decode(erros="replace") but let's  be simple
        with open(vocab_file, "w",encoding="utf-8") as writer:
            for token, _ in sorted(self.mergeable_ranks.items(), key=lambda kv: kv[1]):
                writer.write(token.decode("utf-8", errors="replace")+"\n")
        return vocab_file
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name=None, **kwargs):
        resolved_vocab_files = {"vocab_file":"vocab.txt",}
        for file_id, file_path in resolved_vocab_files.items():
            resolved_vocab_files[file_id] = cached_file(pretrained_model_name, file_path, **kwargs)

        # Instantiate the tokenizer.
        try:
            tokenizer = GPT4Tokenizer(**resolved_vocab_files)
        except OSError:
            raise OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )
        return tokenizer
        

if __name__ == "__main__":
    # let's take it for a spin!

    # tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    # vs.
    tokenizer = GPT4Tokenizer()
    # fight!
    
    tokenizer.push_to_hub("ArthurZ/gpt-min", private=True)
    tokenizer = GPT4Tokenizer.from_pretrained("ArthurZ/gpt-min")

    text = "hello world!!!? (안녕하세요!) lol123 😉"
    print(text)
    print(enc.encode(text)) # tiktoken
    print(tokenizer.encode(text)) # ours
    print(tokenizer.decode(tokenizer.encode(text))) # ours back to text

    # two quick tests: equality (to tiktoken) and identity
    print("OK" if enc.encode(text) == tokenizer.encode(text) else "FAIL")
    print("OK" if text == tokenizer.decode(tokenizer.encode(text)) else "FAIL")

    # let's also tokenize all of taylor swift, a bigger document just to make sure
    text = open("taylorswift.txt", "r", encoding="utf-8").read()
    t1 = enc.encode(text) # tiktoken
    t2 = tokenizer.encode(text) # ours
    print("OK" if t1 == t2 else "FAIL")
    print("OK" if text == tokenizer.decode(tokenizer.encode(text)) else "FAIL")
