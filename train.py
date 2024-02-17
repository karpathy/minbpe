"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

# feel free to use either
from bpe_regex import RegexTokenizer
from bpe_basic import BasicTokenizer

# open some text and train a vocab of 512 tokens
text = open("taylorswift.txt", "r", encoding="utf-8").read()

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)

    # pretty print the final vocab into a file
    vocab_file = f"{name}.vocab"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for idx, token in tokenizer.vocab.items():
            if idx < 256:
                # the first 256 tokens are just bytes, render them in <0xHH> format
                token_string = f"<0x{idx:02x}>"
            else:
                # otherwise let's attempt to render the token as a string
                token_string = token.decode('utf-8', errors='replace')
            f.write(f"{token_string} {idx}\n")
