"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    # construct the Tokenizer object and kick off verbose training
    t0 = time.perf_counter()
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=False)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
    t1 = time.perf_counter()
    print(f"Training {name} tokenizer took {t1 - t0:.2f} seconds")
    # > Training basic tokenizer took 3.46 seconds
    # > Training regex tokenizer took 8.84 seconds
