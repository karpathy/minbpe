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

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")

# test encode
for TokenizerClass, name in zip([BasicTokenizer], ["basic"]):

    # construct the Tokenizer object and load pretrained model
    tokenizer = TokenizerClass()
    tokenizer.load(model_file=os.path.join("models", name + ".model"))
    # run original encode on some text
    t0 = time.time()
    ids = tokenizer.encode(text)
    t1 = time.time()

    print(f"\n{name} tokenizer: Original encode took {t1 - t0:.2f} seconds")

    # run new encode on some text
    t0 = time.time()
    my_ids = tokenizer.my_encode(text)
    t1 = time.time()

    print(f"{name} tokennizer: New encode took {t1 - t0:.2f} seconds")
    print(f"original encode output == new encode output: {ids == my_ids}\n")
    text_bytes = text.encode("utf-8") # raw bytes
    raw_ids = list(text_bytes)
    print(f"raw ids:\n{raw_ids[:20]}\n")
    print(f"original encode output:\n{ids[:20]}\n")
    print(f"new encode output:\n{my_ids[:20]}\n")
