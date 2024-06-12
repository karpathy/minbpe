"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, BatchTokenizer, RegexTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

# save trained Tokenizers for optional inspection later
tokenizers = {}
timings = {}

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer, BatchTokenizer], ["basic", "regex", "batch"]):
    t0 = time.time()
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    t1 = time.time()

    # writes three files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
    tokenizers[name] = tokenizer

    # demonstrate that encoding a text and decoding is equal to the original (also time it)
    t2 = time.time()
    test = tokenizer.encode(text)
    t3 = time.time()
    res = tokenizer.decode(test)
    t4 = time.time()
    assert(text == res)

    # timings
    timings[name] = [t1-t0, t3-t2, t4-t3]

for name, times in timings.items():
    print('\n*****************************')
    print(f"Training {name} tokenizer took:   {times[0]:.2f} seconds")
    print(f"Encoding took:                   {times[1]:.4f} seconds")
    print(f"Decoding took:                   {times[2]:.4f} seconds")

# uncomment the next line to enter interpreter mode with all the above variables in scope
# import code; code.interact(local=locals())