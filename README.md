# minbpe

Minimal, clean code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This algorithm was popularized for LLMs by the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the associated GPT-2 [code release](https://github.com/openai/gpt-2) from OpenAI. [Sennrich et al. 2015](https://arxiv.org/abs/1508.07909) is cited as the original reference for the use of BPE in NLP applications. Today, all modern LLMs (e.g. GPT, Llama, Mistral) use this algorithm to train their tokenizers.

There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The files of the repo are as follows:

1. [minbpe/base.py](minbpe/base.py): Implements the `Tokenizer` class, which is the base class. It contains the `train`, `encode`, and `decode` stubs, save/load functionality, and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.
2. [minbpe/basic.py](minbpe/basic.py): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on text.
3. [minbpe/regex.py](minbpe/regex.py): Implements the `RegexTokenizer` that further splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (think: letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This was introduced in the GPT-2 paper and continues to be in use as of GPT-4. This class also handles special tokens, if any.
4. [minbpe/gpt4.py](minbpe/gpt4.py): Implements the `GPT4Tokenizer`. This class is a light wrapper around the `RegexTokenizer` (2, above) that exactly reproduces the tokenization of GPT-4 in the [tiktoken](https://github.com/openai/tiktoken) library. The wrapping handles some details around recovering the exact merges in the tokenizer, and the handling of some unfortunate (and likely historical?) 1-byte token permutations.

Finally, the script [train.py](train.py) trains the two major tokenizers on the input text [tests/taylorswift.txt](tests/taylorswift.txt) (this is the Wikipedia entry for her kek) and saves the vocab to disk for visualization. This script runs in about 25 seconds on my (M1) MacBook.

## usage

All of the files above are very short and thoroughly commented, and also contain a usage example on the bottom of the file. As a quick example, following along the [Wikipedia article on BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding), we can reproduce it as follows:

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# writes two files: toy.model (for loading) and toy.vocab (for viewing)
```

The result above is exactly as expected, please see bottom of [minbpe/basic.py](minbpe/basic.py) for more details. To use the `GPT4Tokenizer`, simple example and comparison to [tiktoken](https://github.com/openai/tiktoken):

```python
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# ours
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

(you'll have to `pip install tiktoken` to run).

## tests

We use the pytest library for tests. All of them are located in the `tests/` directory. First `pip install pytest` if you haven't already, then:

```bash
$ pytest .
```

to run the tests.

## todos

- write a more optimized Python version that could run over large files and big vocabs
- write an even more optimized C or Rust version (think through)
- rename GPT4Tokenizer to GPTTokenizer and support GPT-2 as well?
- write a LlamaTokenizer similar to GPT4Tokenizer (i.e. attempt sentencepiece equivalent)
- video coming soon ;)

## License

MIT
