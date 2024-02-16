# minbpe

Minimal, clean, educational code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This algorithm was popularized for LLMs by the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the associated GPT-2 [code release](https://github.com/openai/gpt-2) from OpenAI. [Sennrich et al. 2015](https://arxiv.org/abs/1508.07909) is cited as the original reference for the use of BPE in NLP applications. Today, all modern LLMs (e.g. GPT, Llama, Mistral) use this algorithm to train their tokenizers.

There are two primary Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The two tokenizers are:

1. [bpe_basic.py](bpe_basic.py): Implements the `BasicTokenizer`, the simplest implementation of the BPE algorithm that runs directly on text.
2. [bpe_regex.py](bpe_regex.py): Implements the `RegexTokenizer` that further splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (think: letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This was introduced in the GPT-2 paper and continues to be in use as of GPT-4.
3. [bpe_gpt4.py](bpe_gpt4.py): Implements the `GPT4Tokenizer`. This class is a light wrapper around the `RegexTokenizer` (2, above) that exactly reproduces the tokenization of GPT-4 in the [tiktoken](https://github.com/openai/tiktoken) library. The wrapping handles some details around recovering the exact merges in the tokenizer, and the handling of some unfortunate and likely historical 1-byte token permutations. Note that the parity is not fully complete yet because we do not handle special tokens.

Finally, the script [train.py](train.py) trains the two major tokenizers on the input text [taylorswift.txt](taylorswift.txt) (this is the Wikipedia entry for her kek) and saves the vocab to disk for visualization. This script runs in about 25 seconds on my (M1) MacBook.

## todos

- handle special tokens (?)
- save and load Tokenizers to/from disk
- video coming soon ;)

## License

MIT
