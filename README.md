# minbpe

Minimal, clean, educational code for the (byte-level) Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings.

This algorithm was popularized for LLMs by the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and the associated GPT-2 [code release](https://github.com/openai/gpt-2) from OpenAI. Today, all modern LLMs (e.g. GPT, Llama, Mistral) use this algorithm to train their tokenizers.

There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The two tokenizers are:

1. [bpe_basic.py](bpe_basic.py): The simplest implementation of the BPE algorithm that runs directly on text.
2. [bpe_regex.py](bpe_regex.py): This implementation further splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (think: letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This was introduced in the GPT-2 paper and continues to be in use as of GPT-4.

Finally, the script [train.py](train.py) trains both of these tokenizers on the input text [taylorswift.txt](taylorswift.txt) (this is the Wikipedia entry for her kek) and saves the vocab to disk for visualization. This script runs in about 25 seconds on my (M1) MacBook.

# License

MIT
