# LLM Tokenization

Hi everyone, today we are going to look at Tokenization in Large Language Models (LLMs). Sadly, tokenization is a relatively complex and gnarly component of the state of the art LLMs, but it is necessary to understand in some detail because a lot of the shortcomings of LLMs that may appear mysterious otherwise actually trace back to tokenization.

### previously: character-level tokenization

So what is tokenization? Well it turns out that in our previous video, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY), we already did tokenization but it was only a very simple, naive, character-level version of it. When you go to the [Google colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) for that video, you'll see that we started with our training data ([Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)), which is just a large string in Python:

```
First Citizen: Before we proceed any further, hear me speak.

All: Speak, speak.

First Citizen: You are all resolved rather to die than to famish?

All: Resolved. resolved.

First Citizen: First, you know Caius Marcius is chief enemy to the people.

All: We know't, we know't.
```

But how do we feed strings into a language model? Well we saw that we did this by first constructing a vocabulary of all the possible characters we found in the entire training set:

```python
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# 65
```

And then creating a lookup table for converting between individual string characters and integers in the vocabulary above. This lookup table was just a Python dictionary:

```python
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

# [46, 47, 47, 1, 58, 46, 43, 56, 43]
# hii there
```

Once we have a sequence of integers, we saw that each integer was used to index into a 2-dimensional embedding of trainable parameters. Because we have a vocabulary size of `vocab_size=65` in this example, this embedding table will also have 65 rows:

```python
class BigramLanguageModel(nn.Module):

def __init__(self, vocab_size):
	super().__init__()
	self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

def forward(self, idx, targets=None):
	tok_emb = self.token_embedding_table(idx) # (B,T,C)
```

Here, the integer "plucks out" a row of this embedding table and this row is the vector that represents this token. This vector then feeds into the Transformer as the input at the corresponding time step.

This is all well and good for the naive setting of a character-level language model. But in practice, in state of the art language models, people use a lot more complicated schemes for constructing these token vocabularies. So we're dealing not on a character level, but on chunk level. And the way these vocabularies of chunks are constructed is by using algorithms such as the **Byte Pair Encoding** (BPE) algorithm, which we are going to now cover in detail.

The paper that popularized the use of the byte-level BPE algorithm for language model tokenization is the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) from OpenAI in 2019, "Language Models are Unsupervised Multitask Learners". If you scroll to Section 2.2 on Input Representation where they describe and motivate this algorithm, you'll see them say:

> The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.

So recall that in the attention layer every token is attending to a finite list of tokens previously in the sequence, and the paper here says that the GPT-2 model has a context length of 1024 tokens, up from 512 in GPT-1. In other words, tokens are the fundamentally "atoms" at the input to the LLM. And tokenization is the process for taking raw strings in Python and converting them to a list of tokens, and vice versa. If you go to the [Llama 2](https://arxiv.org/abs/2307.09288) paper as well and you search for "token", you're going to get 63 hits. So for example, the paper claims that they trained on 2 trillion tokens, etc.

Let us now turn to the implementation of the Tokenizer.

(TODO: may continue this unless we figure out how to generate it automatically from the video :))
