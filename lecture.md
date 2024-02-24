# LLM Tokenization

Hi everyone, today we are going to look at Tokenization in Large Language Models (LLMs). Sadly, tokenization is a relatively complex and gnarly component of the state of the art LLMs, but it is necessary to understand in some detail because a lot of the shortcomings of LLMs that may be attributed to the neural network or otherwise appear mysterious actually trace back to tokenization.

### Previously: character-level tokenization

So what is tokenization? Well it turns out that in our previous video, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY), we already covered tokenization but it was only a very simple, naive, character-level version of it. When you go to the [Google colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) for that video, you'll see that we started with our training data ([Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)), which is just a large string in Python:

```
First Citizen: Before we proceed any further, hear me speak.

All: Speak, speak.

First Citizen: You are all resolved rather to die than to famish?

All: Resolved. resolved.

First Citizen: First, you know Caius Marcius is chief enemy to the people.

All: We know't, we know't.
```

But how do we feed strings into a language model? Well, we saw that we did this by first constructing a vocabulary of all the possible characters we found in the entire training set:

```python
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# 65
```

And then creating a lookup table for converting between individual characters and integers according to the vocabulary above. This lookup table was just a Python dictionary:

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

Once we've converted a string into a sequence of integers, we saw that each integer was used as an index into a 2-dimensional embedding of trainable parameters. Because we have a vocabulary size of `vocab_size=65`, this embedding table will also have 65 rows:

```python
class BigramLanguageModel(nn.Module):

def __init__(self, vocab_size):
	super().__init__()
	self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

def forward(self, idx, targets=None):
	tok_emb = self.token_embedding_table(idx) # (B,T,C)
```

Here, the integer "plucks out" a row of this embedding table and this row is the vector that represents this token. This vector then feeds into the Transformer as the input at the corresponding time step.

### "Character chunks" for tokenization using the BPE algorithm

This is all well and good for the naive setting of a character-level language model. But in practice, in state of the art language models, people use a lot more complicated schemes for constructing these token vocabularies. In particular, these schemes work not on a character level, but on character chunk level. And the way these chunk vocabularies are constructed is by using algorithms such as the **Byte Pair Encoding** (BPE) algorithm, which we are going to cover in detail below.

Turning to the historical development of this approach for a moment, the paper that popularized the use of the byte-level BPE algorithm for language model tokenization is the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) from OpenAI in 2019, "Language Models are Unsupervised Multitask Learners". Scroll down to Section 2.2 on "Input Representation" where they describe and motivate this algorithm. At the end of this section you'll see them say:

> *The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.*

Recall that in the attention layer of a Transformer, every token is attending to a finite list of tokens previously in the sequence. The paper here says that the GPT-2 model has a context length of 1024 tokens, up from 512 in GPT-1. In other words, tokens are the fundamental "atoms" at the input to the LLM. And tokenization is the process for taking raw strings in Python and converting them to a list of tokens, and vice versa. As another popular example to demonstrate the pervasiveness of this abstraction, if you go to the [Llama 2](https://arxiv.org/abs/2307.09288) paper as well and you search for "token", you're going to get 63 hits. So for example, the paper claims that they trained on 2 trillion tokens, etc.

### Brief taste of the complexities of tokenization

Before we dive into details of the implementation, let's briefly motivate the need to understand the tokenization process in some detail. Tokenization is at the heart of a lot of weirdness in LLMs and I would advise that you do not brush it off. A lot of the issues that may look like issues with the neural network architecture actually trace back to tokenization. Here are just a few examples:

- Why can't LLM spell words? **Tokenization**.
- Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
- Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
- Why is LLM bad at simple arithmetic? **Tokenization**.
- Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
- Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
- What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
- Why did the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
- Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
- Why is LLM not actually end-to-end language modeling? **Tokenization**.
- What is the real root of suffering? **Tokenization**.

We will loop back around to these at the end of the video.

### Visual preview of tokenization

Next, let's load this [tokenization webapp](https://tiktokenizer.vercel.app). What is nice about this webapp is that tokenization is running live in your web browser, allowing you to easily input some text string at the input, and see the tokenization on the right. On the top, you can see that we are currently using the `gpt2` tokenizer, and we see that the string that we pasted in with this example is currently tokenizing into 300 tokens. Here they are shown explicitly in colors:

![tiktokenizer](assets/tiktokenizer.png)

So for example, the string "Tokenization" encoded into the tokens 30642 followed by the token 1634. The token " is" (note that these is three characters, including the space in the front, this is important!) is index 318. Be careful with whitespace because it is absolutely present in the string and must be tokenized along with all the other characters, but is usually omitted in visualization for clarity. You can toggle on and off its visualization at the bottom of the app. In the same way, the token " at" is 379, " the" is 262, etc.

Next, we have a simple example of some arithmetic. Here, we see that numbers may be inconsistently decomposed by the tokenizer. For example, the number 127 is a single token of three characters, but the number 677 because two tokens: the token " 6" (again, note the space in the front!) and the token "77". We rely on the large language model to make sense of this arbitrariness. It has to learn inside its parameters and during training that these two tokens (" 6" and "77" actually combine to create the number 677). In the same way, we see that if the LLM wanted to predict that the result of this sum is the number 804, it would have to output that in two time steps: first it has to emit the token " 8", and then the token "04". Note that all of these splits look completely arbitrary. In the example right below, we see that 1275 is "12" followed by "75", 6773 is actually three tokens " 6", "77", "3", and 8041 is " 8", "041".

### BPE algorithm for tokenization

The BPE algorithm allows us to compress byte sequences from the UTF-8 encoding to a variable size vocabulary. The key idea is that we iteratively find the most frequent pair of bytes (byte pair) in the input text, create a new token to represent that byte pair, append that token to our vocabulary, and replace all occurrences of that byte pair with the new token. 

We repeat this process to compress the text more and more. For example, let's say we start with a vocabulary of size 4 (A, B, C, D) and a sequence of length 11. We may identify that "AA" is the most common byte pair. We then:

1. Add a new token "Z" to represent "AA" 
2. Replace all occurrences of "AA" with "Z", compressing the length to 9
3. Identify next most common byte pair, say "AB", add token "Y"
4. Replace all "AB" with "Y" etc...

After several rounds, we end up with a much shorter sequence but a larger vocabulary that now contains tokens like "Z", "Y", etc. that each represent common byte pairs.

The same process allows us to compress UTF-8 byte streams of text to shorter sequences of tokens. We start with 256 possible byte values, but mint new tokens corresponding to common byte pairs, triples, etc. This gives us a trainable compression algorithm.

### Potential for tokenization-free modeling

Ideally, we could feed raw UTF-8 byte streams directly into language models. However, this causes issues:

- Vocabulary size of 256 is far too small 
- Text sequences become extremely long
- Attention becomes expensive over such long sequences
- We lose ability to attend to sufficient context

There is interesting [recent work](https://arxiv.org/abs/2207.03602) on modifications to Transformer architecture that could allow tokenization-free modeling at scale, but this is still at early stage. 

So for now, we rely on BPE to compress UTF-8 bytes into a trainable variable-size vocabulary. Let's implement this!

### Implementing BPE iterations 

We now have the key functions to find the most frequent byte pair (`get_stats`) and merge that byte pair into a new token (`merge`). We can iterate these steps to create our full tokenizer vocabulary. 

As a reminder, the number of iterations is a hyperparameter we can tune. For example, GPT-4 currently uses ~100k tokens. 

I went back to the blog post and copied a longer piece of text to have more data to train on. Now we can implement the BPE iterations:

```python
merges = {} # map (byte, byte) -> new token id
vocab = {n: bytes([n]) for n in range(256)} # map new token id -> byte sequence

for i in range(300): # do 300 merge iterations
    
    # find most frequent pair of bytes
    stats = get_stats(tokens) 
    pair = max(stats, key=stats.get)
    
    # merge pair into new token, update vocab
    idx = 256 + i
    tokens = merge(tokens, pair, idx)
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
```

Walk through this:
- We start with vocab size 256 (raw bytes) 
- We will do 300 merge iterations, creating tokens 256 --> 255+300
- `get_stats` finds most frequent pair
- `merge` replaces pair with next available token index
- We update `merges` and `vocab` after each iteration

After the iterations, we have our full tokenizer consisting of:
- `merges`: map from byte pair to the token id it was merged into
- `vocab`: map from token id to actual byte sequence 

These two mappings contain everything needed to encode and decode between raw bytes and our learned token ids!

Now we can try encoding the text we trained on to observe the compression:

```
print(f"vocab size: {len(vocab)}") 
print(f"text length before: {len(text_bytes)}, after: {len(tokens)}")
```

Which prints out that we do indeed now have a vocabulary size of 256+300=556 tokens. And the text length before was 4000+ bytes, but is now down to 985 tokens after compression. So BPE worked to compress the text using our learned vocabulary!

Next we would wrap this into an actual Tokenizer class with encode/decode methods.

### Decoding tokens back to text 

We now have a trained tokenizer consisting of:

- `merges`: map from byte pair to new token id 
- `vocab`: map from token id back to actual byte sequence

We will use these two mappings to implement a `decode()` method that takes a sequence of token ids and converts it back into a raw text string. 

Here is one way to implement the decode:

```python
def decode(self, ids):

    # create vocab mapping id --> byte sequence 
    vocab = {n: bytes([n]) for n in range(256)} 
    for (i1, i2), idx in merges.items():
        vocab[idx] = vocab[i1] + vocab[i2]

    # look up byte sequence for each token id
    tokens = [vocab[idx] for idx in ids]
    
    # join byte sequences into one long byte sequence
    all_bytes = b"".join(tokens)  

    # decode bytes into text string
    text = all_bytes.decode("utf-8", errors="replace")

    return text
```

Walk through it:

- First recreate vocab map from token id to byte sequence
- Look up byte sequence for each token id
- Concatenate byte sequences into one long byte stream
- Decode bytes into text string using utf-8
   - Handle invalid bytes with "replace" error mode

One tricky aspect is that not every arbitrary sequence of bytes is valid UTF-8 encoding. So we have to handle cases where the language model predicts an invalid sequence of tokens. We do this using the "replace" error mode when decoding.

This gives us a full decode functionality - we can now take a sequence of token ids predicted by the language model and convert it back into text!

Next we would wrap this into a Tokenizer class and implement the corresponding encode() method to finish our BPE tokenizer.


### GPT tokenizers add complexity on top of BPE

The GPT papers motivate use of BPE, but don't apply it naively. For example, common words like "dog" would merge with punctuation (dog., dog?, etc) which is undesirable. 

So GPT introduces a regex pattern to chunk up text before BPE. This prevents merges across categories like letters, numbers, punctuation.

Walking through the GPT-2 tokenizer code:

- They compile a complex regex pattern 
- Use `re.findall` on this pattern to split up text
- Only merge within each chunk, never across chunks
- This prevents merges like letter+punctuation

However, some oddities:

- The regex uses specific apostrophe symbols, inconsistent handling
- Doesn't properly separate whitespace, spaces never merge
- Training code was not released, unclear if just chunk+BPE

The GPT-4 tokenizer makes some improvements:

- Case insensitive regex matching 
- Handles whitespace more efficiently for code
- Limits number digit merges to 3 digits
- But overall very complex, undocumented process

**Special tokens** are also used, like `<|endoftext|>` to delimit documents. Special tokens are handled directly in code, not part of BPE merges. Can be added after training as well.

So in summary, GPT tokenizers add:

- Regex text splitting 
- Special case handling of spaces, apostrophes
- Special tokens
- Non-transparent training process

As a result, while they build on BPE, they are quite complex compared to the basic algorithm.

Next we'll look at wrapping the core ideas into a nice `Tokenizer` interface in Python.
