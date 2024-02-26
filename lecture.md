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

## Byte Pair Encoding Algorithm

Now that we have a sense for what tokenization looks like, let's dive into the details of implementing the Byte Pair Encoding (BPE) algorithm that is used to train these tokenizers.

The BPE algorithm itself is actually not too complicated. At a high level, here is what it does:

1. Start with all the raw bytes (0-255) as your initial "vocabulary"
2. Scan through the training text and count up all the pairs of consecutive bytes
3. Identify the most frequent pair of bytes 
4. Replace this frequent pair with a new, single token and add it to the vocabulary
5. Repeat steps 2-4, iteratively merging frequent pairs to build the vocabulary
<markdown "BPE Algorithm Example">
Let's delve into an alternative example to clearly demonstrate the Byte Pair Encoding (BPE) Algorithm, maintaining the original tone and format for consistency. Imagine our training text is a simple sequence of characters:

```
ZZXXZZXX
```

Our initial vocabulary consists of the unique characters Z and X:

First, we identify all consecutive pairs in the sequence:

```
ZZ : 2
XX : 2
ZX : 1
XZ : 1
```

Noticing "ZZ" and "XX" as the most frequent pairs, we decide to replace "ZZ" with a new token "A", updating our vocabulary:

```
Vocabulary: Z X A

AXXAXX
```

After this initial replacement, we scan the updated string to identify the next set of frequent pairs:

```
AX : 2
XX : 2
XA : 1
```

Choosing "XX" for our next replacement, we introduce "B" as a new token to our vocabulary:

```
Vocabulary: Z X A B

ABAB
```

This iterative process of scanning, counting, and merging continues until we achieve a desired vocabulary size, simplifying the original text's representation with each step.

So let's now implement this in Python. We'll start with some text, encode it to bytes, and then:

1. Implement getting pair statistics
2. Implement the merge operation
3. Put it together in a loop to iteratively merge frequent pairs

### Getting token statistics

Here is one way to implement getting the statistics. We simply iterate through consecutive token pairs, and count up frequencies using a dictionary:

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

### Merging pairs

Next, we need a `merge()` function that replaces occurrences of a particular pair with a new token:

```python
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i]) 
            i += 1
    return newids
```

This iterates through, copying elements one by one, but if we see the pair we are looking for, we append the new index instead and advance by 2.

### Putting it together

We iterate these steps, each time:

1. Call `get_stats()` to find the most frequent pair
2. Pass that pair and a new index to `merge()` to replace occurrences of that pair with the new token
3. Update the ids and vocabulary with the result

Here is what that looks like:

```python
# Vocabulary starts as raw bytes  
vocab = {i:bytes([i]) for i in range(256)} 

for i in range(num_merges):
    # Find most frequent pair
    stats = get_stats(ids)  
    pair = max(stats, key=stats.get)
    
    # Merge pair 
    idx = 256+i 
    ids = merge(ids, pair, idx)  
    
    # Add to vocab
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
```

And that's it! After iterating this for some number of merges, we will have trained a vocabulary using byte pair encoding. Next we'll look at how to encode and decode text using the trained merges.

Let me know if you would like me to elaborate on any part of this explanation!


## Encoding and Decoding

Now that we have trained a byte pair encoding tokenizer by iteratively merging frequent pairs, we can use the resulting `vocab` dictionary and `merges` dictionary to encode raw text into tokens, and decode tokens back into text.

### Decoding Tokens

First, let's implement decoding tokens back into text. We are given a list of integer token IDs, and we want to turn that into a Python string representing the original text.

The key insight is that our `vocab` dictionary maps from token IDs to the raw byte sequence for that token. For tokens 0-255, this is just the raw bytes. For merged tokens like 256 and up, this concatentates the byte sequences of the two tokens that were merged to create it.

So decoding just involves looking up each token ID in the vocab, concatenating all the byte sequences, and decoding from bytes into a string:

```python
def decode(ids):
    # Concatenate byte sequences 
    tokens = b"".join(vocab[idx] for idx in ids) 
    
    # Decode bytes into text
    text = tokens.decode("utf-8", errors="replace") 
    
    return text
```

Note that we have to handle invalid UTF-8 errors, just in case the model predicts an invalid sequence.

Let's check this works:

```python
print(decode([256])) # The first merged token
# 'ab' 
```

Great!

### Encoding Text

The other direction is encoding a raw text string into a list of token IDs. 

To do this, we:

1. Encode the text into bytes
2. Repeat merging pairs from the `merges` dictionary until nothing left to merge
3. Return final list of tokens

Here is an implementation:

```python 
def encode(text):
    # Encode text into bytes
    tokens = list(text.encode("utf-8"))  
    
    # Repeat merges 
    while len(tokens) >= 2:
        stats = get_stats(tokens)  
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break 
        idx = merges[pair]  
        tokens = merge(tokens, pair, idx)
        
    return tokens
```

We reuse the same merge logic as before. But we stop merging once there are no more eligible pairs in our set of `merges`.

Let's verify encoding and decoding works as expected:

```python
print(decode(encode("hello world"))) # hello world
```

And that's it! With these two functions, we can now translate bidirectionally between text and sequences of tokens.


## Special Tokens

So far we have focused on tokens that originate from merging byte pairs in the text. However, in practice language models rely heavily on special tokens that are manually defined and carry special meaning.

For example, in the GPT-2 tokenizer that we saw earlier, there is one special token defined:

```
<|endoftext|>
```

This end-of-text token is inserted between documents in the training data. So if we have some text like:

```
Text from document 1. <|endoftext|> Text from document 2.
```

The language model will learn that the appearance of the end-of-text token signals that what follows is unrelated to what came before. This provides a way to delimit documents during training.

In GPT-3 and GPT-4, additional special tokens were added. For example, if we look at the GPT-4 tokenizer definition:

```python
SPECIAL_TOKENS = {
    "bos_token": "<|endoftext|>", 
    "eos_token": "<|endoftext|>",
    "unk_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",  
    # ...
}
```

Here we see the end-of-text token, but also begin-of-sequence, end-of-sequence, unknown token, and more. As language models become more sophisticated, more special tokens are added to handle particular situations.

When new special tokens are added to the vocabulary, some model surgery is typically required as well:

- The embedding matrix needs new rows added and initialized for the new tokens 
- The final classifier layer needs output dimension increased to predict new tokens

So in summary, special tokens:

- Carry special meaning to delimit or structure the data
- Require changes to model architecture when added
- Enable more advanced LM capabilities

Some common use cases when adding special tokens:

- Switching from unsupervised pretraining to supervised finetuning 
- Adding support for conversations, multiple speakers, etc
- Adding interfaces to external environments


## Forced Splits Using Regex

So far we have covered the core byte pair encoding algorithm for merging frequent pairs. However, as discussed earlier, naively applying BPE can result in unhelpful merges. For example, common words followed by punctuation may get merged into a single token.

To prevent this, the GPT series tokenizers use additional regex rules to enforce that certain character types should never be merged.

### GPT-2

Here again is the regex used in the GPT-2 tokenizer:

```
gpt2_pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

Let's break this down:

- `'\s|'t|...` - Split on single apostrophes like 's, 't, etc.
- `?\p{L}+` - One or more letters
- `?\p{N}+` - One or more numbers
- `?[^\s\p{L}\p{N}]+` - One or more non-letter/non-number character
- `\s+(?!\S)` - Leading whitespace
- `\s+` Trailing whitespace

When this regex is applied to the text, it splits it greedily into chunks matching this pattern. BPE is then applied individually within each chunk.

So for example:

```python
print(re.findall(gpt2_pat, "Hello, how's it going?")) 

# ['Hello', ', ', 'how', "'s", ' ', 'it', ' ', 'going', '?']
```

By keeping punctuation like apostrophes in separate chunks, we prevent problematic merges across categories.

### GPT-3

In GPT-3, the regex was updated to additionally handle casing properly:

```
gpt3_pat = re.compile(r"""...'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

The key change is the added `re.IGNORECASE` flag, which causes it to match apostrophes regardless of case.

So in summary, these regex rules explicitly constrain the merges that BPE can consider, to prevent undesirable tokenizations.


## OpenAI's GPT-2 Encoder 

Now that we have covered the core BPE algorithm and extensions like special tokens and forced splits, let's briefly analyze the actual encoder code released by OpenAI for GPT-2. This will help connect the dots between the algorithm and a real-world implementation.

The file we are interested in is [encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py). There is a lot going on in this file, but let's focus on a few key points:

### Loading the tokenizer

At the bottom of the file, the encoder and merges (our `vocab` and `merges` respectively) are loaded from two files:

```python
with open('encoder.json', 'r') as f:
    encoder = json.load(f) 

with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
```

This is exactly equivalent to how we have loaded the trained `vocab` and `merges` so far.

### Encoding text

The main `encode()` method handles converting text into tokens. If we strip away some of the minor details, we can see that this follows the same general structure as our implementation:

```python
def encode(self, text):
    bpe_tokens = []
    for token in re.findall(self.pat, text):
        bpe_tokens.extend(self.bpe(token).split(' '))
    return bpe_tokens
```

Specifically:

1. Use regex to split text into chunks 
2. Apply BPE merge operations on each chunk
3. Concatenate the results

This matches our previous understanding.

### Decoding tokens

Similarly, the `decode()` method uses the `encoder` (vocab) to look up the bytes for each token and decodes from bytes into text, the same as we saw earlier.

So in summary, while the OpenAI implementation has additional complexities, at its core it is relying on the same BPE algorithm and data structures like `vocab` and `merges` that we have covered. The code connects nicely with our understanding of tokenization.


## SentencePiece Tokenizer

So far we have focused on byte-level Byte Pair Encoding, as used by OpenAI's GPT series. However, another commonly used tokenization library is [SentencePiece](https://github.com/google/sentencepiece) from Google. 

SentencePiece implements subword tokenization via BPE, but has a few key differences from what we have seen so far:

### Works on Unicode code points

The main difference is that SentencePiece runs BPE on the Unicode code points directly, rather than encoding to UTF-8 bytes first. 

So during training, it looks at the raw code points in the text, counts up pairs, and merges frequent pairs of code points rather than bytes.

### Byte fallback for rare code points

SentencePiece has a `character_coverage` parameter that determines the rarity threshold for code points. 

Code points that occur fewer times than this threshold are not assigned their own token. By default, rare code points are simply all mapped to a special `<unk>` token.

However, the `byte_fallback` option provides a secondary behavior - rare code points are UTF-8 encoded into bytes, and special byte tokens are added to the vocabulary.

So for example with coverage 99.995% and byte fallback, a code point seen only once in a huge corpus would become a byte token rather than `<unk>`.

### Training SentencePiece

Let's see a quick example of training SentencePiece on a tiny example text file:

```python
spm.SentencePieceTrainer.train(input="input.txt", 
                              model_prefix="sp10k", 
                              vocab_size=10000, byte_fallback=True)

sp = spm.SentencePieceProcessor()  
sp.load("sp10k.model")
```

We can then encode, decode, and inspect the vocabulary:

```python
ids = sp.encode("hello world")
print(sp.decode(ids))
print(sp.id_to_piece(257)) # First merge
```

So in summary, SentencePiece is an efficient tokenizer used by models like mT5 and FLAN, but has some differences from byte-level BPE to be aware of.



## Considerations for Vocabulary Size

An important hyperparameter in training a tokenizer is selecting the vocabulary size. What size vocabulary should we use? How does the vocabulary size impact model performance?

There are several tradeoffs to consider:

### Model Efficiency

Larger vocabularies require:

- Larger embedding matrices to store representation for each token
- More computation for the word prediction softmax

So they increase the memory usage and FLOPs for the model.

### Sequence Length

However, a larger vocabulary also allows more character chunks to be merged into single tokens. This makes the tokenized text more compact, allowing the model to attend over more context.

So there is a tradeoff between sequence length and model efficiency.

### Undertraining

With an extremely large vocabulary (e.g. millions of tokens), each token may occur very rarely in the training data.

This could result in "undertraining" - the token embeddings don't get updated enough during training to build good representations.

### Information Density

On the other hand, an extremely small vocabulary over-merges information into a single token. This could make it hard for the model to learn the significance of different components that have been merged.

So in reality there is a "sweet spot" vocabulary size that balances these tradeoffs - typically tens to hundreds of thousands of tokens for state-of-the-art models.

### Extending a Trained Model

Note that it is also possible to extend a trained model's vocabulary with additional special tokens using the "model surgery" techniques we discussed earlier.

The core model vocabulary provides broad coverage, and additional tokens can add capabilities.

## Recommendations

Taking the above into account, here are my recommendations when considering vocabulary size:

- Start in the 10,000-100,000 range based on model size 
- Evaluate model quality and sequence length tradeoffs
- Increase size if seeing over-compression issues
- Avoid >1 million tokens unless huge data
- Add special tokens judiciously for new capabilities

## Recommendations and Conclusion

We have covered a lot of ground on the complex topic of tokenization in large language models! Let's recap some key points and recommendations:

### Don't brush it off
There are many sharp edges and pitfalls that can arise from tokenization. It is an often overlooked contributor to issues around spelling, arithmetic, coding, safety, and more in LLMs. Take the time to understand it.

### Reuse existing tokenizers when possible
Tools like the tiktoken library make it easy to reuse robust, pre-trained tokenizers like GPT-2 and GPT-4. Leverage these instead of training your own when possible.

### SentencePiece offers efficiency
When you do need to train a custom tokenizer, SentencePiece provides an efficient implementation of BPE training and inference. But watch out for the many configuration pitfalls.

### Look out for advances
There is active research ongoing into potential methods for eliminating tokenization entirely. This would be a major breakthrough! Follow papers from groups like Anthropic working in this space.

### Mind the context length
When selecting vocabulary size and merges, keep in mind the impact on overall sequence length and context an LLM can attend over. Find the right balance for your use case. 

### Special tokens enable capabilities
Judicious addition of special tokens allows extending a base LLM for downstream tasks while minimizing changes to the core model.

And that concludes our tour of tokenization for LLMs! As you continue working with language models, keep tokenization in mind as a source of quirks or issues. And hopefully in time, more advanced techniques may start to simplify or eliminate this required step altogether.

Let me know if you have any final questions!
