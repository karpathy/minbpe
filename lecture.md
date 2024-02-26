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


### BPE Algorithm Implementation

Now that we have a visual sense of how tokenization works, let's implement the BPE algorithm ourselves. While the BPE algorithm itself is actually not too complicated, there are some subtleties in getting an implementation that properly handles encoding and decoding. 

We'll start by loading some sample text that we can tokenize. To get some realistic text, I'm going to grab the first paragraph from [this nice blog post](https://www.reedbeta.com/blog/programmers-intro-to-unicode/) introducing Unicode:

```python
text = """Unicode! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."""
```

The first step is to encode this text into bytes using UTF-8. Let's also convert the bytes into a list of integers to make them easier to work with:

```python 
tokens = text.encode("utf-8") 
tokens = list(map(int, tokens))
print(tokens[:10])

# [85, 110, 105, 99, 111, 100, 101, 33, 32, 8230]
```

Next, we'll write a function that scans through this list of tokens and counts up the frequency of each byte pair. This will allow us to identify the most common byte pair to merge first:

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```

We can now find the most common pair:

```python
stats = get_stats(tokens)
top_pair = max(stats, key=stats.get) 
print(top_pair)

# (101, 32)
```

The next step is to write code to merge this top pair. We will do this by iterating through the list of tokens, and whenever we see an occurrence of the pair, we will replace it with a new token ID (e.g. 256).

```python
def merge(ids, pair, idx):
  newids = []
  for i in range(len(ids)):
    if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        newids.append(idx)
        i += 1 
    else:
        newids.append(ids[i])
  return newids

ids = merge(ids, top_pair, 256)  
```

We can now repeat this process, by continuously finding the next most common pair, merging it, and updating the tokens list. We'll do this in a loop until we reach our target vocabulary size.

Finally, we can build decoding tables to map the new token IDs back to their corresponding byte sequences, allowing us to convert tokenized IDs back into text.

And that's it! With these basic building blocks, we now have a full BPE tokenizer for encoding text into tokens and decoding tokens back into text.


### Special Tokens

In addition to the tokens that come from raw bytes and byte pair merges, we can also introduce special tokens that serve particular purposes in our models. Special tokens allow us to embed additional structure and meaning into the token streams.

For example, in the GPT-2 tokenizer we were looking at earlier, there is only one special token defined:

```python
special_tokens = ['<|endoftext|>']
```

This `<|endoftext|>` token is used to delimit documents in the training data. So when we are creating our training set, we take a large corpus of text documents from the internet, tokenize them independently, and then insert this special `<|endoftext|>` token in between each document. 

This gives the language model a signal that one document has ended and a new, likely unrelated document is beginning next. The model has to learn from the training data that when it sees this token, it should "reset" its context and stop trying to continue the text from the previous document.

We can see it in action with the tiktokenizer web app example from earlier:

```
Hello world!!! <|endoftext|> This is a new document.
```

The `<|endoftext|>` token has a dedicated ID (in this case 50256) that is handled specially by the tokenizer code and will be substituted when encoding text.

In the GPT-4 tokenizer, a few additional special tokens were added:

```python
special_tokens = ['<|endoftext|>', '<|im_0|>', '<|im_1|>', '<|im_2|>']  
```

The `<|im_*|>` tokens are related to a strategy called [Fill-in-the-Middle](https://arxiv.org/abs/2211.01458) prompting. The details of this approach are beyond the scope here, but the key point is that it is very common to take a pretrained model and extend it with additional special tokens to enable new functionality when doing fine-tuning.

Adding special tokens requires some surgery to the model architecture itself. For example, we need to:

- Expand token embedding matrix by adding a new row 
- Expand the final classifier layer to produce an extra logit
- Initialize the new parameters and train just those new parts

But it is relatively straightforward to do. And many compelling prompting-based techniques build on this idea of injecting semantic meaning into token streams through the injection of special tokens.

The tiktokenizer library makes it easy to add/handle new special tokens when using models derived from GPT-2 or GPT-4. So in your own projects, reuse of existing tokens or extension with new special tokens is something you should consider.


### Tokenization and LLM Performance

Now that we understand what tokenization is and how algorithms like BPE work, an important question is: how does our choice of tokenizer impact model performance? There are several ways tokenization can influence LLM accuracy, efficiency, and generalization capability:

**Accuracy**: The tokenization scheme determines the model's effective "vocabulary". Better coverage of concepts with dedicated tokens tends to improve accuracy. For example, in code tasks, handling indentation and whitespace with merge rules can improve Python coding ability.

**Efficiency**: The number of tokens to represent text impacts context length and memory usage. An inefficient encoding bloats up sequence lengths, using up the model's fixed context window for nothing. Finding the right level of compression is key.

**Generalization**: How consistently similar concepts share token representations affects generalization. Groupings that are too narrow can reduce ability to connect related ideas. But groupings that are too broad also have downsides. There are tradeoffs around granularity.

In practice, extensive experimentation goes into finding the optimal tokenizer configuration and vocabulary size. For example, when moving from GPT-2 to GPT-3, there were [significant tweaks](https://arxiv.org/abs/2005.14165) to the vocabulary and merging rules that improved performance. So your tokenization can make or break your LLM!

### Tokenization Libraries and Tools

There are several popular tokenization libraries available that implement variants of the BPE algorithm:

- **[YTTM](https://github.com/youtokentome/yttm)**: Python, focuses on efficiency 
- **[HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)**: Rust, feature-rich
- **[SentencePiece](https://github.com/google/sentencepiece)**: C++, often used with TensorFlow
- **[tiktoken](https://github.com/ajALT/tiktoken)**: Rust, simple implementation from Anthropic

And tools like the **[tiktokenizer](https://tiktokenizer.com/)** web app provide convenient visualization.

When evaluating tokenizers, some key aspects to consider are:

- **Speed**: Critical path for inference, aim for ≥ 10K tokens/sec
- **Vocabulary**: Size, optimal merges, special tokens provided 
- **Control**: Ability to customize vocabulary as needed
- **Ease of use**: Clean APIs in your language of choice

Try out a few options hands-on with your own data to determine what fits your needs.

### Advanced Topics 

There are several advanced techniques around tokenization that are useful to know about:

- **Subword regularization**: Further break down rare words into common subwords to improve generalization.

- **Masked language modeling (MLM)**: Special pretraining approach that masks random tokens and tries to predict them. Changes distributional properties.

- **Discrete VAEs**: Use a VAE to compress text into a smaller latent code, and decode tokens from there. More efficient.

- **Character-level modeling**: Replace tokenization entirely by working on raw UTF-8 bytes. Very difficult but potential for future improvements.

I may cover some of these approaches in more detail in future posts. But I hope this tutorial has given you a solid starting point for thinking about and working with text encoding for large language models. Tokenization may be an annoyance, but with deeper knowledge you can turn it to your advantage!