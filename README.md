# QuickBPE

This is a much faster version of the MinBPE tokenizer from andrej karpathy. The main functions are optimized in c++ and then connected to python using ctypes, so that you can call them conveniently. I already successfully tokenized the entire TinyStories dataset in around 8 minutes, which is ~3.5 gb of text. The main bottleneck is now the regex splitting, which is hard to optimize since i decided to keep it integrated into python (so that it is still easy to change the split pattern). The training algorithm i used is from [here](https://arxiv.org/abs/2306.16837), which mentions a running time of O(nlog(m)). This is an overestimation. The true running time is O(n + mlog(m)) i think, which is linear in the sequence length in practice. The training took about 2 minutes on ~100mb of text, which seems to be decent. But there is probably still a lot of improvement that can be done. Also the encode function is much slower than the encode_ordinary function if the special tokens are distributed evenly because of the splitting. This still needs to be fixed.

# Quickstart
You can use the repo in the same way as the MinBPE repo. Make sure to use RegexTokenizerFast and encode_ordinary (the encode function is not as fast sometimes, but still faster than the python version)

```python
from minbpe import RegexTokenizerFast
tokenizer = RegexTokenizerFast()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
print(tokenizer.encode_ordinary(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
```

## todos

- rename GPT4Tokenizer to GPTTokenizer and support GPT-2/GPT-3/GPT-3.5 as well?
- write a LlamaTokenizer similar to GPT4Tokenizer (i.e. attempt sentencepiece equivalent)

## License

MIT
