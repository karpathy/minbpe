"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer, ParallelTokenizer, ParallelRegexTokenizer, BlockBPETokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/siku.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

# 记录总时间
total_start_time = time.time()

# 存储每个分词器的训练时间
training_times = []

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer, ParallelTokenizer, BlockBPETokenizer], ["basic", "regex", "parallel", "BlockBPE"]):
    print(f"\n--- Training {name} tokenizer ---")

    # 记录单个分词器的开始时间
    tokenizer_start_time = time.time()

    # 构造分词器对象并开始详细训练
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)

    # 记录单个分词器的结束时间
    tokenizer_end_time = time.time()
    tokenizer_time = tokenizer_end_time - tokenizer_start_time
    training_times.append((name, tokenizer_time))

    # 在模型目录中写入两个文件：name.model 和 name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)

    print(f"{name} tokenizer training completed in {tokenizer_time:.2f} seconds")

# 记录总结束时间
total_end_time = time.time()
total_time = total_end_time - total_start_time

print(f"\n=== Training Summary ===")
for name, t_time in training_times:
    print(f"{name} tokenizer: {t_time:.2f} seconds")

print(f"\nTotal training time: {total_time:.2f} seconds")
print(f"Average per tokenizer: {total_time/len(training_times):.2f} seconds")
