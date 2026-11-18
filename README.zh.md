# minbpe

[English](README.md) | [简体中文](README.zh.md)

用于大语言模型（LLM）分词的高质量、极简（字节级）字节对编码（Byte Pair Encoding, BPE）算法实现。由于该算法直接运行在 UTF-8 编码的字节流上，因此属于“字节级”分词算法。

该算法由 OpenAI 的 [GPT-2 论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 及其开源的 [GPT-2 代码实现](https://github.com/openai/gpt-2) 推广普及至大语言模型领域。[Sennrich 等人 2015 年的论文](https://arxiv.org/abs/1508.07909) 被公认为将 BPE 引入自然语言处理（NLP）应用的鼻祖文献。如今，几乎所有现代前沿大模型（如 GPT、Llama、Mistral）均采用该算法训练对应的分词器（Tokenizer）。

本代码库包含两个主要分词器，二者均完整具备分词器的三大核心功能：1）基于给定语料文本训练词表（Vocabulary）与合并规则（Merges）；2）将文本编码为词元 ID（Encode）；3）将词元 ID 解码还原为文本（Decode）。代码库结构如下：

1. [minbpe/base.py](minbpe/base.py)：实现分词器基类 `Tokenizer`。包含 `train`、`encode`、`decode` 接口原型、模型存取（Save/Load）功能以及通用辅助工具函数。该类仅供派生继承，不建议直接实例化使用。
2. [minbpe/basic.py](minbpe/basic.py)：实现 `BasicTokenizer`，属于最纯粹基础的 BPE 算法实现，直接在文本字节流上进行合并统计。
3. [minbpe/regex.py](minbpe/regex.py)：实现 `RegexTokenizer`。在分词前先使用正则表达式切分输入文本，按照类别（如字母、数字、标点符号）进行预处理分块，确保合并操作绝不跨越类别边界。该策略由 GPT-2 论文引入，并在 GPT-4 中延续使用。该类同时负责处理特殊词元（Special Tokens）。
4. [minbpe/gpt4.py](minbpe/gpt4.py)：实现 `GPT4Tokenizer`。该类是对 `RegexTokenizer` 的轻量封装，能够 1:1 精确复现 [tiktoken](https://github.com/openai/tiktoken) 库中 GPT-4 分词器的编码结果。封装中处理了分词器合并规则逆向还原以及历史遗留的个别单字节词元置换细节。

脚本 [train.py](train.py) 在示例语料 [tests/taylorswift.txt](tests/taylorswift.txt)（维基百科条目）上训练两个核心分词器，并将词表持久化至磁盘以便可视化对比。该脚本在 Apple M1 MacBook 上运行耗时约 25 秒。

上述所有代码文件均保持极致精简，并附有详尽的代码注释，文件底部均提供开箱即用的使用范例。

## 快速上手 (Quick Start)

作为最简示例，可通过以下代码复现 [维基百科 BPE 词条](https://en.wikipedia.org/wiki/Byte_pair_encoding) 中的合并过程：

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 前 256 个为单字节基础词元，随后执行 3 次合并
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# 写入两个文件：toy.model（用于加载）与 toy.vocab（用于查阅展示）
```

维基百科示例中，对输入字符串 `"aaabdaaabac"` 进行 3 次合并后得到 `"XdXac"`（其中 $X=ZY$, $Y=ab$, $Z=aa$）。在 `minbpe` 的设计中，前 256 个词元 ID（0 ~ 255）始终保留分配给基础字节，随后的合并词元 ID 从 256 开始递增。因此：$a=97, b=98, c=99, d=100$（对应其 [ASCII 码](https://www.asciitable.com)）。当 $(a, a)$ 合并为 $Z$ 时，$Z$ 的 ID 为 256；同理 $Y$ 为 257，$X$ 为 258。初始词表包含 256 个字节，执行 3 次合并后词表大小为 259，最终编码输出即为预期的 `[258, 100, 258, 97, 99]`。

## 推理验证：与 GPT-4 (tiktoken) 对齐比对

可通过以下代码验证 `RegexTokenizer` 与 OpenAI [tiktoken](https://github.com/openai/tiktoken) 库中 GPT-4 分词器的功能等价性：

```python
text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken 官方实现
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]

# minbpe 实现
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text))
# [15339, 4513, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 57037]
```

（运行前需执行 `pip install tiktoken` 安装依赖）。底层实现上，`GPT4Tokenizer` 仅是对 `RegexTokenizer` 传入 GPT-4 的预置合并规则与特殊词元后的轻量封装。同时可验证特殊词元（Special Tokens）的正确解析：

```python
text = "<|endoftext|>hello world"

# tiktoken 官方实现
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text, allowed_special="all"))
# [100257, 15339, 1917]

# minbpe 实现
from minbpe import GPT4Tokenizer
tokenizer = GPT4Tokenizer()
print(tokenizer.encode(text, allowed_special="all"))
# [100257, 15339, 1917]
```

注意：与 tiktoken 一致，在调用 `encode` 时必须显式声明解析特殊词元的意图。否则容易引发严重安全漏洞（例如意外将不可信的用户提示词解析为特殊控制词元）。`allowed_special` 参数可设为 `"all"`、`"none"` 或允许的特殊词元白名单列表。

## 词表训练 (Training)

与仅提供推理功能的 tiktoken 不同，本代码库支持从零训练自定义分词器。理论上，若在海量语料上将 `RegexTokenizer` 的词表规模训练至 100K，即可完全复现 GPT-4 分词器。

训练支持两种技术路径：

第一种，如果无需正则分块预处理的复杂度，亦不需要特殊词元，可直接选用 `BasicTokenizer`：

```python
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.train(very_long_training_string, vocab_size=4096)
tokenizer.encode("hello world") # 文本 -> 词元列表
tokenizer.decode([1000, 2000, 3000]) # 词元列表 -> 文本
tokenizer.save("mymodel") # 写入 mymodel.model 与 mymodel.vocab
tokenizer.load("mymodel.model") # 从磁盘恢复模型；.vocab 仅用于肉眼查看
```

第二种，如需遵循 OpenAI 现代分词器的成熟范式，建议使用正则表达式按字符类别进行预先分割。`RegexTokenizer` 默认采用 GPT-4 的拆分正则模式：

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.encode("hello world") # 文本 -> 词元列表
tokenizer.decode([1000, 2000, 3000]) # 词元列表 -> 文本
tokenizer.save("tok32k") # 写入 tok32k.model 与 tok32k.vocab
tokenizer.load("tok32k.model") # 从磁盘重新加载模型
```

其中 `vocab_size` 应根据训练语料的规模大小进行合理调整。

**特殊词元注册**：若需要向分词器添加特殊控制词元，可使用 `register_special_tokens` 函数。例如，当训练的 `vocab_size` 为 32768 时，前 256 个为单字节词元，接下来的 32768 - 256 个为合并词元。最后一个真实合并词元的 ID 为 32767（`vocab_size - 1`），因此首个特殊词元的 ID 应紧随其后分配为 32768：

```python
from minbpe import RegexTokenizer
tokenizer = RegexTokenizer()
tokenizer.train(very_long_training_string, vocab_size=32768)
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
```

支持按需继续追加更多特殊词元。本项目核心宗旨是保持代码的极致纯净、高可读性与易魔改性。建议直接阅读源码以深入理解其底层运行机理，测试用例集中亦包含丰富的调用示范。

## 单元测试 (Tests)

项目使用 pytest 进行自动化单元测试，所有测试代码均位于 `tests/` 目录中。首先执行 `pip install pytest` 安装依赖，随后运行：

```bash
$ pytest -v .
```

（`-v` 为详细输出模式，界面展示更加直观）。

## 社区拓展 (Community Extensions)

* [gnp/minbpe-rs](https://github.com/gnp/minbpe-rs)：`minbpe` 的 Rust 语言实现版本，与 Python 版本保持高度的一致性与对应关系。

## 进阶练习 (Exercise)

对于深入学习 BPE 算法的研究者与开发者，项目提供了循序渐进构建专属 minbpe 的指导练习手册，详见 [exercise.md](exercise.md)。

## 教学视频与讲义 (Lecture)

本代码库的完整构建讲解过程可参阅作者的 [YouTube 教学视频](https://www.youtube.com/watch?v=zduSFxRajkE)。该视频的完整文字版讲义可阅读 [lecture.md](lecture.md)。

## 后续规划 (TODOs)

- 编写优化版 Python 实现，以便更高效地处理超大文件与大规模词表
- 编写深度优化的 C 或 Rust 版本
- 将 `GPT4Tokenizer` 重构更名为 `GPTTokenizer`，并同步支持 GPT-2 / GPT-3 / GPT-3.5
- 编写类似于 `GPT4Tokenizer` 的 `LlamaTokenizer`（即尝试对齐 SentencePiece 的功能）

## 许可证 (License)

[MIT 许可证](LICENSE)
