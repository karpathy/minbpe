"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Based on BasicTokenizer:
- Parallel counting with multi CPU cores
- Does not parallel merging to avoid cross-trunk issue
"""

from .base import Tokenizer, get_stats, merge
import multiprocessing
from collections import Counter

# 将函数移到类外部，使其成为模块级函数
def parallel_get_stats(chunk):
    return get_stats(chunk)

class ParallelTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.num_workers = multiprocessing.cpu_count()  # 自动获取CPU核心数

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 输入文本预处理
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # 初始化词汇表和合并记录
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # 迭代合并最常见字节对
        for i in range(num_merges):
            # 将ID列表分块处理
            chunk_size = (len(ids) + self.num_workers - 1) // self.num_workers
            chunks = [ids[j:j+chunk_size] for j in range(0, len(ids), chunk_size)]

            # 使用进程池并行计算
            with multiprocessing.Pool(self.num_workers) as pool:
                results = pool.map(parallel_get_stats, chunks)

            # 合并统计结果
            stats = Counter()
            for res in results:
                stats.update(res)

            if not stats:
                break  # 如果没有可合并的pair则提前终止

            # 找到频率最高的字节对
            pair = max(stats, key=stats.get)

            # 创建新token
            idx = 256 + i

            # 执行合并操作
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # 保存结果
        self.merges = merges
        self.vocab = vocab
