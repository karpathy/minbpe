"""
BlockBPE: Parallel BPE Tokenization Implementation

基于论文《BlockBPE: Parallel BPE Tokenization》设计的GPU并行BPE分词器实现
实现了字节级预分词、并行合并操作和GPU优化策略
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch

class BlockBPETokenizer:
    """
    并行BPE分词器实现，优化GPU批量推理场景
    
    关键特性:
    - 字节级预分词替代Regex预分词
    - 并行合并操作优化
    - GPU内存访问优化
    - 动态块大小调整
    """
    
    def __init__(self, 
                 vocab_size: int = 50257, 
                 special_tokens: Optional[Dict[str, int]] = None,
                 max_block_size: int = 1024,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化BlockBPE分词器
        
        参数:
            vocab_size: 词汇表大小，默认50257(GPT-2标准)
            special_tokens: 特殊标记字典，如{'<|endoftext|>': 100257}
            max_block_size: 最大GPU线程块大小
            device: 计算设备，自动检测CUDA可用性
        """
        self.vocab_size = vocab_size
        self.max_block_size = max_block_size
        self.device = device
        
        # 初始化特殊标记
        self.special_tokens = special_tokens or {}
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # 初始化合并表和词汇表
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self._initialize_base_vocab()
        
        # GPU优化数据结构
        self.merge_ranks_gpu = None
        self.vocab_gpu = None
        
    def _initialize_base_vocab(self):
        """初始化基础词汇表(0-255字节)"""
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def train(self, text, vocab_size, verbose=False):
        """
        训练BPE分词器
    
        参数:
            text: 训练文本
            vocab_size: 词汇表大小
            verbose: 是否输出详细训练信息
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 输入文本预处理：字节级预分词
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        # 初始化合并记录和词汇表
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
    
        # 迭代合并最常见字节对
        for i in range(num_merges):
            # 统计当前字节对频率
            stats = self._get_stats(ids)
            if not stats:
                break  # 没有可合并的字节对时提前终止

            # 找到频率最高的字节对
            pair = max(stats, key=stats.get)
        
            # 创建新token ID
            idx = 256 + i
        
            # 执行合并操作
            ids = self._merge(ids, pair, idx)
        
            # 保存合并规则和更新词汇表
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        
            # 合并成功后打印详细信息
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
    
        # 保存训练结果
        self.merges = merges
        self.vocab = vocab

    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """统计字节对频率"""
        counts = defaultdict(int)
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            counts[pair] += 1
        return counts
    
    def _merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """执行合并操作"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _prepare_gpu_data_structures(self):
        """准备GPU优化的数据结构"""
        if self.device.startswith("cuda"):
            # 将合并规则转换为GPU张量
            merge_pairs = list(self.merges.keys())
            merge_values = list(self.merges.values())
            
            if merge_pairs:
                # 创建合并对和对应值的张量
                pairs_tensor = torch.tensor(merge_pairs, dtype=torch.long, device=self.device)
                values_tensor = torch.tensor(merge_values, dtype=torch.long, device=self.device)
                
                # 使用哈希表优化查找
                self.merge_ranks_gpu = (pairs_tensor, values_tensor)
            
            # 词汇表也可以转换为GPU数据结构
            vocab_keys = list(self.vocab.keys())
            vocab_values = list(self.vocab.values())
            
            # 注意: 这里需要特殊处理bytes类型
            # 实际实现中可能需要将词汇表内容转换为更适合GPU的格式
            
    def encode(self, 
               text: Union[str, List[str]], 
               allowed_special: str = "none_raise") -> Union[List[int], List[List[int]]]:
        """
        编码文本为token IDs
        
        参数:
            text: 输入文本或文本列表
            allowed_special: 特殊标记处理方式
            
        返回:
            token IDs或token IDs列表
        """
        if isinstance(text, list):
            return [self._encode_single(t, allowed_special) for t in text]
        return self._encode_single(text, allowed_special)
    
    def _encode_single(self, text: str, allowed_special: str) -> List[int]:
        """编码单个文本"""
        # 处理特殊标记
        special = self._process_allowed_special(allowed_special, text)
        
        if not special:
            return self._encode_ordinary(text)
        
        # 分割包含特殊标记的文本
        special_pattern = "(" + "|".join(re.escape(k) for k in special.keys()) + ")"
        chunks = re.split(special_pattern, text)
        
        # 编码各片段
        ids = []
        for chunk in chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self._encode_ordinary(chunk))
        
        return ids
    
    def _process_allowed_special(self, allowed_special: str, text: str) -> Dict[str, int]:
        """处理特殊标记选项"""
        if allowed_special == "all":
            return self.special_tokens
        elif allowed_special == "none":
            return {}
        elif allowed_special == "none_raise":
            # 检查文本中是否包含特殊标记
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(f"Special token {token} found in text but not allowed")
            return {}
        elif isinstance(allowed_special, set):
            return {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
    
    def _encode_ordinary(self, text: str) -> List[int]:
        """普通编码(无特殊标记处理)"""
        # 字节级预分词
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # 应用合并规则
        if self.device.startswith("cuda"):
            # GPU优化编码
            return self._encode_gpu(ids)
        else:
            # CPU编码
            return self._encode_cpu(ids)
    
    def _encode_cpu(self, ids: List[int]) -> List[int]:
        """CPU编码实现"""
        # 应用合并规则直到无法继续合并
        while True:
            # 查找可合并的字节对
            stats = self._get_stats(ids)
            if not stats:
                break
                
            # 找到优先级最高的合并对(最低的合并索引)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            # 如果没有可合并的对，终止
            if pair not in self.merges:
                break
                
            # 执行合并
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
        
        return ids
    
    def _encode_gpu(self, ids: List[int]) -> List[int]:
        """
        GPU优化编码实现
        
        注意: 这是一个简化的GPU编码模拟
        实际实现需要使用CUDA内核和GPU并行原语
        """
        # 将数据移动到GPU
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        n = len(ids)
        
        # 确定最佳块大小
        block_size = self._determine_block_size(n)
        
        # 迭代应用合并规则
        changed = True
        while changed and n > 1:
            changed = False
            
            # 分块处理(模拟GPU并行)
            for start in range(0, n, block_size):
                end = min(start + block_size, n)
                block_ids = ids_tensor[start:end]
                
                # 在块内查找可合并的对
                for i in range(len(block_ids) - 1):
                    if i >= len(block_ids) - 1:
                        continue
                        
                    pair = (block_ids[i].item(), block_ids[i+1].item())
                    if pair in self.merges:
                        # 执行合并
                        idx = self.merges[pair]
                        # 更新块(实际实现中需要更复杂的并行合并策略)
                        new_block = torch.cat([
                            block_ids[:i],
                            torch.tensor([idx], dtype=torch.long, device=self.device),
                            block_ids[i+2:]
                        ])
                        block_ids = new_block
                        changed = True
                        break
            
            # 更新IDs(实际实现中需要更高效的方法)
            if changed:
                # 这里简化处理，实际需要更复杂的并行压缩算法
                ids_tensor = torch.cat([ids_tensor[:start], block_ids, ids_tensor[end:]])
                n = len(ids_tensor)
        
        return ids_tensor.cpu().tolist()
    
    def _determine_block_size(self, seq_length: int) -> int:
        """
        根据序列长度确定最佳块大小
        
        基于论文中的发现:
        - 长序列适合大块大小
        - 短序列适合小块大小
        """
        if seq_length >= 2048:
            return min(self.max_block_size, 1024)  # 大块处理长序列
        elif seq_length <= 256:
            return min(self.max_block_size, 256)   # 小块处理短序列
        else:
            return min(self.max_block_size, 512)   # 中等块大小
    
    def encode_batch(self, 
                     texts: List[str], 
                     allowed_special: str = "none_raise") -> List[List[int]]:
        """
        批量编码文本
        
        参数:
            texts: 文本列表
            allowed_special: 特殊标记处理方式
            
        返回:
            token IDs列表
        """
        # 确定最佳批处理策略
        if self.device.startswith("cuda"):
            # GPU批量编码
            return self._encode_batch_gpu(texts, allowed_special)
        else:
            # CPU批量编码
            return [self.encode(text, allowed_special) for text in texts]
    
    def _encode_batch_gpu(self, texts: List[str], allowed_special: str) -> List[List[int]]:
        """
        GPU批量编码实现
        
        注意: 这是一个简化的批量编码模拟
        实际实现需要使用CUDA流和并行执行
        """
        results = []
        
        # 并行处理每个文本(实际实现中会使用GPU并行)
        for text in texts:
            # 处理特殊标记
            special = self._process_allowed_special(allowed_special, text)
            
            if not special:
                # 普通编码
                text_bytes = text.encode("utf-8")
                ids = list(text_bytes)
                results.append(self._encode_gpu(ids))
            else:
                # 包含特殊标记的编码
                special_pattern = "(" + "|".join(re.escape(k) for k in special.keys()) + ")"
                chunks = re.split(special_pattern, text)
                
                # 编码各片段
                ids = []
                for chunk in chunks:
                    if chunk in special:
                        ids.append(special[chunk])
                    else:
                        chunk_bytes = chunk.encode("utf-8")
                        chunk_ids = list(chunk_bytes)
                        ids.extend(self._encode_gpu(chunk_ids))
                
                results.append(ids)
        
        return results
    
    def decode(self, ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        解码token IDs为文本
        
        参数:
            ids: token IDs或token IDs列表
            
        返回:
            文本或文本列表
        """
        if isinstance(ids[0], list):
            return [self._decode_single(id_list) for id_list in ids]
        return self._decode_single(ids)
    
    def _decode_single(self, ids: List[int]) -> str:
        """解码单个token IDs序列"""
        text_bytes = b""
        
        for idx in ids:
            if idx in self.vocab:
                text_bytes += self.vocab[idx]
            elif idx in self.inverse_special_tokens:
                text_bytes += self.inverse_special_tokens[idx].encode("utf-8")
            else:
                # 处理未知token
                text_bytes += b"<unk>"
        
        return text_bytes.decode("utf-8", errors="replace")
    
    def register_special_tokens(self, special_tokens: Dict[str, int]) -> None:
        """
        注册特殊标记
        
        参数:
            special_tokens: 特殊标记字典
        """
        self.special_tokens.update(special_tokens)
        self.inverse_special_tokens.update({v: k for k, v in special_tokens.items()})
    
    def get_vocab(self) -> Dict[int, bytes]:
        """获取词汇表"""
        return self.vocab
    
    def get_merges(self) -> Dict[Tuple[int, int], int]:
        """获取合并规则"""
        return self.merges
    
    def save(self, filepath: str) -> None:
        """保存分词器到文件"""
        import pickle
        
        data = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'merges': self.merges,
            'vocab': self.vocab,
            'max_block_size': self.max_block_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, device: str = None) -> 'BlockBPETokenizer':
        """从文件加载分词器"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # 创建分词器实例
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            special_tokens=data['special_tokens'],
            max_block_size=data.get('max_block_size', 1024),
            device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # 恢复状态
        tokenizer.merges = data['merges']
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_special_tokens = {v: k for k, v in tokenizer.special_tokens.items()}
        
        # 准备GPU数据结构
        if tokenizer.device.startswith("cuda"):
            tokenizer._prepare_gpu_data_structures()
        
        return tokenizer


# 辅助函数和工具类
class BPETrainer:
    """BPE训练工具类"""
    
    @staticmethod
    def learn_bpe_from_file(filepath: str, 
                           vocab_size: int, 
                           special_tokens: Dict[str, int] = None) -> BlockBPETokenizer:
        """
        从文件学习BPE分词器
        
        参数:
            filepath: 训练文本文件路径
            vocab_size: 词汇表大小
            special_tokens: 特殊标记
            
        返回:
            训练好的分词器
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokenizer = BlockBPETokenizer(
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        
        tokenizer.train(text, verbose=True)
        return tokenizer


# 性能测试工具
class Benchmark:
    """分词器性能测试工具"""
    
    @staticmethod
    def benchmark_tokenizer(tokenizer: BlockBPETokenizer, 
                           texts: List[str], 
                           num_runs: int = 10) -> Dict[str, float]:
        """
        基准测试分词器性能
        
        参数:
            tokenizer: 要测试的分词器
            texts: 测试文本列表
            num_runs: 测试运行次数
            
        返回:
            性能指标字典
        """
        import time
        
        # 预热
        for text in texts[:10]:
            tokenizer.encode(text)
        
        # 编码时间测试
        start_time = time.time()
        for _ in range(num_runs):
            for text in texts:
                tokenizer.encode(text)
        encode_time = time.time() - start_time
        
        # 解码时间测试
        encoded_texts = [tokenizer.encode(text) for text in texts]
        start_time = time.time()
        for _ in range(num_runs):
            for ids in encoded_texts:
                tokenizer.decode(ids)
        decode_time = time.time() - start_time
        
        # 批量编码时间测试
        start_time = time.time()
        for _ in range(num_runs):
            tokenizer.encode_batch(texts)
        batch_encode_time = time.time() - start_time
        
        return {
            'encode_time': encode_time,
            'decode_time': decode_time,
            'batch_encode_time': batch_encode_time,
            'avg_encode_time': encode_time / (num_runs * len(texts)),
            'avg_decode_time': decode_time / (num_runs * len(texts)),
            'avg_batch_encode_time': batch_encode_time / (num_runs * len(texts))
        }
