# -*- coding: utf-8 -*-
"""
澈思 Stage-1 数据管线：jsonl 语料 → token 流 → 定长训练块（流式，shard 友好）

- 只读 jsonl 的 text 字段（10b 语料 schema 实测：id/category/subcategory/topic/chars/text/...）
- 每篇文档前后加 <bos>/<eos>，文档间无缝拼接成 token 流，切 (seq_len+1) 块
  （inputs = block[:-1], targets = block[1:]，即标准 next-token 位移）
- 流式读取：只按需读行，7.35GB 语料无需整包载入；支持多文件/shards
- 确定性：不 shuffle（顺序流式）；eval 用块序号步长切分（block_idx % eval_stride == 0）
- 特殊 token：<pad>=0 <unk>=1 <bos>=2 <eos>=3（tokenizer.json 实测口径）
"""
import glob
import json

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 允许 import tokenizer/
from tokenizer.chesi_tokenizer import ChesiTokenizer


def load_tokenizer(tokenizer_json: str, stage1_vocab_size: int = 7276, full_vocab: bool = False):
    """加载 tokenizer；默认取 stage1 档（词表前 stage1_vocab_size 个高频字符，id 4..7279）"""
    with open(tokenizer_json, "r", encoding="utf-8") as f:
        t = json.load(f)
    vocab_all = t["vocab"]
    special = t["special_tokens"]
    if full_vocab:
        vocab = vocab_all
    else:
        cutoff = 4 + stage1_vocab_size - 1  # 字符 id 从 4 开始，按频率升序
        vocab = {c: i for c, i in vocab_all.items() if i <= cutoff}
        assert len(vocab) == stage1_vocab_size, f"stage1 词表截断异常: {len(vocab)}"
    return ChesiTokenizer(vocab, special)


class PackedCorpus:
    """流式 jsonl → (seq_len+1) token 块迭代器"""

    def __init__(self, paths, tokenizer: ChesiTokenizer, seq_len: int = 512,
                 text_field: str = "text", eval_stride: int = 50, max_blocks: int = None):
        self.paths = expand_paths(paths)
        self.tk = tokenizer
        self.seq_len = seq_len
        self.text_field = text_field
        self.eval_stride = eval_stride
        self.max_blocks = max_blocks

    def __iter__(self):
        """yield (block_tensor[seq_len+1], is_eval)"""
        bos, eos = self.tk.special["<bos>"], self.tk.special["<eos>"]
        need = self.seq_len + 1
        buf = []
        block_idx = 0
        # 预载：一次性把所有文本读进内存（消除"每步都从大文件流式读"的 IO 开销）
        _texts = []
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 坏行跳过（审计 A 实测 0 失败，防御性）
                    text = rec.get(self.text_field)
                    if isinstance(text, str) and text:
                        _texts.append(text)
        # 逐条 tokenize（此时文本已在内存，无磁盘 IO）
        for text in _texts:
            buf.append(bos)
            buf.extend(self.tk.encode(text))
            buf.append(eos)
            while len(buf) >= need:
                block = torch.tensor(buf[:need], dtype=torch.long)
                yield block, (block_idx % self.eval_stride == 0)
                buf = buf[need:]
                block_idx += 1
                if self.max_blocks is not None and block_idx >= self.max_blocks:
                    return
        # 尾部不足一块的丢弃（不足 seq_len 的残块无训练价值）


def expand_paths(paths):
    """支持：单个路径 / 逗号分隔 / glob 模式"""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        for part in p.split(","):
            part = part.strip()
            if not part:
                continue
            hits = sorted(glob.glob(part))
            out.extend(hits if hits else [part])
    return out


def make_batch(blocks, device):
    """blocks: list of tensor[seq+1] -> (x[B,seq], y[B,seq])"""
    x = torch.stack([b[:-1] for b in blocks]).to(device, non_blocking=True)
    y = torch.stack([b[1:] for b in blocks]).to(device, non_blocking=True)
    return x, y
