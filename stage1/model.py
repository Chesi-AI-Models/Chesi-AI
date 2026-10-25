# -*- coding: utf-8 -*-
"""
澈思 Stage-1 模型：10M 级字符级 GPT（设备无关，纯 PyTorch 标准算子）

规格（NEXT.md 锁定）：6 层 / d=384 / 8 头 / seq 512 / vocab 7,280（stage1 字符 7,276 + 特殊 4）
设计约束（启智三卡适配前提，见 docs/OPENI_MULTIMODAL_PLAN.md 第 6 节）：
  - 只用标准算子：Linear/LayerNorm/Embedding/GELU/matmul/softmax，禁止 CUDA kernel 与融合注意力依赖
  - 不 import 任何第三方训练框架，PyTorch 是唯一依赖
  - FP32 为项目锁定精度（DECISION_LOG 2026-10-06）；bf16 需用户单独授权
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Pre-LN Transformer block，朴素显式因果注意力（不用 F.sdpa，保证全后端一致）"""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B,H,T,hd)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                                                  # (B,H,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class ChesiGPT(nn.Module):
    """字符级 GPT；head 与 tok embedding 权重绑定（小模型标准做法，省 2.8M 参数）"""

    def __init__(
        self,
        vocab_size: int = 7280,
        seq_len: int = 512,
        n_layer: int = 6,
        d_model: int = 384,
        n_head: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cfg = dict(
            vocab_size=vocab_size, seq_len=seq_len, n_layer=n_layer,
            d_model=d_model, n_head=n_head, dropout=dropout,
        )
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init_weights)
        # 残差投影 scaled init（GPT-2 风格：0.02/sqrt(2*n_layer)）
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.cfg["seq_len"], f"seq {T} > max {self.cfg['seq_len']}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self) -> int:
        # tied head 只计一次
        return sum(p.numel() for p in self.parameters())

    def num_params_unique(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
