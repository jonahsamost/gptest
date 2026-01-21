import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import (
    apply_rotary_emb, rms_norm
)
from src.sdpa import attention_func


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.attn.num_heads
        self.num_kv_heads = config.attn.num_kv_heads
        self.hidden_dim = config.attn.hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads

        # MQA (num_kv_heads==1), GQA (1 < num_kv_heads < num_heads)
        assert self.hidden_dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        assert self.num_kv_heads <= self.num_heads

        # why bias==False.
        # 1. doesnt hurt perf
        # 2. works better with RMSNorm (RMSNorm has no bias term)
        # 3. simplifies kv caching
        # 4. tensor parallelism simplification (no sharding biases across GPUs)
        self.c_q = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    
    def forward(self, x, cos_sin):
        B, T, HD = x.size()

        # shape: (B, T, H, D)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)

        y = attention_func(q, k, v, causal=True)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y