import torch
import torch.nn as nn
import torch.nn.functional as F

from gptest.backbone.common import (
    apply_rotary_emb, rms_norm
)
from gptest.backbone.sdpa import attention_func, attention_with_kv


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.attn.num_heads
        self.num_kv_heads = config.attn.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.hidden_dim = config.attn.hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads

        self.gate_headwise = config.attn.gate_headwise
        self.gate_elementwise = config.attn.gate_elementwise
        self.use_og_resformer = config.meta.use_og_resformer

        # MQA (num_kv_heads==1), GQA (1 < num_kv_heads < num_heads)
        assert self.hidden_dim % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0
        assert self.num_kv_heads <= self.num_heads

        # why bias==False.
        # 1. doesnt hurt perf
        # 2. works better with RMSNorm (RMSNorm has no bias term)
        # 3. simplifies kv caching
        # 4. tensor parallelism simplification (no sharding biases across GPUs)

        # Gated attention: https://arxiv.org/pdf/2505.06708
        if self.gate_headwise:
            self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim + self.num_heads, bias=False)
        elif self.gate_elementwise:
            self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim * 2, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=False)

        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
    
    def forward(self, x, cos_sin, kv_cache=None, **kwargs):
        B, T, HD = x.size()

        # shape: (B, T, H, D)
        q = self.q_proj(x)
        if self.gate_headwise:
            q = q.view(B, T, self.num_kv_heads, -1)
            # split into (B, T, kv_heads, head_dim * kv_groups) and (B, T, kv_heads, kv_groups)
            q, gate_score = torch.split(q, [self.head_dim * self.num_kv_groups, self.num_kv_groups], dim=-1)
            # shape: (B, T, kv_heads * kv_groups, 1)
            # i.e. each head in each group gets its own score; all elements in same head get same score
            gate_score = gate_score.reshape(B, T, -1, 1)
            # q stays as shape (B, T, num_heads, head_dim)
            q = q.reshape(B, T, -1, self.head_dim)
        elif self.gate_elementwise:
            q = q.view(B, T, self.num_kv_heads, -1)
            q, gate_score = torch.split(q, [self.head_dim * self.num_kv_groups] * 2, dim=-1)
            # each element across all heads gets own score
            gate_score = gate_score.reshape(B, T, -1, self.head_dim)
            q = q.reshape(B, T, -1, self.head_dim)
        else:
            q = q.view(B, T, self.num_heads, self.head_dim)

        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # ResFormer: V_n = V_n * lambda_2 + lambda_1 * V_0
        if self.use_og_resformer:
            if self.layer_idx == 0:
                kwargs['v0'] = v
            else:
                v0 = kwargs.get('v0', None)
                resf_l1 = kwargs.get('resformer_lambda_1', None)
                resf_l2 = kwargs.get('resformer_lambda_2', None)
                if v0 and resf_l1 is not None and resf_l2 is not None:
                    v = v * resf_l2[self.layer_idx] + v0 * resf_l1[self.layer_idx]

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        
        # output shape is (B, T, H, D)
        if kv_cache is None:
            y = attention_func(q, k, v, causal=True)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = attention_with_kv(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True
            )
            # only advance cache is last layer
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        if self.gate_elementwise or self.gate_headwise:
            y = y * torch.sigmoid(gate_score)

        y = y.contiguous().view(B, T, -1)
        y = self.o_proj(y)
        return y