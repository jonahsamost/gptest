import torch
import torch.nn as nn
import torch.nn.functional as F


def _sdpa_attention(q, k, v, enable_gqa):
    # q, k, v shapes == (B, H, T, D)
    Tq = q.size(2)
    Tk = k.size(2)

    # training or prefill
    if Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    
    # decode step
    if Tq == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    
    raise ValueError("Chunked inference not implemented")


def attention_func(q, k, v, causal=False):
    # shape: (B, T, H, D) -> (B, H, T, D)
    # .transpose will:
    #   swap stride metadata between dimensions
    #   result usually non-contigous
    #   O(1) cost
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    enable_gqa = q.size(1) != k.size(1)
    output = _sdpa_attention(q, k, v, enable_gqa)
    # output shape: (B, T, H, D)
    return output.transpose(1, 2) 


def attention_with_kv(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, causal=False):
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()

    if k is not None and v is not None:
        k_cache[:, pos:pos + T_new, :, :] = k
        v_cache[:, pos:pos + T_new, :, :] = v
    
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, enable_gqa)
    return y_sdpa.transpose(1, 2)
