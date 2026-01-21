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
