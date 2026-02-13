import torch
from gptest.utils.utils import DTYPE_MAP


class KVCache:
    def __init__(self, config, batch_size, device, seq_len=None, dtype=None):
        self.config = config
        self.batch_size = batch_size
        self.device = device

        self.n_layers = config.gpt.layers
        self.n_heads = config.attn.num_kv_heads
        self.head_dim = config.attn.hidden_dim // self.n_heads
        self.max_seq_len = seq_len or config.gpt.seq_len

        if dtype is None:
            dtype = DTYPE_MAP[config.meta.inference_dtype]
        self.dtype = dtype

        shape = (self.n_layers, self.batch_size, self.max_seq_len, self.n_heads, self.head_dim)
        self.k_cache = torch.zeros((shape), device=device, dtype=self.dtype)
        self.v_cache = torch.zeros((shape), device=device, dtype=self.dtype)
        """
        cache_seqlens:
            track current position in the KV cache for each batch element.
            Used by FA3 to know where to write new KV values
            Purpose:
                1. cache_seqlens[i] is number of tokens already cached for batch element i
                2. tells FA3 where to write when new tokens arrive
                3. enables variable length sequences (each batch element can ahve different cached
                lengths)
        """
        self.cache_seqlens = torch.zeros(self.batch_size, dtype=torch.int32, device=device)
    
    def reset(self):
        self.cache_seqlens.zero_()
    
    def get_pos(self):
        """Current position, assumes all batch elements at same position"""
        return self.cache_seqlens[0].item()
    
    def get_batch_pos(self, i):
        return self.cache_seqlens[i].item()
    
    def get_layer_cache(self, idx):
        return self.k_cache[idx], self.v_cache[idx]
    
    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens
    
    def prefill(self, other):
        assert self.get_pos() == 0, 'Cannot prefill a non-empty KV cache'
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len

        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
    

