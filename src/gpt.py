import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ffn import BaseMLP
from src.attention import CausalSelfAttention
from src.common import (
    rms_norm, precompute_rotary_embeddings
)


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = BaseMLP(config)
    
    def forward(self, x, cos_sin):
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = self.config.tokenizer.vocab_size
        hidden_dim = self.config.mlp.hidden_dim
        layers = self.config.gpt.layers

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, hidden_dim),
            'h': nn.ModuleList([
                TransformerBlock(config, idx) for idx in range(layers)
            ])
        })
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.rotary_seq_len = self.config.gpt.seq_len * 10
        head_dim = hidden_dim // config.attn.num_heads
        cos, sin = precompute_rotary_embeddings(
            config, self.rotary_seq_len, head_dim
        )
        # not in model.parameters(), not updated by optimizer,
        # dont recieve gradients, not in state_dict() (persistent=False)
        # get moved with .to()
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)