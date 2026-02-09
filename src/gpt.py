from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from scripts.base_train import muon_optimizer
from src.ffn import BaseMLP
from src.attention import CausalSelfAttention
from src.common import (
    rms_norm, precompute_rotary_embeddings
)
from src.utils import get_dist_info, log0
from src.adamw import DistAdamW
from src.muon import Muon, DistMuon


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = BaseMLP(config)
    
    def forward(self, x, cos_sin):
        # TODO pre-norm vs post-norm ?
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.softcap = 15
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

        # shape: (B, T, H, D) 
        cos, sin = precompute_rotary_embeddings(
            config, self.rotary_seq_len, head_dim
        )
        self.cos, self.sin = cos, sin
        # not in model.parameters(), not updated by optimizer,
        # dont recieve gradients, not in state_dict() (persistent=False)
        # get moved with .to()
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        # TODO add residual lambdas, x0 lambdas, and value embs (ResFormer)
    
    def forward(self, inputs, loss_reduction='mean'):
        B, T = inputs.size()

        assert T <= self.cos.size(1), f'Sequence length larger than rotary embedding cache: {T} > {self.cos.size(1)}'
        assert inputs.device == self.cos.device, f'Rotary embeddings and inputs are on different devices: {inputs.device} != {self.cos.device}'

        # current sequence length
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(inputs) # (B, T, H)
        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin)
        x = rms_norm(x)

        logits = self.lm_head(x) # shape: (B, T, VS)
        logits = logits.float()
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits
    
    @torch.no_grad()
    def init_weights(self):
        """
        how weights get initialized:

        wte (embedding layer):      normal, std=1.0
        lm_head                     normal, std=.001
        for each block:
            attn.c_q                uniform, std=1/sqrt(hidden)
            attn.c_k                uniform, std=1/sqrt(hidden)
            attn.c_v                uniform, std=1/sqrt(hidden)
            attn.c_proj             zeros
            mlp.c_fc                uniform, std=1/sqrt(hidden)
            mlp.c_proj              zeros
        
        why we want uniform initialization:
            - outliers matter a lot in attention. a single large weight will dominate dot products
            and create spiky attention early. normal dist can have huge outliers.
            - residual layers amplify variance layer by layer. with uniform the worst case is bounded
        """

        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.transformer.lm_head.weight, mean=0.0, std=0.001)

        # uniform init with sqrt(3) * std == same std as normal
        hidden_size = self.config.mlp.hidden_dim
        s = 3**0.5 * hidden_size**-.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.proj.weight)
        
        head_dim = hidden_size // self.config.attn.num_heads
        cos, sin = precompute_rotary_embeddings(
            self.config, self.rotary_seq_len, head_dim
        )
        self.cos, self.sin = cos, sin

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.mlp.hidden_dim
        ddp = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        log0(f'Scaling the LR for AdamW parameters => 1 / sqrt({model_dim} / 768) => {dmodel_lr_scale:.6f}')
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale)
        ]
        # hardcode weight decay at 0 for adam, only use wd for muon
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        AdamWFactory = DistAdamW if ddp.is_ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)

        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp.is_ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group['initial_lr'] = group['lr']
        return optimizers
