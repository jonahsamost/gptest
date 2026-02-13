import torch
from collections import deque

from gptest.inference.kv import KVCache
from gptest.utils.utils import DTYPE_MAP
from gptest.inference.utils import sample_next_token


class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.complete = False
    

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = self.model.config
        self.dtype = DTYPE_MAP[self.config.meta.inference_dtype]
    
    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list) and isinstance(tokens[0], int), 'Want list of ints'
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        conf = self.config
        kv_model_kwargs = dict(
            num_heads=conf.attn.num_kv_heads,
            head_dim=conf.mlp.hidden_dim // conf.attn.num_heads,
            num_layers=conf.gpt.layers
        )
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=self.dtype
        )

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        # TODO -- add kv cache handling to model
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.config.gpt.seq_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=self.dtype
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break
            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                ...



