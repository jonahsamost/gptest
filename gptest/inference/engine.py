import torch
from collections import deque

from gptest.inference.kv import KVCache
from gptest.utils.utils import DTYPE_MAP
from gptest.inference.utils import sample_next_token, use_calculator


class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False
    

class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = self.model.config
        self.dtype = DTYPE_MAP[self.config.meta.inference_dtype]

        get_special = lambda s: self.tokenizer.encode_special(s)
        self.python_start = get_special("<|python_start|>")
        self.python_end = get_special("<|python_end|>")
        self.output_start = get_special("<|output_start|>")
        self.output_end = get_special("<|output_end|>")
        self.assistant_end = get_special("<|assistant_end|>")
        self.bos = self.tokenizer.get_bos_token_id()
    
    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list) and isinstance(tokens[0], int), 'Want list of ints'
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        kv_cache_prefill = KVCache(
            config=self.config,
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
            config=self.config,
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
                # a forced token is a token injected into the model without sampling for tool use
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == self.assistant_end or next_token == self.bos:
                    state.completed = True
                
                if next_token == self.python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == self.python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(self.output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(self.output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)
            
            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == self.assistant_end or token == self.bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)

            if all(completed):
                break
        return results, masks
                    
