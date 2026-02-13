from contextlib import nullcontext
import torch
import time

from gptest.utils.gpu_utils import autodetect_device_type, compute_init
from gptest.utils.checkpoint import Checkpoint, BASE
from gptest.inference.engine import Engine


if __name__ == '__main__':
    device_type = autodetect_device_type()
    ddp, device = compute_init()
    autocast_ctx = torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16
    ) if device_type == 'cuda' else nullcontext()
    
    checkpoint = Checkpoint(BASE, ddp.rank)
    model, tokenizer, meta = checkpoint.load_model_from_dir(device)
    bos_token_id = tokenizer.get_bos_token_id()
    kwargs = dict(max_tokens=64, temperature=0.0)    
    prompt_tokens = tokenizer.encode('The chemical formula of water is', prepend=bos_token_id)

    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end='', flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()

    print(f'Reference time: {t1 - t0:.2f}s')
    reference_ids = generated_tokens

    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0]
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end='', flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f'Engine time: {t1 - t0:.2f}s')

    for i in range(len(reference_ids)):
        ri = reference_ids[i]
        gt = generated_tokens[i]
        if ri != gt:
            print(f'Mismatch at {i}, reference: {ri}, engine: {gt}')
            break
    print(f'All matched: {reference_ids == generated_tokens}')
