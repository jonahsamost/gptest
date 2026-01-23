"""
train the model
python3 -m scripts.base_train

or dist with:
torchrun --nproc_per_node=8 -m scripts.base_train

"""

import os
# segments can grow dynamically
# reduces fragmentation by allowing existing segments to expand
os.environ['PYTORCH_ALLOC_CONF'] = "expandable_segments:True"
import time
import math
from contextlib import nullcontext
import wandb
from datetime import datetime
import torch

from src.checkpoint_manager import load_checkpoint
from src.gpt import GPT
from src.dataloader import (
    tokenizing_dist_data_loader_bos,
    tokenizing_dist_data_loader_with_state_bos
)
from src.utils import (
    compute_init, compute_cleanup, print0, log0,
    get_config_cli, DummyWandb, get_base_dir, autodetect_device_type
)
from src.tokenizer import ( get_tokenizer, get_token_bytes )
from src.common import DTYPE_MAP


config = get_config_cli(log=True)
device_type = autodetect_device_type()
ddp, device = compute_init(device_type)
master_process = ddp.rank == 0
autocast_ctx = torch.amp.autocast(
    device_type=device_type, dtype=DTYPE_MAP[config.meta.train_dtype]
) if device_type == 'cuda' else nullcontext()
synchronize = torch.cuda.synchronize if device_type == 'cuda' else lambda: 0
get_max_memory = torch.cuda.max_memory_allocated if device_type == 'cuda' else lambda: 0

use_dummy_wandb = not config.meta.use_wandb or not master_process
run_name = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project='GPTest', name=run_name, config=config)

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size() 
assert vocab_size == config.tokenizer.vocab_size, 'Vocab size is different than expected'
token_bytes = get_token_bytes(device=device)

# structurally creates device without allocating memory/storage
# params exist on as shape/dtype/metadata
with torch.device('meta'):
    model = GPT(config)

model.to_empty(device=device) # storage on device, uninit'd data
model.init_weights() 

base_dir = get_base_dir()
checkpoint_dir = os.path.join(base_dir, 'base_checkpoints')
resume_step = config.meta.resume_from_step
if resume_step > 0:
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, resume_step, device, load_optimizer=True, rank=ddp.rank
    )
    # check keys match and replace module params with tensors from state dict
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

orig_model = model
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
log0(f'Number of parameters: {num_params}')

# TODO calculate flops

batch_lr_scale = math.sqrt(
    config.meta.device_batch_size * ddp.world_size * config.meta.grad_accum_steps    
)
adam_betas = (config.meta.adam_beta1, config.meta.adam_beta2)
optimizers = model.setup_optimizers(
    unembedding_lr=config.gpt.unembedding_lr * batch_lr_scale,
    embedding_lr=config.gpt.embedding_lr * batch_lr_scale,
    matrix_lr=config.gpt.matrix_lr * batch_lr_scale,
    weight_decay=config.meta.weight_decay,
    adam_betas=adam_betas,
    scalar_lr=config.gpt.scalar_lr * batch_lr_scale
)
adamw_optimizer, muon_optimizer = optimizers