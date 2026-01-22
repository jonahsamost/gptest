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
from contextlib import nullcontext
import wandb
import torch

from src.gpt import GPT
from src.dataloader import (
    tokenizing_dist_data_loader_bos,
    tokenizing_dist_data_loader_with_state_bos
)
from src.utils import (
    compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, 
)
from src.tokenizer import ( get_tokenizer, get_token_bytes )