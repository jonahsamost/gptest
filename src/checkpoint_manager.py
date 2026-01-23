import json
import os
import re
import glob
import torch

from src.utils import ( get_base_dir, get_config, logger )
from src.gpt import GPT
from src.tokenizer import get_tokenizer



def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f'model_{step:06d}.pt')
        torch.save(model_data, model_path)
        logger.info(f'Saved model parameters to: {model_path}')
        meta_path = os.path.join(checkpoint_dir, f'meta_{step:06d}.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f'Saved metadata to {meta_path}')
    
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, exist_ok=True)
        torch.save(optimizer_data, optimizer_path)
        logger.info(f'Saved optimizer state to: {optimizer_path}')


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    model_path = os.path.join(checkpoint_dir, f'model_{step:06d}.pt')
    model_data = torch.load(model_path, map_location=device)

    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f'optim_{step:06d}_rank{rank:d}.pt')
        optimizer_data = torch.load(optimizer_path, map_location=device)
    
    meta_path = os.path.join(checkpoint_dir, map_location=device)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data
