import os
import re
import glob
import json
import logging
import torch
from omegaconf import OmegaConf

from gptest.backbone.gpt import GPT
from gptest.tokenizer.tokenizer import get_tokenizer
from gptest.utils.ddp_utils import log0
from gptest.utils.utils import get_base_dir

BASE = 'base'
MID = 'mid'
SFT = 'sft'
RL = 'rl'


class Checkpoint:
    def __init__(self, source, rank):
        self.rank = rank
        model_dir = {
            BASE: "base_checkpoints",
            MID: "mid_checkpoints",
            SFT: "chatsft_checkpoints",
            RL: "chatrl_checkpoints",
        }[source]
        base_dir = get_base_dir()
        self.checkpoint_dir = os.path.join(base_dir, model_dir)
    
    def save(self, step, model_data, optimizer_data, meta_data):
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            model_path = os.path.join(self.checkpoint_dir, f'model_{step:06d}.pt')
            torch.save(model_data, model_path)
            log0(f'Saved model parameters to: {model_path}')
            meta_path = os.path.join(self.checkpoint_dir, f'meta_{step:06d}.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2)
            log0(f'Saved metadata to: {meta_path}')
        
        if optimizer_data is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            optimizer_path = os.path.join(self.checkpoint_dir, f'optim_{step:06d}_rank{self.rank:d}.pt')
            torch.save(optimizer_data, optimizer_path)
            log0(f'Saved optimizer state to: {optimizer_path}')
    
    def load(self, step, device, load_optimizer=False):
        model_path = os.path.join(self.checkpoint_dir, f'model_{step:06d}.pt')
        model_data = torch.load(model_path, map_location=device)

        optimizer_data = None
        if load_optimizer:
            optimizer_path = os.path.join(self.checkpoint_dir, f'optim_{step:06d}_rank{self.rank:d}.pt')
            optimizer_data = torch.load(optimizer_path, map_location=device)
        
        meta_path = os.path.join(self.checkpoint_dir, f'meta_{step:06d}.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        return model_data, optimizer_data, meta_data
    
    def build_model(self, step, device, is_eval=True):
        model_data, optimizer_data, meta_data = self.load(step, device, load_optimizer=False)
        if device.type == 'cpu':
            model_data = {
                k: v.float() if v.dtype == torch.bfloat16 else v
                for k, v in model_data.items() 
            }
        
        # torch compile artifact
        model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
        model_config_kwargs = meta_data["model_config"]
        model_config = OmegaConf.create(model_config_kwargs)
        log0(f'Building model with config: {model_config}')

        with torch.device('meta'):
            model = GPT(model_config)
        
        model.to_empty(device=device)
        model.init_weights()
        model.load_state_dict(model_data, strict=True, assign=True)

        if is_eval:
            model.eval()
        else:
            model.train()

        tokenizer = get_tokenizer()
        tsize = tokenizer.get_vocab_size()
        tsize_config = model_config.tokenizer.vocab_size
        assert tsize == tsize_config, f'Tokenizer vocab size ({tsize}) != config vocab size ({tsize_config})'
        return model, tokenizer, meta_data

    def find_latest_by_time(self):
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "model_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        
        latest_file = max(checkpoint_files, key=os.path.getmtime)
        step = int(os.path.basename(latest_file).split("_")[-1].split(".")[0])
        return step
    
    def load_model_from_dir(self, device, is_eval=True, step=None):
        if step is None:
            step = self.find_last_step(self.checkpoint_dir)
            log0(f"No step provided, using latest step: {step}")
        
        log0(f"Loading model from {self.checkpoint_dir} with step {step}")
        model, tokenizer, meta_data = self.build_model(step, device, is_eval=is_eval)
        return model, tokenizer, meta_data
    
    def load_source_model(self, source, *args, **kwargs):
        base_dir = get_base_dir()
        checkpoints_dir = os.path.join(base_dir, self.checkpoint_dir)
        return self.load_model_from_dir(checkpoints_dir, *args, **kwargs)
