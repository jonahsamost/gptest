import math
import os
import time
from pathlib import Path
from typing import Any
from contextlib import nullcontext

from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
import wandb
from wandb.sdk import wandb_run

from gptest.tokenizer.tokenizer import ( get_tokenizer, get_token_bytes )
from gptest.utils.utils import ( DTYPE_MAP, get_base_dir)
from gptest.utils.ddp_utils import (log0, print0, DDP)
from gptest.backbone.gpt import GPT
from gptest.data.dataloader import (
    tokenizing_dist_data_loader_with_state_bos,
    tokenizing_dist_data_loader_bos
)
from gptest.evals.bpb_loss_eval import evaluate_bpb
from gptest.evals.base_eval import evaluate_model


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        ddp: DDP,
        device: torch.device,
        wandb_run: Any | None = None,    
    ):
        self.ddp = ddp
        self.device = device
        self.device_type = device.type
        self.rank0 = self.ddp.rank == 0
        self.iteration = 0
        self.config = config
        self.wandb_run = wandb_run

        self.autocast_ctx = torch.amp.autocast(
            device_type=self.device_type, dtype=DTYPE_MAP[config.meta.train_dtype]
        ) if self.device_type == 'cuda' else nullcontext()
        self.synchronize = torch.cuda.synchronize if self.device_type == 'cuda' else lambda: 0

        self.base_dir = get_base_dir()

        self.tokenizer = get_tokenizer()
        vocab_size = self.tokenizer.get_vocab_size() 
        assert vocab_size == config.tokenizer.vocab_size, 'Vocab size is different than expected'
        self.token_bytes = get_token_bytes(device=self.device)

        # structurally creates device without allocating memory/storage
        # params exist on as shape/dtype/metadata
        with torch.device('meta'):
            model = GPT(config)

        model.to_empty(device=self.device) # storage on device, uninit'd data
        model.init_weights() 
        self.uncompiled_model = model
        self.model = torch.compile(model, dynamic=False)

        num_params = sum(p.numel() for p in self.model.parameters())
        log0(f'Number of parameters: {num_params}')
        self.optimizers = self.model.setup_optimizers(
            config, self.ddp
        )

        log0(f"Creating dataloaders")
        self.train_loader = tokenizing_dist_data_loader_with_state_bos(
            self.tokenizer, config.meta.device_batch_size, config.gpt.seq_len,
            split='train', device=self.device
        )

        self.build_val_loader = lambda: tokenizing_dist_data_loader_bos(
            self.tokenizer, config.meta.device_batch_size, config.gpt.seq_len,
            split='val', device=self.device
        )
        x, y, _ = next(self.train_loader)
        log0("Trainer inited")
    
    def eval_bpb(self):
        ee = self.config.meta.eval_every
        if not ee or not self.iteration:
            return
        good_iter = self.iteration % ee == 0
        is_end = self.iteration == self.config.meta.max_steps - 1
        if not (good_iter or is_end):
            return
        self.model.eval()
        val_loader = self.build_val_loader()
        eval_tokens = 20 * 524288
        eval_steps = eval_tokens // (
            self.config.meta.device_batch_size * self.config.gpt.seq_len * self.ddp.world_size
        )
        with self.autocast_ctx:
            val_bpb = evaluate_bpb(self.model, val_loader, eval_steps, self.token_bytes)
        log0(f"Step: {self.iteration:05d} | Validation bpb: {val_bpb:.6f}")
        self.wandb_run.log({
            "step": self.iteration,
            "val/bpb": val_bpb,
        })
        self.model.train()
    
    def eval_core(self):
        cme = self.config.meta.core_metric_every
        if not cme or not self.iteration:
            return
        good_iter = self.iteration % cme == 0
        is_end = self.iteration == self.config.meta.max_steps - 1
        if not (good_iter or is_end):
            return
        self.model.eval()
        with self.autocast_ctx:
            results = evaluate_model(
                self.uncompiled_model, self.tokenizer, self.device,
                max_per_task=self.meta.core_metric_max_per_task
            )
        cm = results['core_metric']
        log0(f"Step: {self.iteration:05d} | Validation bpb: {cm:.4f}")
        wandb_run.log({
            "step": self.iteration,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        self.model.train()
    
    def train(self):
        
        while self.iteration < self.config.meta.max_steps:
            self.eval_bpb()
            self.eval_core()
            
            self.iteration += 1
    
    def generate(self):
        ...


        
