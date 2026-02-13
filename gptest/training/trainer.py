import time
from typing import Any
from contextlib import nullcontext

from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

from gptest.tokenizer.tokenizer import ( get_tokenizer, get_token_bytes )
from gptest.utils.utils import ( DTYPE_MAP, get_base_dir, get_peak_flops, get_step_count)
from gptest.training.utils import get_lr_multiplier
from gptest.utils.ddp_utils import (log0, DDP, compute_cleanup)
from gptest.backbone.gpt import GPT
from gptest.data.dataloader import (
    tokenizing_dist_data_loader_with_state_bos,
    tokenizing_dist_data_loader_bos
)
from gptest.evals.bpb_loss_eval import evaluate_bpb
from gptest.evals.base_eval import evaluate_model
from gptest.training.loss import cross_entropy
from gptest.utils.checkpoint import Checkpoint, BASE


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
        self.grad_accum_steps = self.config.meta.grad_accum_steps
        self.smooth_train_loss = 0.0
        self.total_training_time = 0.0
        self.min_val_bpb = 100_000_000
        self.val_bpb = 100_000_000
        self.total_batch_size = (
            ddp.world_size * config.meta.device_batch_size * config.gpt.seq_len * config.meta.grad_accum_steps
        )
        self.lrm = 0.0
        self.checkpoint = Checkpoint(BASE, ddp.rank)

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

        self.num_iterations = int(get_step_count(model, config, ddp))
        self.flops_per_token = self.model.estimate_flops()
        self.gpu_peak_flops = get_peak_flops()

        num_params = self.model.params_count()
        log0(f'Number of model parameters: {num_params}')
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
        self.x, self.y, self.pqd = next(self.train_loader)
        log0("Trainer inited")
    
    def eval_bpb(self):
        ee = self.config.meta.eval_every
        if not ee or not self.iteration:
            return
        good_iter = self.iteration % ee == 0
        is_end = self.iteration == self.num_iterations - 1
        if not (good_iter or is_end):
            return
        self.model.eval()
        val_loader = self.build_val_loader()
        eval_steps = 20
        log0(f"Eval bpb for {eval_steps} steps")
        with self.autocast_ctx:
            self.val_bpb = evaluate_bpb(self.model, val_loader, eval_steps, self.token_bytes)
        self.min_val_bpb = min(self.min_val_bpb, self.val_bpb)
        log0(f"Step: {self.iteration:05d} | Validation bpb: {self.val_bpb:.6f}")
        self.wandb_run.log({
            "step": self.iteration,
            "val/bpb": self.val_bpb,
        })
        self.model.train()
    
    def eval_core(self):
        cme = self.config.meta.core_metric_every
        if not cme or not self.iteration:
            return
        good_iter = self.iteration % cme == 0
        is_end = self.iteration == self.num_iterations - 1
        if not (good_iter or is_end):
            return
        self.model.eval()
        log0("Eval core!")
        with self.autocast_ctx:
            results = evaluate_model(
                self.uncompiled_model, self.tokenizer, self.device,
                max_per_task=self.config.meta.core_metric_max_per_task
            )
        cm = results['core_metric']
        log0(f"Step: {self.iteration:05d} | Validation bpb: {cm:.4f}")
        self.wandb_run.log({
            "step": self.iteration,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        self.model.train()
    
    def sample_model(self):
        se = self.config.meta.sample_every
        if not se or not self.iteration or self.ddp.rank != 0:
            return
        good_iter = self.iteration % se == 0
        is_end = self.iteration == self.num_iterations - 1
        if not (good_iter or is_end):
            return
        self.model.eval()
        log0("Sampling model!")
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        for prompt in prompts:
            gen_tokens = []
            tokens = self.tokenizer(prompt, prepend='<|bos|>')
            stream = self.model.generate(tokens)
            with self.autocast_ctx:
                for token in stream:
                    gen_tokens.append(token)
                response = self.tokenizer.decode(gen_tokens)
                log0(f'Prompt: {prompt}\nResponse: {response}\n\n')
        self.model.train()
    
    def step(self):
        self.synchronize()
        t0 = time.time()
        for micro_step in range(self.grad_accum_steps):
            with self.autocast_ctx:
                logits = self.model(self.x)
                loss = cross_entropy(logits, self.y)
            train_loss = loss.detach()
            loss /= self.grad_accum_steps
            loss.backward()
            self.x, self.y, self.pqd = next(self.train_loader)
        
        self.lrm = get_lr_multiplier(self.config, self.iteration, self.num_iterations)
        for opt in self.optimizers:
            for group in opt.param_groups:
                group['lr'] = group['initial_lr'] * self.lrm

        for opt in self.optimizers:
            opt.step()

        self.model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        self.synchronize()
        t1 = time.time()
        dt = t1 - t0
        return train_loss_f, dt
    
    def log(self, train_loss_f, dt):
        ema_beta = .9
        self.smooth_train_loss = ema_beta * self.smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = self.smooth_train_loss / (1 - ema_beta**(self.iteration + 1))
        pct_done = 100 * self.iteration / self.num_iterations
        tok_per_sec = int(self.total_batch_size / dt)

        steps_done = self.iteration - 10
        eta_str = f''
        if steps_done > 0:
            self.total_training_time += dt
            avg_time_per_step = self.total_training_time / steps_done
            remain = self.num_iterations - self.iteration
            eta_seconds = remain * avg_time_per_step
            eta_str = f' | eta: {eta_seconds/60:.1f}m'
        
        log0(
            f"step: {self.iteration:05d} / {self.num_iterations:05d} ({pct_done:.2f}%) | "
            f"loss: {debiased_smooth_loss:.6f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
            f"total time: {self.total_training_time/60:.2f}m{eta_str}"
        )

        flops_per_step = self.flops_per_token * self.total_batch_size
        flops_so_far = flops_per_step * self.iteration 
        flops_per_sec = flops_per_step / dt
        mfu = 100 * flops_per_sec / (self.gpu_peak_flops * self.ddp.world_size)

        is_end = self.iteration == self.num_iterations - 1
        is_good = self.iteration and self.iteration % self.config.meta.wandb_log_steps == 0
        if is_good or is_end:
            self.wandb_run.log({
                "step": self.iteration,
                "total_training_flops": flops_so_far,
                "total_training_time": self.total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": self.lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/epoch": self.pqd.epoch,
            })
        
        do_ckpt = self.iteration and (self.iteration % self.config.meta.checkpoint_every == 0)
        if is_end or do_ckpt:
            self.save_checkpoint()
        
    def save_checkpoint(self):
        log0(f'Saving checkpoint at step: {self.iteration}')
        meta_data = {
            "step": self.iteration,
            "val_bpb": self.val_bpb,
            "model_config": OmegaConf.to_container(self.config, resolve=True),
            "dataloader_state_dict": self.pqd.to_dict(),
            "loop_state": {
                "min_val_bpb": self.min_val_bpb,
                "smooth_train_loss": self.smooth_train_loss,
                "total_training_time": self.total_training_time,
            },
        }
        self.checkpoint.save(
            self.iteration,
            self.uncompiled_model.state_dict(),
            [opt.state_dict() for opt in self.optimizers],
            meta_data
        )
    
    def finish(self):
        self.wandb_run.finish()
        compute_cleanup()
    
    def train(self):
        while self.iteration < self.num_iterations:
            self.eval_bpb()
            self.eval_core()
            self.sample_model()
            loss, dt = self.step()
            self.log(loss, dt)
            self.iteration += 1
        
