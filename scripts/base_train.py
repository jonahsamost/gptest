"""
train the model
python3 -m scripts.base_train

or dist with:
torchrun --nproc_per_node=8 -m scripts.base_train

"""

import os
import token
# segments can grow dynamically
# reduces fragmentation by allowing existing segments to expand
os.environ['PYTORCH_ALLOC_CONF'] = "expandable_segments:True"
import time
import math
from contextlib import nullcontext
import wandb
from datetime import datetime
import torch

from src.checkpoint_manager import load_checkpoint, save_checkpoint
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
from src.common import DTYPE_MAP, get_lr_multipler, get_muon_momentum


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

resume = config.meta.resume_from_step != -1
if resume:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

dataloader_resume_state_dict = None if not resume else meta_data['dataloader_state_dict']
train_loader = tokenizing_dist_data_loader_with_state_bos(
    tokenizer, config.meta.device_batch_size, device.gpt.seq_len, split='train',
    device=device, resume_state_dict=dataloader_resume_state_dict
)

build_val_loader = lambda: tokenizing_dist_data_loader_bos(
    tokenizer, config.meta.device_batch_size, device.gpt.seq_len, split='train', device=device
)
x, y, dataloader_state_dict = next(train_loader)

# TODO how to determine num iterations?
num_iterations = 100_000
num_flops_per_token = 1 # TODO
get_weight_decay = lambda it: config.meta.weight_decay * (1 - it / num_iterations)

if not resume:
    step = 0
    val_bpb = None
    min_val_bpb = float('inf')
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]


while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * config.meta.total_batch_size * step
    
    results = {}
    do_meter = last_step or (step > 0 and step % config.meta.core_metric_every == 0)
    if config.meta.core_metric_every > 0 and do_meter:
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=config.meta.core_metric_max_per_task)
        cm = results['core_metric']
        print0(f"Step {step:05d} | core metric {cm:.4f}")
        wandb_run({
            'step': step,
            'total_training_flops': flops_so_far,
            'core_metric': cm,
            'centered_results': results['centered_results'],
        })
        model.train()
        
    do_sample = last_step or (step > 0 and step % config.meta.sample_every == 0)
    if config.meta.sample_every > 0 and master_process and do_sample:
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend='<|bos|>')
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperate=0)
            print0(tokenizer.decode(sample[0]))
        model.train()
    
    do_save = (
        step > 0 and step != config.meta.resume_from_step and
        config.meta.save_every > 0 and step % config.meta.save_every == 0
    )
    if last_step and do_save:
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": config,
                "device_batch_size": config.meta.device_batch_size,
                "max_seq_len": config.gpt.seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp.rank
        )

    if last_step:
        break 

    synchronize()
    t0 = time.time()
    grad_accum_steps = config.meta.grad_accum_steps
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # logging
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    
    lrm = get_lr_multipler(config, step, num_iterations)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in muon_optimizer.param_groups:
        group['momentum'] = muon_momentum
        group['weight_decay'] = muon_weight_decay
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    t1 = time.time()
    dt = t1 - t0

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(config.meta.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * config.meta.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp.world_size)
    if step > 10:
        total_training_time += dt
    
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    step += 1


print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

wandb_run.finish()
compute_cleanup()