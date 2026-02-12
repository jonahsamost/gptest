import argparse
import urllib
from pathlib import Path
import os
from filelock import FileLock
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from gptest.utils.ddp_utils import DDP, log0

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': torch.float8_e4m3fn,
}


def get_config():
    parent = Path(__file__).parent.parent
    path = os.path.join(parent, 'configs/base_config.yaml')
    return OmegaConf.load(path)


def get_config_cli(args, log=True):
    cfg = OmegaConf.load(args.config)
    # Apply CLI overrides
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    if log:
        log_config(cfg)
    return cfg


def check_config(config):
    choices = ['flops', 'params', 'none']
    assert config.meta.chinchilla in choices, f"Step count not in {choices}"


def log_config(cfg):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(OmegaConf.to_yaml(cfg, resolve=True))


def get_base_dir():
    gptest_base = os.environ.get('GPTEST_BASE_DIR')
    if not gptest_base:
        home_dir = os.path.expanduser('~')
        gptest_base = os.path.join(home_dir, 'gptest')
    os.makedirs(gptest_base, exist_ok=True)
    return gptest_base

    
class DummyWandb:
    def __init__(self_):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def download_file_with_lock(url, filename, postprocess_fn=None):
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + '.lock'

    if os.path.exists(file_path):
        return file_path
    
    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path
        
        print(f"Downloading from {url}")
        with urllib.request.urlopen(url) as response:
            content = response.read()
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        if postprocess_fn is not None:
            postprocess_fn(file_path)
    
    return file_path


def get_step_count(model: torch.nn.Module, config: DictConfig, ddp: DDP):
    """
    From: https://arxiv.org/abs/2203.15556 (Chinchilla)
    Blog: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
    """
    choice = config.meta.chinchilla
    total_batch_size = ddp.world_size * config.meta.device_batch_size * config.gpt.seq_len
    if choice == 'none':
        steps = config.meta.max_steps
        log0(f"Steps -- using user provided steps: {steps}")
        return steps
    if choice == 'params':
        total_tokens = model.params_count() * config.meta.data_param_ratio
        steps = total_tokens // total_batch_size
        log0(f"Steps -- using empircal Chinchilla rule based on parameters: {steps}")
        return steps
    if choice == 'flops':
        assert config.meta.total_flops > 0
        flops = model.estimate_flops()
        denom = flops * total_batch_size
        steps = round(config.meta.target_flops / denom)
        log0(f"Steps -- using Chinchilla scaling based on flops: {steps}")
        return steps


