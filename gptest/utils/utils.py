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


def get_peak_flops() -> float:
    device_name = torch.cuda.get_device_name(0)
    name = device_name.lower()

    # --- NVIDIA Blackwell ---
    if "gb200" in name or "grace blackwell" in name:
        return 2.5e15
    if "b200" in name:
        return 2.25e15
    if "b100" in name:
        return 1.8e15

    # --- NVIDIA Hopper (H100/H200/H800) ---
    if "h200" in name:
        if "nvl" in name or "pcie" in name:
            return 836e12
        return 989e12  # H200 SXM
    if "h100" in name:
        if "nvl" in name:
            return 835e12
        if "pcie" in name:
            return 756e12
        return 989e12  # H100 SXM
    if "h800" in name:
        if "nvl" in name:
            return 989e12
        return 756e12  # H800 PCIe

    # --- NVIDIA Ampere data center ---
    if "a100" in name or "a800" in name:
        return 312e12
    if "a40" in name:
        return 149.7e12
    if "a30" in name:
        return 165e12

    # --- NVIDIA Ada data center ---
    if "l40s" in name or "l40-s" in name or "l40 s" in name:
        return 362e12
    if "l4" in name:
        return 121e12

    # --- AMD CDNA accelerators ---
    if "mi355" in name:
        return 2.5e15
    if "mi325" in name or "mi300x" in name:
        return 1.3074e15
    if "mi300a" in name:
        return 980.6e12
    if "mi250x" in name:
        return 383e12
    if "mi250" in name:
        return 362.1e12

    # --- Intel ---
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # --- Consumer RTX (for hobbyists) ---
    if "5090" in name:
        return 209.5e12
    if "4090" in name:
        return 165.2e12
    if "3090" in name:
        return 71e12

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    log0(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')

