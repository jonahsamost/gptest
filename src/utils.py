import torch
import logging
import torch.distributed as dist
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass

def setup_logging():
    logger = logging.getLogger('GPTest')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler) 
    logger.propagate = False
    return logger

logger = setup_logging()

@dataclass
class DDP:
    is_ddp: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 0


def get_config():
    return OmegaConf.load("config.yaml")




def get_base_dir():
    gptest_base = os.environ.get('GPTEST_BASE_DIR')
    if not gptest_base:
        home_dir = os.path.expanduser('~')
        gptest_base = os.path.join(home_dir, 'gptest')
    os.makedirs(gptest_base, exist_ok=True)
    return gptest_base


def is_ddp_requested():
    """
    True if launched by torchrun (env present)
    """
    return all(k in os.environ for k in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])


def get_dist_info():
    if is_ddp_requested():
        return DDP(
            is_ddp=True,
            rank = int(os.environ['RANK']),
            local_rank = int(os.environ['LOCALRANK']),
            world_size = int(os.environ['WORLD_SIZE']),
       )
    else:
        return DDP(is_ddp=False)
    

def compute_init(device_type='cuda'): # cuda | cpu | mps
    assert device_type in ['cuda', 'mps', 'cpu'], "Invalid device type atm"
    if device_type == 'cuda':
        assert torch.cuda.is_available(), "Your PyTorch installation not configured for CUDA"
    if device_type == 'mps':
        assert torch.backends.mps.is_available(), "Your Torch is not configured for MPS"
    
    torch.manual_seed(42)
    if device_type == 'cuda':
        torch.cuda.manual_seed(42)
    
    if device_type == 'cuda':
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
    ddp = get_dist_info()
    if ddp.is_ddp and device_type == 'cuda':
        device = torch.device('cuda', ddp.local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)
    
    if ddp.rank == 0:
        logger.info(f'Distributed world size: {ddp.world_size}')
    
    return ddp. device


def is_ddp_initialized():
    return dist.is_available() and dist.is_initialized()


def compute_cleanup():
    if is_ddp_initialized():
        dist.destroy_process_group()


def print0(s='', **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)


class DummyWandb:
    def __init__(self_):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    else:
        device_type = 'cpu'
    print0(f'Autodetected device type: {device_type}')
    return device_type


