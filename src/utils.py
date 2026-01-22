import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass


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
    os.markdirs(gptest_base, exist_ok=True)
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
    

