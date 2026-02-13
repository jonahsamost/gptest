import torch.distributed as dist
import os
from dataclasses import dataclass
from gptest.utils.logger import logger


@dataclass
class DDP:
    is_ddp: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1


def is_ddp_requested():
    """
    True if launched by torchrun (env present)
    """
    return all(k in os.environ for k in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])


def synchronize():
    if is_ddp_initialized():
        dist.barrier()


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


def is_ddp_initialized():
    return dist.is_available() and dist.is_initialized()


def compute_cleanup():
    if is_ddp_initialized():
        dist.destroy_process_group()


def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)


def print0(s='', **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)