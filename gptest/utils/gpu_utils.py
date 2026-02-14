import torch
import torch.distributed as dist
from gptest.utils.logger import logger
from gptest.utils.ddp_utils import get_dist_info, print0


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
        torch.backends.cuda.matmul.allow_tf32 = True
    
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
    
    return ddp, device


def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device_type = 'mps'
    else:
        device_type = 'cpu'
    print0(f'Autodetected device type: {device_type}')
    return device_type
