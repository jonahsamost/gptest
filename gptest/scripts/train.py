from dotenv import load_dotenv
assert(load_dotenv())

import torch
import torch.distributed as dist
import datetime
import wandb
import argparse

from gptest.training.trainer import Trainer
from gptest.utils.utils import get_config_cli, get_config, DummyWandb
from gptest.utils.gpu_utils import ( compute_init, autodetect_device_type )
from gptest.utils.ddp_utils import compute_cleanup


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gptest/configs/base_config.yaml')
    parser.add_argument('overrides', nargs='*')  # e.g. model.hidden_dim=1024
    args = parser.parse_args()
    config = get_config_cli(args)

    device_type = autodetect_device_type()
    ddp, device = compute_init(device_type)

    time_start = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    run_name = f'Trainer_{time_start}'
    rank0 = ddp.rank == 0
    # wandb_run = (
    #     wandb.init(project='GPTest', name=run_name, config=dict(config))
    #     if rank0 else DummyWandb()
    # )
    wandb_run = DummyWandb()
    trainer = Trainer(config, ddp=ddp, device=device, wandb_run=wandb_run)
    trainer.train()

    if wandb_run:
        wandb_run.finish()
    
    compute_cleanup()

