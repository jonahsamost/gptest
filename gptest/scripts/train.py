from dotenv import load_dotenv
assert(load_dotenv())

import torch
import torch.distributed as dist
import datetime
import wandb
import argparse

from gptest.training.trainer import Trainer
from gptest.utils.utils import (
    get_config_cli, get_config, DummyWandb, check_config
)
from gptest.utils.gpu_utils import ( compute_init, autodetect_device_type )
from gptest.utils.utils import download_file_with_lock
from gptest.evals.base_eval import place_eval_bundle, EVAL_BUNDLE_URL
from gptest.utils.ddp_utils import synchronize, log0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gptest/configs/base_config.yaml')
    parser.add_argument('overrides', nargs='*')  # e.g. model.hidden_dim=1024
    args = parser.parse_args()
    config = get_config_cli(args)
    check_config(config)

    device_type = autodetect_device_type()
    ddp, device = compute_init(device_type)

    time_start = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    run_name = f'Trainer_{time_start}'
    rank0 = ddp.rank == 0
    wandb_run = (
        wandb.init(project='GPTest', name=run_name, config=dict(config))
        if rank0 and config.meta.use_wandb else DummyWandb()
    )
    if rank0:
        # download eval stuff first
        log0(f'Downloading eval bundle')
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle, use_lock=False)
    synchronize()

    trainer = Trainer(config, ddp=ddp, device=device, wandb_run=wandb_run)
    trainer.train()
    trainer.finish()


