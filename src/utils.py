import hydra
from omegaconf import DictConfig, OmegaConf

def get_config():
    return OmegaConf.load("config.yaml")
