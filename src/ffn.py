import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hd = config.mlp.hidden_dim
        psize = config.mlp.expand_size
        self.fc = nn.Linear(hd, hd * psize, bias=False)
        self.proj = nn.Linear(hd * psize, hd, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square()
        x = self.proj(x)
        return x