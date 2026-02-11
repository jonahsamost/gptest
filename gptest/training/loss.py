import torch.nn.functional as F


def cross_entropy(logits, targets, loss_reduction='mean'):
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=-1, reduction=loss_reduction
    )
    return loss
