import torch
import torch.nn as nn


METRICS = {
    'cross_entropy': nn.CrossEntropyLoss
}


def build_loss(name: str) -> nn.Module:
    if name in METRICS:
        return METRICS[name]

    raise ValueError('Unknown loss name')
