import torch
import torch.nn as nn

from typing import List


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels: int, filters: List[int], n_classes: int):
        super(SimpleConvNet, self).__init__()

        assert len(filters) == 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], 5, padding=2, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters[0], filters[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(filters[1]),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(filters[1], filters[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters[2], n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x
