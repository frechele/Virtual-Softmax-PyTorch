import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class VirtualLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super(VirtualLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x: torch.Tensor, target: Union[torch.Tensor, None] = None) -> torch.Tensor:
        weight = self.weight

        output = F.linear(x, weight)

        if self.training:
            target = target.unsqueeze(1)
            W_yi = torch.gather(self.weight, 0, target.expand(
                target.size(0), self.weight.size(1)))
            
            W_virt = torch.norm(W_yi) * x / torch.norm(x)
            virt = torch.bmm(W_virt.unsqueeze(1), x.unsqueeze(-1)).squeeze(-1)

            output = torch.concat([output, virt], dim=1)

        return output
