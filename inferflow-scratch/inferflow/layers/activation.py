import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(Step4): x = [gate | up]（最后一维对半分），返回 silu(gate) * up
        raise NotImplementedError
