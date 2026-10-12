import torch
from torch import nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight，内部用 float32 计算。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(Step4): 注意不要原地修改输入 x（调用方可能还把它当 residual 用）
        raise NotImplementedError

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO(Step4): 融合 residual add + RMSNorm：h = x + residual；返回 (norm(h), h)
        raise NotImplementedError

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
