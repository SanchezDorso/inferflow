import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, vocab]，temperatures: [B]；temperature == 0 表示贪心。
        TODO(Step5):
          贪心：argmax(logits)
          采样：probs = softmax(logits / T)，再用 Gumbel-max 等价形式 argmax(probs / E), E ~ Exp(1)
               （比 torch.multinomial 少一次 GPU 同步）
          用 torch.where 按 temperature 是否为 0 合并两种结果（注意 T=0 时不能真的除以 0）
        """
        raise NotImplementedError
