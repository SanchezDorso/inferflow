import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        is_greedy = temperatures == 0
        temperatures = torch.where(is_greedy, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)
        # Gumbel-max 等价形式：argmax(p / E), E ~ Exp(1)，避免 torch.multinomial 的同步开销
        sampled_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(is_greedy, greedy_tokens, sampled_tokens)
