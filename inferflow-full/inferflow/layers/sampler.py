import os

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _gumbel_argmax_kernel(
        logits_ptr,
        temp_ptr,
        out_ptr,
        stride,
        V,
        seed,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        t = tl.load(temp_ptr + row).to(tl.float32)
        t_safe = tl.where(t == 0, 1.0, t)
        base = logits_ptr + row * stride
        best_score = tl.full((), -1.0e30, tl.float32)
        best_idx = tl.full((), 0, tl.int32)
        start = 0
        while start < V:
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            logit = tl.load(base + offs, mask=mask, other=-1.0e30).to(tl.float32)
            # E ~ Exp(1) = -log(U)。与 softmax 后 argmax(p / E) 等价，不用写出整张概率。
            u = tl.maximum(tl.rand(seed, row * V + offs), 1.0e-20)
            e = tl.maximum(-tl.log(u), 1.0e-10)
            score = logit / t_safe - tl.log(e)
            score = tl.where(t == 0, logit, score)
            score = tl.where(mask, score, -1.0e30)
            local_max = tl.max(score, axis=0)
            local_idx = tl.argmax(score, axis=0)
            take = local_max > best_score
            best_score = tl.where(take, local_max, best_score)
            best_idx = tl.where(take, start + local_idx, best_idx)
            start += BLOCK
        tl.store(out_ptr + row, best_idx.to(tl.int64))


def _gumbel_argmax(logits: torch.Tensor, temperatures: torch.Tensor, seed: int) -> torch.Tensor:
    batch, vocab = logits.shape
    out = torch.empty(batch, dtype=torch.int64, device=logits.device)
    block = 1024
    _gumbel_argmax_kernel[(batch,)](
        logits,
        temperatures,
        out,
        logits.stride(0),
        vocab,
        seed,
        block,
        num_warps=4,
    )
    return out


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self._seed = 1

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        # 全贪心时词表 softmax 会被扔掉。0.6B 的词表有 15 万维，这是解码步上的大头之一。
        if torch.all(temperatures == 0):
            return logits.argmax(dim=-1)
        if (
            triton is not None
            and logits.is_cuda
            and os.environ.get("INFERFLOW_FUSED_SAMPLE", "1") != "0"
        ):
            self._seed = (self._seed + 1) & 0x7FFFFFFF
            return _gumbel_argmax(logits, temperatures, self._seed)
        return self._sample_reference(logits, temperatures)

    @torch.compile
    def _sample_reference(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        is_greedy = temperatures == 0
        temperatures = torch.where(is_greedy, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)
        # Gumbel-max 等价形式：argmax(p / E), E ~ Exp(1)，避免 torch.multinomial 的同步开销
        sampled_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(is_greedy, greedy_tokens, sampled_tokens)
