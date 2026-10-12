from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    TODO(Step4): Qwen/LLaMA 的 "rotate_half" 形式：把 head_dim 对半分为 x1, x2，
      y1 = x1*cos - x2*sin, y2 = x2*cos + x1*sin，cat 后转回原 dtype（用 float32 计算）。
    """
    raise NotImplementedError


class RotaryEmbedding(nn.Module):

    def __init__(self, head_size: int, rotary_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # TODO(Step4): 预计算 cos/sin 表
        #   inv_freq = 1 / base^(arange(0, rotary_dim, 2) / rotary_dim)
        #   freqs = outer(arange(max_position_embeddings), inv_freq)       # [max_pos, rotary_dim/2]
        #   cache = cat(cos, sin) 并 unsqueeze(1)，以便广播到 head 维      # [max_pos, 1, rotary_dim]
        #   register_buffer("cos_sin_cache", cache, persistent=False)，全程 float32
        raise NotImplementedError

    @torch.compile
    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        # TODO(Step4): 按 positions 查表得到 cos/sin，分别作用于 query / key（形状 [N, num_heads, head_dim]）
        raise NotImplementedError


@lru_cache(1)
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float, rope_scaling: tuple | None = None):
    """所有层共享同一个 RotaryEmbedding，避免每层各存一份 cos/sin 表。"""
    assert rope_scaling is None, "rope_scaling is not supported"
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
