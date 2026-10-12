import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from inferflow.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表按行切分：每个 rank 只查自己负责的词表区间，区间外置零后 all-reduce。"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # TODO(Step3): 取词表的第 tp_rank 段
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO(Step3):
          tp_size == 1：直接 F.embedding
          tp_size > 1 ：mask = x 是否落在 [vocab_start_idx, vocab_end_idx)；
                        把 x 平移到本地下标（区间外的随便映射到 0），查表后乘 mask 置零，再 all_reduce
        """
        raise NotImplementedError


class ParallelLMHead(VocabParallelEmbedding):
    """输出层同样按词表切分，各 rank 算局部 logits，gather 到 rank 0 拼接后采样。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        """
        TODO(Step3):
          1. prefill 时只需要每条序列最后一个位置：用 get_context().cu_seqlens_q[1:] - 1 取出这些行
          2. 局部 logits = F.linear(x, weight)
          3. tp_size > 1：dist.gather 到 rank 0，在最后一维 cat；非 rank 0 返回 None
        """
        raise NotImplementedError
