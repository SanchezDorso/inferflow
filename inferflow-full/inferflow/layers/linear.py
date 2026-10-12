import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(self, input_size: int, output_size: int, bias: bool = False, tp_dim: int | None = None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """按输出维切分：Y = X [W_0, W_1, ...]，每个 rank 持有 W_i，输出无需通信。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, divide(output_size, dist.get_world_size()), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        param_data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """多个列并行矩阵拼在一起（如 gate_proj + up_proj），一次 GEMM 完成。"""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        param_data.copy_(loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank])


class QKVParallelLinear(ColumnParallelLinear):
    """q/k/v 三个投影合并；按注意力头切分到各 rank（GQA 下 kv 头数少于 q 头数）。"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        assert loaded_shard_id in ("q", "k", "v")
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if loaded_shard_id == "q":
            shard_offset, shard_size = 0, q_size
        elif loaded_shard_id == "k":
            shard_offset, shard_size = q_size, kv_size
        else:
            shard_offset, shard_size = q_size + kv_size, kv_size
        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        param_data.copy_(loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank])


class RowParallelLinear(LinearBase):
    """按输入维切分：Y = sum_i X_i W_i，各 rank 计算部分和后 all-reduce。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(divide(input_size, dist.get_world_size()), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.dim() == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        param_data.copy_(loaded_weight.narrow(self.tp_dim, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
