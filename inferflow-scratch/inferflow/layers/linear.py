import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class LinearBase(nn.Module):
    """
    权重形状 [output_size, input_size]（与 F.linear 一致）。
    每个参数挂一个 weight_loader，加载 checkpoint 时由 loader 调用，负责只拷贝属于当前 rank 的分片。
    tp_dim：切分所在维度（0 = 输出维 / 列并行，1 = 输入维 / 行并行）。
    """

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
    """不切分，每个 rank 持有完整权重（示例实现）。"""

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """按输出维切分：Y = X [W_0, W_1, ...]，每个 rank 持有 W_i，输出无需通信。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, divide(output_size, dist.get_world_size()), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # TODO(Step3): 在 tp_dim 上取第 tp_rank 段（narrow），拷贝进 param.data
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(Step3)
        raise NotImplementedError


class MergedColumnParallelLinear(ColumnParallelLinear):
    """多个列并行矩阵拼在一起（如 gate_proj + up_proj），一次 GEMM 完成。"""

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        TODO(Step3): loaded_weight 是第 loaded_shard_id 个子矩阵的完整权重。
          本 rank 的 param 布局为 [子矩阵0 的第 r 片 | 子矩阵1 的第 r 片 | ...]，
          计算该子矩阵在 param 中的 offset/size，再把 loaded_weight 的第 r 片拷进去。
        """
        raise NotImplementedError


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
        """
        TODO(Step3): loaded_shard_id ∈ {"q","k","v"}。本 rank 的 param 布局为
          [本 rank 的 q 头 | 本 rank 的 k 头 | 本 rank 的 v 头]。
          注意 bias 也会走这里（1 维，同样在 dim 0 上切）。
        """
        raise NotImplementedError


class RowParallelLinear(LinearBase):
    """按输入维切分：Y = sum_i X_i W_i，各 rank 计算部分和后 all-reduce。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(divide(input_size, dist.get_world_size()), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # TODO(Step3): weight 在 dim 1 上切；bias（1 维）不切，完整拷贝
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(Step3): 局部 matmul；bias 只能加一次（仅 rank 0 加）；tp_size > 1 时 all_reduce
        raise NotImplementedError
