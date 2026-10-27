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
    def _rms_norm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        res_out_ptr,
        stride_x,
        stride_r,
        stride_out,
        stride_res,
        n_cols,
        eps,
        HAS_RESIDUAL: tl.constexpr,
        ROUND_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
            x = x + residual
            # 新 residual 是相加后的 bf16，方差仍用相加后的 fp32，和原来的 compile 区域一致。
            tl.store(res_out_ptr + row * stride_res + offs, x, mask=mask)
        var = tl.sum(x * x, axis=0) / n_cols
        x = x * tl.rsqrt(var + eps)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if ROUND_BF16:
            y = x.to(tl.bfloat16) * w.to(tl.bfloat16)
        else:
            y = x * w
        tl.store(out_ptr + row * stride_out + offs, y, mask=mask)


def _triton_rms(x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    res_out = torch.empty_like(x) if residual is not None else None
    block = triton.next_power_of_2(n_cols)
    _rms_norm_kernel[(n_rows,)](
        x,
        residual if residual is not None else x,
        weight,
        out,
        res_out if res_out is not None else out,
        x.stride(0),
        residual.stride(0) if residual is not None else 0,
        out.stride(0),
        res_out.stride(0) if res_out is not None else 0,
        n_cols,
        eps,
        residual is not None,
        x.dtype == torch.bfloat16,
        block,
    )
    if residual is None:
        return out
    return out, res_out


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(orig_dtype) * self.weight

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """融合 residual add + RMSNorm，返回 (norm 结果, 新 residual)。"""
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(orig_dtype) * self.weight, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if (
            triton is not None
            and x.is_cuda
            and x.stride(-1) == 1
            and os.environ.get("INFERFLOW_TRITON_NORM_ACT", "1") != "0"
        ):
            return _triton_rms(x, self.weight, self.eps, residual)
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
