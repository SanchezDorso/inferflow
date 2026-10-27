import os

import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _silu_and_mul_kernel(
        x_ptr,
        out_ptr,
        stride_x,
        stride_out,
        d,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1)
        offs = col * BLOCK + tl.arange(0, BLOCK)
        mask = offs < d
        gate = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + row * stride_x + d + offs, mask=mask, other=0.0).to(tl.float32)
        out = gate * tl.sigmoid(gate) * up
        tl.store(out_ptr + row * stride_out + offs, out, mask=mask)


def _triton_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    n_rows, two_d = x.shape
    d = two_d // 2
    out = torch.empty(n_rows, d, device=x.device, dtype=x.dtype)
    block = 1024
    grid = (n_rows, triton.cdiv(d, block))
    _silu_and_mul_kernel[grid](x, out, x.stride(0), out.stride(0), d, block)
    return out


class SiluAndMul(nn.Module):

    @torch.compile
    def _compiled(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            triton is not None
            and x.is_cuda
            and x.stride(-1) == 1
            and os.environ.get("INFERFLOW_TRITON_NORM_ACT", "1") != "0"
        ):
            return _triton_silu_and_mul(x)
        return self._compiled(x)
