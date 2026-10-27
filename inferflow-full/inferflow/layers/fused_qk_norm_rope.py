"""把 Q/K 的按 head RMSNorm 和 RoPE 合成一次读写。

Qwen3 先对每个 head 做 RMSNorm，再做 rotate-half RoPE。拆开算时 Q 写回两次、K 写回两次，
prefill 还要把 K 再写进分页 cache。这个 kernel 在寄存器里做完归一化和旋转，Q 只写出一次；
prefill 时 K 按 slot_mapping 直接写入对应页，slot == -1 的位置跳过。
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _norm_then_rope(x1, x2, w1, w2, inv, cos, sin, ROUND_BF16: tl.constexpr):
        # 与拆开的 RMSNorm + RoPE 对齐：bf16 上先把归一化结果乘 weight，再在 fp32 里旋转。
        x1 = x1 * inv
        x2 = x2 * inv
        if ROUND_BF16:
            x1 = (x1.to(tl.bfloat16) * w1.to(tl.bfloat16)).to(tl.float32)
            x2 = (x2.to(tl.bfloat16) * w2.to(tl.bfloat16)).to(tl.float32)
        else:
            x1 = x1 * w1
            x2 = x2 * w2
        return x1 * cos - x2 * sin, x2 * cos + x1 * sin

    @triton.jit
    def _qk_norm_rope_kernel(
        q_ptr,
        k_ptr,
        q_out_ptr,
        k_out_ptr,
        q_w_ptr,
        k_w_ptr,
        cos_sin_ptr,
        pos_ptr,
        slot_ptr,
        k_cache_ptr,
        eps,
        q_stride_n,
        k_stride_n,
        q_out_stride_n,
        k_out_stride_n,
        cos_stride,
        cache_stride_slot,
        n_k_heads: tl.constexpr,
        head_dim: tl.constexpr,
        half: tl.constexpr,
        BLOCK: tl.constexpr,
        WRITE_K_OUT: tl.constexpr,
        WRITE_K_CACHE: tl.constexpr,
        ROUND_BF16: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offs = tl.arange(0, BLOCK)
        mask = offs < half
        pos = tl.load(pos_ptr + token)
        cos = tl.load(cos_sin_ptr + pos * cos_stride + offs, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(cos_sin_ptr + pos * cos_stride + half + offs, mask=mask, other=0.0).to(tl.float32)

        q1 = tl.load(q_ptr + token * q_stride_n + head * head_dim + offs, mask=mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + token * q_stride_n + head * head_dim + half + offs, mask=mask, other=0.0).to(tl.float32)
        qw1 = tl.load(q_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        qw2 = tl.load(q_w_ptr + half + offs, mask=mask, other=0.0).to(tl.float32)
        q_inv = tl.rsqrt((tl.sum(q1 * q1, axis=0) + tl.sum(q2 * q2, axis=0)) / head_dim + eps)
        q1, q2 = _norm_then_rope(q1, q2, qw1, qw2, q_inv, cos, sin, ROUND_BF16)
        tl.store(q_out_ptr + token * q_out_stride_n + head * head_dim + offs, q1, mask=mask)
        tl.store(q_out_ptr + token * q_out_stride_n + head * head_dim + half + offs, q2, mask=mask)

        if head < n_k_heads:
            k1 = tl.load(k_ptr + token * k_stride_n + head * head_dim + offs, mask=mask, other=0.0).to(tl.float32)
            k2 = tl.load(k_ptr + token * k_stride_n + head * head_dim + half + offs, mask=mask, other=0.0).to(tl.float32)
            kw1 = tl.load(k_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            kw2 = tl.load(k_w_ptr + half + offs, mask=mask, other=0.0).to(tl.float32)
            k_inv = tl.rsqrt((tl.sum(k1 * k1, axis=0) + tl.sum(k2 * k2, axis=0)) / head_dim + eps)
            k1, k2 = _norm_then_rope(k1, k2, kw1, kw2, k_inv, cos, sin, ROUND_BF16)
            if WRITE_K_OUT:
                tl.store(k_out_ptr + token * k_out_stride_n + head * head_dim + offs, k1, mask=mask)
                tl.store(k_out_ptr + token * k_out_stride_n + head * head_dim + half + offs, k2, mask=mask)
            if WRITE_K_CACHE:
                slot = tl.load(slot_ptr + token)
                if slot >= 0:
                    base = slot.to(tl.int64) * cache_stride_slot + head * head_dim
                    tl.store(k_cache_ptr + base + offs, k1, mask=mask)
                    tl.store(k_cache_ptr + base + half + offs, k2, mask=mask)


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    k_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    write_k_out: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """q: [N, Hq, D]，k: [N, Hk, D]。返回旋转后的 Q，以及可选的 K。

    k_cache 非空时，把旋转后的 K 按 slot_mapping 写入分页 cache。
    """
    if triton is None or not q.is_cuda:
        raise RuntimeError("fused_qk_norm_rope 需要 CUDA 上的 Triton")
    n_tokens, n_q_heads, head_dim = q.shape
    n_k_heads = k.shape[1]
    assert k.shape[0] == n_tokens and k.shape[2] == head_dim
    assert n_k_heads <= n_q_heads
    assert head_dim % 2 == 0 and q_weight.numel() == head_dim and k_weight.numel() == head_dim
    assert q.stride(-1) == 1 and k.stride(-1) == 1

    q_out = torch.empty(n_tokens, n_q_heads, head_dim, device=q.device, dtype=q.dtype)
    write_cache = k_cache is not None and k_cache.numel() > 0 and slot_mapping is not None
    k_out = torch.empty(n_tokens, n_k_heads, head_dim, device=k.device, dtype=k.dtype) if write_k_out else None
    if cos_sin_cache.dim() == 3:
        cos_sin = cos_sin_cache.reshape(cos_sin_cache.shape[0], head_dim)
    else:
        cos_sin = cos_sin_cache
    if positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    block = triton.next_power_of_2(head_dim)
    cache_stride = k_cache.stride(1) if write_cache else 0
    _qk_norm_rope_kernel[(n_tokens, n_q_heads)](
        q, k,
        q_out, k_out if k_out is not None else q_out,
        q_weight, k_weight,
        cos_sin, positions,
        slot_mapping if write_cache else positions,
        k_cache if write_cache else q_out,
        eps,
        q.stride(0), k.stride(0),
        q_out.stride(0), k_out.stride(0) if k_out is not None else 0,
        cos_sin.stride(0),
        cache_stride,
        n_k_heads,
        head_dim,
        head_dim // 2,
        block,
        write_k_out,
        write_cache,
        q.dtype == torch.bfloat16,
    )
    return q_out, k_out
