import torch
import torch.nn.functional as F
from torch import nn

from inferflow.utils.context import get_context

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None


# KV cache 布局（每层）：k_cache / v_cache 形状 [num_blocks, block_size, num_kv_heads, head_dim]
# 一个 slot = block_id * block_size + 块内偏移，对应 k_cache.view(-1, num_kv_heads, head_dim)[slot]

if triton is not None:

    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        """
        TODO(Step9, 可选优化): 每个 program 处理一个 token：
          读 slot = slot_mapping[idx]，slot == -1 直接返回；
          读取 key/value 的第 idx 行（D = num_kv_heads * head_dim 个连续元素，行步长为 key_stride）；
          写入 k_cache/v_cache 的 slot * D 处。
        """
        pass


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    把本步新算出的 K/V（形状 [N, num_kv_heads, head_dim]）按 slot_mapping 写入分页 KV cache。
    slot == -1 表示跳过（CUDA Graph 的 padding 位置）。
    TODO(Step5): 先用纯 torch 实现：过滤 slot >= 0 的行，k_cache.view(-1, H, D)[slots] = key[valid]
    TODO(Step9): GPU 上换成 Triton kernel（纯 torch 版本的布尔索引会触发同步，无法被 CUDA Graph 录制）
    """
    raise NotImplementedError


# ---------------- 以下为已提供的参考实现（CPU 调试 / 正确性测试用） ----------------


def _gather_paged(cache: torch.Tensor, block_table: torch.Tensor, seqlen: int) -> torch.Tensor:
    num_blocks = (seqlen + cache.size(1) - 1) // cache.size(1)
    return cache[block_table[:num_blocks].long()].flatten(0, 1)[:seqlen]


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """q: [Lq, H, D]，k/v: [Lk, Hkv, D]；q 对齐到 k 的末尾做因果注意力。"""
    lq, lk = q.size(0), k.size(0)
    rep = q.size(1) // k.size(1)
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    q_pos = torch.arange(lk - lq, lk, device=q.device).unsqueeze(1)
    k_pos = torch.arange(lk, device=q.device).unsqueeze(0)
    mask = k_pos <= q_pos
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1).float(), k.transpose(0, 1).float(), v.transpose(0, 1).float(), attn_mask=mask, scale=scale
    )
    return o.transpose(0, 1).to(q.dtype)


def reference_attention(q, k, v, k_cache, v_cache, context, scale):
    """flash_attn 不可用时的逐序列参考实现，语义与下方 flash_attn 调用一致（速度很慢）。"""
    outputs = []
    if context.is_prefill:
        cu_q = context.cu_seqlens_q.tolist()
        cu_k = context.cu_seqlens_k.tolist()
        for i in range(len(cu_q) - 1):
            qi = q[cu_q[i] : cu_q[i + 1]]
            seqlen_k = cu_k[i + 1] - cu_k[i]
            if context.block_tables is not None:
                ki = _gather_paged(k_cache, context.block_tables[i], seqlen_k)
                vi = _gather_paged(v_cache, context.block_tables[i], seqlen_k)
            else:
                ki, vi = k[cu_k[i] : cu_k[i + 1]], v[cu_k[i] : cu_k[i + 1]]
            outputs.append(_sdpa(qi, ki, vi, scale))
        return torch.cat(outputs, 0)
    for i, seqlen_k in enumerate(context.context_lens.tolist()):
        ki = _gather_paged(k_cache, context.block_tables[i], seqlen_k)
        vi = _gather_paged(v_cache, context.block_tables[i], seqlen_k)
        outputs.append(_sdpa(q[i : i + 1], ki, vi, scale))
    return torch.cat(outputs, 0)


class Attention(nn.Module):
    """
    分页注意力。k_cache / v_cache 由 ModelRunner.allocate_kv_cache 在启动后注入；
    warmup 时它们还是空 tensor，此时不写 cache。
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        q: [N, num_heads, head_dim]，k/v: [N, num_kv_heads, head_dim]，N = 本步所有序列的 token 总数（decode 时 N = batch）
        TODO(Step4):
          1. cache 非空时，store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)（Step5 实现）
          2. flash_attn 不可用或在 CPU 上 → return reference_attention(...)
          3. prefill：flash_attn_varlen_func(q, k, v, cu_seqlens_q/k, max_seqlen_q/k, softmax_scale, causal=True,
                                           block_table=context.block_tables)
             Step6 前缀缓存命中时（block_tables 不为 None），K/V 要改用整块 k_cache / v_cache
          4. decode：flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context.context_lens,
                                            block_table=context.block_tables, softmax_scale, causal=True)
        """
        raise NotImplementedError
