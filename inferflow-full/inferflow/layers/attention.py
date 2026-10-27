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
    # FlashAttention 3：分页 KV 走 page_table，变长 prefill 不再接受 block_table。
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None


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
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
        value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """把本步新算出的 K/V 按 slot_mapping 写入分页 KV cache，slot == -1 表示跳过（CUDA Graph 的 padding）。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert slot_mapping.numel() == N
    if triton is not None and key.is_cuda and (D & (D - 1)) == 0:
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
        return
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    k_cache.view(-1, num_heads, head_dim)[slots] = key[valid]
    v_cache.view(-1, num_heads, head_dim)[slots] = value[valid]


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
    """flash_attn 不可用时的逐序列参考实现（仅用于 CPU 调试 / 正确性测试，速度很慢）。"""
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
    分页注意力（FlashAttention 3）：
    - prefill：flash_attn_varlen_func 处理变长拼接的一批序列；命中前缀缓存时
      改用 flash_attn_with_kvcache，以 page_table 读取已写入的分页 KV；
    - decode：flash_attn_with_kvcache 每条序列 1 个 query，按 page_table 读取历史 KV，
      并把本步新的 K/V 追加进 cache。
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        use_flash = flash_attn_varlen_func is not None and q.is_cuda
        # decode 把新的 K/V 交给 flash_attn_with_kvcache 写入，省掉每层一次额外的 store kernel。
        fuse_decode_store = use_flash and not context.is_prefill and k_cache.numel()
        if k_cache.numel() and v_cache.numel() and not fuse_decode_store:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if not use_flash:
            return reference_attention(q, k, v, k_cache, v_cache, context, self.scale)
        if context.is_prefill:
            if context.block_tables is not None:
                # KV 已由 store_kvcache 写入。FA3 的 varlen 接口没有 page table，
                # 前缀命中时用 with_kvcache 的变长 query 从分页 cache 读取完整 K/V。
                seqused_k = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
                return flash_attn_with_kvcache(
                    q,
                    k_cache,
                    v_cache,
                    cache_seqlens=seqused_k,
                    page_table=context.block_tables,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_q=context.max_seqlen_q,
                    softmax_scale=self.scale,
                    causal=True,
                )
            return flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        # cache_seqlens 是写入前的长度，kernel 从这里追加本步的 K/V。
        # num_splits=1：避免按当前长度在 host 上选切分，CUDA Graph 重放时长度会变。
        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            k=k.unsqueeze(1),
            v=v.unsqueeze(1),
            cache_seqlens=context.context_lens - 1,
            page_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
            num_splits=1,
        )
