"""Decode 注意力走 FlashInfer，并且可以把 run() 录进 CUDA Graph。

plan() 会把页表拷回 CPU，不能出现在图里。ModelRunner 在重放前 plan 一次，
图里 28 层只调用 run()。use_cuda_graph=True 时 plan 写入固定 buffer，
重放读到的是这一步的新长度，而不是捕获时的长度。
"""

import os

import torch

_workspace = None
_wrappers: dict[int, object] = {}
_outs: dict[int, torch.Tensor] = {}
_import_failed = False


def flashinfer_decode_enabled() -> bool:
    global _import_failed
    if os.environ.get("INFERFLOW_FLASHINFER", "1") == "0":
        return False
    if _import_failed:
        return False
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        _import_failed = True
        return False
    return True


def _get_wrapper(batch_size: int, max_pages_per_seq: int, device: torch.device):
    global _workspace
    wrapper = _wrappers.get(batch_size)
    if wrapper is not None:
        return wrapper
    import flashinfer

    if _workspace is None:
        _workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    indices = torch.zeros(batch_size * max_pages_per_seq, dtype=torch.int32, device=device)
    last = torch.ones(batch_size, dtype=torch.int32, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        _workspace,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=False,
        paged_kv_indptr_buffer=indptr,
        paged_kv_indices_buffer=indices,
        paged_kv_last_page_len_buffer=last,
    )
    _wrappers[batch_size] = wrapper
    return wrapper


def plan_decode(
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    scale: float,
    max_pages_per_seq: int,
) -> None:
    """按当前页表 plan 一次。batch 维必须和即将 run 的 query 一致（含 CUDA Graph 的 padding）。"""
    lens = context_lens.to(torch.int32).clamp(min=1)
    batch = lens.shape[0]
    pages = (lens + page_size - 1) // page_size
    last = torch.where(lens % page_size == 0, torch.full_like(lens, page_size), lens % page_size)
    width = block_tables.shape[1]
    cols = torch.arange(width, device=block_tables.device)
    indices = block_tables.masked_select(cols.unsqueeze(0) < pages.unsqueeze(1)).to(torch.int32).clamp_min(0)
    indptr = torch.empty(batch + 1, dtype=torch.int32, device=lens.device)
    indptr[0] = 0
    indptr[1:] = pages.cumsum(0, dtype=torch.int32)
    _get_wrapper(batch, max_pages_per_seq, block_tables.device).plan(
        indptr,
        indices,
        last,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        sm_scale=scale,
        non_blocking=False,
    )


def flashinfer_run(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    """q: [B, Hq, D]。调用前必须对本 batch 做过 plan_decode。"""
    batch = q.shape[0]
    out = _outs.get(batch)
    if out is None or out.shape != q.shape or out.device != q.device:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        _outs[batch] = out
    return _wrappers[batch].run(q, (k_cache, v_cache), out=out)


_prefill_wrapper = None


def flashinfer_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """前缀命中或 decode/prefill 混批。KV 在 fp8 页里，FA3 要求 q/k 同 dtype，改走 FlashInfer。

    causal 与 flash_attn_with_kvcache 一致：query 对齐到 KV 的末尾。
    """
    global _prefill_wrapper, _workspace
    import flashinfer

    page_size = k_cache.shape[1]
    lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.int32)
    pages = (lens + page_size - 1) // page_size
    last = torch.where(lens % page_size == 0, torch.full_like(lens, page_size), lens % page_size)
    width = block_tables.shape[1]
    cols = torch.arange(width, device=block_tables.device)
    indices = block_tables.masked_select(cols.unsqueeze(0) < pages.unsqueeze(1)).to(torch.int32).clamp_min(0)
    indptr = torch.empty(lens.shape[0] + 1, dtype=torch.int32, device=lens.device)
    indptr[0] = 0
    indptr[1:] = pages.cumsum(0, dtype=torch.int32)
    if _workspace is None:
        _workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    if _prefill_wrapper is None:
        _prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_workspace, "NHD")
    _prefill_wrapper.plan(
        cu_seqlens_q.to(torch.int32),
        indptr,
        indices,
        last,
        q.shape[1],
        k_cache.shape[2],
        q.shape[2],
        page_size,
        causal=True,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        sm_scale=scale,
        non_blocking=False,
    )
    return _prefill_wrapper.run(q, (k_cache, v_cache))
