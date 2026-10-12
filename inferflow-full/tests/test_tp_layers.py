"""在 CPU 上用 gloo 起 2 个进程，验证张量并行层与单卡等价。"""

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_checks(rank, world_size):
    from inferflow.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
    from inferflow.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from inferflow.utils.context import reset_context

    torch.manual_seed(0)
    x = torch.randn(5, 32)
    errors = {}

    # Column -> Row：MLP 的标准 TP 组合，最终输出应等于完整矩阵乘
    w1, w2 = torch.randn(64, 32), torch.randn(32, 64)
    col = ColumnParallelLinear(32, 64)
    row = RowParallelLinear(64, 32)
    col.weight.weight_loader(col.weight, w1)
    row.weight.weight_loader(row.weight, w2)
    errors["col_row"] = (row(col(x)) - F.linear(F.linear(x, w1), w2)).abs().max().item()

    # Merged gate/up
    wg, wu = torch.randn(48, 32), torch.randn(48, 32)
    merged = MergedColumnParallelLinear(32, [48, 48])
    merged.weight.weight_loader(merged.weight, wg, 0)
    merged.weight.weight_loader(merged.weight, wu, 1)
    g_ref, u_ref = F.linear(x, wg).chunk(world_size, -1)[rank], F.linear(x, wu).chunk(world_size, -1)[rank]
    g, u = merged(x).chunk(2, -1)
    errors["merged"] = max((g - g_ref).abs().max().item(), (u - u_ref).abs().max().item())

    # QKV：4 个 q 头、2 个 kv 头，head_dim 8
    head_dim, nq, nkv = 8, 4, 2
    wq, wk, wv = torch.randn(nq * head_dim, 32), torch.randn(nkv * head_dim, 32), torch.randn(nkv * head_dim, 32)
    bq, bk, bv = torch.randn(nq * head_dim), torch.randn(nkv * head_dim), torch.randn(nkv * head_dim)
    qkv = QKVParallelLinear(32, head_dim, nq, nkv, bias=True)
    for sid, w, b in (("q", wq, bq), ("k", wk, bk), ("v", wv, bv)):
        qkv.weight.weight_loader(qkv.weight, w, sid)
        qkv.bias.weight_loader(qkv.bias, b, sid)
    out = qkv(x)
    q_sz, kv_sz = nq // world_size * head_dim, nkv // world_size * head_dim
    q, k, v = out.split([q_sz, kv_sz, kv_sz], -1)
    errors["qkv"] = max(
        (q - F.linear(x, wq, bq).chunk(world_size, -1)[rank]).abs().max().item(),
        (k - F.linear(x, wk, bk).chunk(world_size, -1)[rank]).abs().max().item(),
        (v - F.linear(x, wv, bv).chunk(world_size, -1)[rank]).abs().max().item(),
    )

    # Embedding + LMHead
    vocab = 16
    emb_w = torch.randn(vocab, 32)
    ids = torch.tensor([0, 3, 8, 15, 9])
    emb = VocabParallelEmbedding(vocab, 32)
    emb.weight.weight_loader(emb.weight, emb_w)
    errors["embed"] = (emb(ids) - F.embedding(ids, emb_w)).abs().max().item()

    reset_context()
    head = ParallelLMHead(vocab, 32)
    head.weight.weight_loader(head.weight, emb_w)
    logits = head(x)
    if rank == 0:
        errors["lm_head"] = (logits - F.linear(x, emb_w)).abs().max().item()
    else:
        assert logits is None

    return errors


def _worker(rank, world_size, port, result_queue):
    try:
        dist.init_process_group("gloo", f"tcp://127.0.0.1:{port}", world_size=world_size, rank=rank)
        result_queue.put((rank, _run_checks(rank, world_size), None))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        result_queue.put((rank, None, traceback.format_exc()))


@pytest.mark.parametrize("world_size", [2])
def test_tensor_parallel_layers(world_size):
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, world_size, port, q)) for r in range(world_size)]
    for p in procs:
        p.start()
    try:
        results = {}
        while len(results) < world_size:
            rank, errors, tb = q.get(timeout=120)
            # 任一 rank 出错时其余 rank 会卡在集合通信里，直接失败并在 finally 中终止
            assert tb is None, f"rank {rank} failed:\n{tb}"
            results[rank] = errors
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
    for name, err in results[0].items():
        assert err < 1e-4, f"{name}: max abs err {err}"
