"""单进程 CPU：不经过引擎，直接比较 Qwen3ForCausalLM 一次 prefill 前向的 logits 与 HF。"""

import socket

import torch
import torch.distributed as dist

from tests.tiny_model import build_tiny_qwen3, hf_model


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_prefill_logits_match_hf(tmp_path):
    from transformers import AutoConfig

    from inferflow.models.qwen3 import Qwen3ForCausalLM
    from inferflow.utils.context import reset_context, set_context
    from inferflow.utils.loader import load_model

    torch._dynamo.config.disable = True
    path = build_tiny_qwen3(str(tmp_path / "tiny"))
    dist.init_process_group("gloo", f"tcp://127.0.0.1:{_free_port()}", world_size=1, rank=0)
    try:
        model = Qwen3ForCausalLM(AutoConfig.from_pretrained(path))
        load_model(model, path)
        seqs = [list(range(7)), list(range(100, 150)), [3, 1, 4, 1, 5]]
        input_ids = torch.tensor([t for s in seqs for t in s])
        positions = torch.tensor([i for s in seqs for i in range(len(s))])
        cu = torch.tensor([0] + [sum(len(s) for s in seqs[: i + 1]) for i in range(len(seqs))], dtype=torch.int32)
        # 无 KV cache 的 varlen prefill：多条序列拼接成一维，用 cu_seqlens 区分边界
        set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=50, max_seqlen_k=50)
        with torch.inference_mode():
            logits = model.compute_logits(model(input_ids, positions))
        reset_context()

        ref = hf_model(path)
        for i, s in enumerate(seqs):
            with torch.no_grad():
                ref_logits = ref(torch.tensor([s])).logits[0, -1]
            torch.testing.assert_close(logits[i], ref_logits, rtol=1e-4, atol=1e-4)
    finally:
        dist.destroy_process_group()
