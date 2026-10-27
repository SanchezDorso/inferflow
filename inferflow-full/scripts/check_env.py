"""检查 GPU 推理环境是否就绪：torch / CUDA / triton / flash-attn 3 / NCCL，并跑一次最小的分页注意力。"""

import importlib
import sys


def status(name, ok, info=""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name:<12} {info}")
    return ok


def main():
    ok = True
    print(f"python {sys.version.split()[0]}")
    try:
        import torch
    except ImportError:
        status("torch", False, "not installed")
        sys.exit(1)
    ok &= status("torch", True, torch.__version__)
    cuda = torch.cuda.is_available()
    ok &= status("cuda", cuda, f"{torch.version.cuda}, {torch.cuda.device_count()} GPU(s)" if cuda else "not available")
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"      GPU{i}: {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}")
        ok &= status("nccl", torch.distributed.is_nccl_available(), str(torch.cuda.nccl.version()))

    for mod in ("triton", "transformers", "flash_attn_3", "xxhash", "safetensors"):
        try:
            m = importlib.import_module(mod)
            ok &= status(mod, True, getattr(m, "__version__", ""))
        except Exception as e:
            ok &= status(mod, False, repr(e))

    if cuda:
        try:
            from flash_attn_interface import flash_attn_with_kvcache

            q = torch.randn(2, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
            kc = torch.randn(4, 16, 8, 128, device="cuda", dtype=torch.bfloat16)
            vc = torch.randn_like(kc)
            bt = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
            lens = torch.tensor([20, 10], device="cuda", dtype=torch.int32)
            flash_attn_with_kvcache(q, kc, vc, cache_seqlens=lens, page_table=bt, causal=True, num_splits=1)
            ok &= status("paged-attn", True, "flash_attn_3 flash_attn_with_kvcache + page_table works")
        except Exception as e:
            ok &= status("paged-attn", False, repr(e))

    print("environment ready" if ok else "environment NOT ready")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
