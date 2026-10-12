"""
GPU 端到端测试：真实 Qwen3-0.6B，比较 InferFlow 与 HF transformers 的贪心输出。
运行：MODEL_PATH=~/huggingface/Qwen3-0.6B pytest tests/test_qwen3_gpu.py -s
bf16 下两者 kernel 不同，长序列后段可能因数值误差分叉，因此只比较前若干个 token。
"""

import os

import pytest
import torch

MODEL_PATH = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.path.isdir(MODEL_PATH), reason="needs CUDA and MODEL_PATH"
)

PROMPTS = ["介绍一下你自己", "list all prime numbers within 50", "请用一句话解释什么是 KV cache"]
NUM_TOKENS = 16
NUM_COMPARE = 8


@pytest.fixture(scope="module")
def hf_outputs():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()
    outs = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            y = model.generate(ids, max_new_tokens=NUM_TOKENS, do_sample=False)
        outs.append(y[0, ids.size(1) :].tolist())
    del model
    torch.cuda.empty_cache()
    return outs


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_matches_hf(hf_outputs, enforce_eager):
    from inferflow import LLM, SamplingParams

    llm = LLM(MODEL_PATH, enforce_eager=enforce_eager, gpu_memory_utilization=0.5)
    try:
        sp = SamplingParams(temperature=0, max_tokens=NUM_TOKENS)
        outs = llm.generate(PROMPTS, sp, use_tqdm=False)
        for ref, out in zip(hf_outputs, outs):
            assert out["token_ids"][:NUM_COMPARE] == ref[:NUM_COMPARE]
    finally:
        llm.exit()
