"""
端到端正确性测试（CPU，无需 GPU / 下载模型）：
随机初始化迷你 Qwen3，比较 InferFlow 与 HuggingFace transformers 的贪心解码结果，
覆盖 分页 KV cache / 连续批处理 / 前缀缓存 / 抢占回滚 / 张量并行 几条路径。
"""

import random
import socket

import pytest

from tests.tiny_model import build_tiny_qwen3, hf_greedy

from inferflow import LLM, SamplingParams


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return build_tiny_qwen3(str(tmp_path_factory.mktemp("tiny_qwen3")))


def _make_llm(model_dir, **kwargs):
    kwargs.setdefault("max_model_len", 2048)
    return LLM(model_dir, device="cpu", dist_port=_free_port(), **kwargs)


def _rand_prompt(rng, n):
    return [rng.randrange(0, 500) for _ in range(n)]


def _check(llm, model_dir, prompts, max_tokens):
    sp = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    for prompt, out in zip(prompts, outs):
        assert out["token_ids"] == hf_greedy(model_dir, prompt, max_tokens)


def test_batch_greedy_matches_hf(model_dir):
    rng = random.Random(0)
    prompts = [_rand_prompt(rng, n) for n in (5, 97, 256, 300, 611)]
    llm = _make_llm(model_dir)
    try:
        _check(llm, model_dir, prompts, max_tokens=12)
    finally:
        llm.exit()


def test_prefix_cache_reuse(model_dir):
    rng = random.Random(1)
    shared = _rand_prompt(rng, 600)
    prompts = [shared + _rand_prompt(rng, 30), shared + _rand_prompt(rng, 45)]
    llm = _make_llm(model_dir)
    try:
        _check(llm, model_dir, prompts[:1], max_tokens=8)
        _check(llm, model_dir, prompts[1:], max_tokens=8)
        # 600 token 共享前缀，块大小 16：37 个满块命中，最后不足一块重算
        assert llm.stats()["prefix_hit_tokens"] >= 37 * 16
    finally:
        llm.exit()


def test_preemption_rollback(model_dir):
    rng = random.Random(2)
    prompts = [_rand_prompt(rng, 15) for _ in range(3)]
    # 4 个 16-token 块：3 条请求各占 1 块，跨过 16 边界时只剩 1 个空闲块，必然抢占
    llm = _make_llm(model_dir, max_model_len=64, num_kvcache_blocks=4)
    try:
        _check(llm, model_dir, prompts, max_tokens=20)
        assert llm.stats()["num_preemptions"] > 0
    finally:
        llm.exit()


def test_tensor_parallel_cpu(model_dir):
    rng = random.Random(3)
    prompts = [_rand_prompt(rng, n) for n in (17, 300)]
    llm = _make_llm(model_dir, tensor_parallel_size=2)
    try:
        _check(llm, model_dir, prompts, max_tokens=10)
    finally:
        llm.exit()
