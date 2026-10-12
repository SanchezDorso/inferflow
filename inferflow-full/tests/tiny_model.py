"""构造一个随机初始化的迷你 Qwen3 模型目录（权重 + 分词器），用于无需下载的端到端测试。"""

import os
from functools import lru_cache

VOCAB_SIZE = 512
EOS_ID = VOCAB_SIZE - 1


def build_tiny_qwen3(path: str, seed: int = 0) -> str:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    if os.path.exists(os.path.join(path, "config.json")):
        return path
    os.makedirs(path, exist_ok=True)
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=2048,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        # 放大初始化方差，让 logits 分布更尖锐，避免贪心解码出现数值近似并列
        initializer_range=0.2,
        eos_token_id=EOS_ID,
        torch_dtype="float32",
    )
    model = Qwen3ForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)

    vocab = {f"t{i}": i for i in range(VOCAB_SIZE - 1)}
    vocab["<eos>"] = EOS_ID
    tok = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", unk_token="t0").save_pretrained(path)
    return path


@lru_cache(None)
def hf_model(path: str):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()


def hf_greedy(path: str, prompt: list[int], max_new_tokens: int) -> list[int]:
    """不做任何 EOS 处理的朴素贪心（generate 的 min_new_tokens 会压制 EOS，与 ignore_eos 语义不同）。"""
    import torch

    model = hf_model(path)
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(torch.tensor([ids])).logits[0, -1]
            ids.append(int(logits.argmax()))
    return ids[len(prompt) :]
