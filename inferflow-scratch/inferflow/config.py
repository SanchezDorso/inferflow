import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_prefix_caching: bool = True
    dist_port: int = 2333
    # "cuda" 为正式推理路径；"cpu" 仅用于无 GPU 环境下的调试与正确性测试
    device: str = "cuda"
    dtype: str | None = None
    hf_config: Any = None
    eos: tuple[int, ...] = ()

    def __post_init__(self):
        from transformers import AutoConfig

        assert os.path.isdir(self.model), f"model path not found: {self.model}"
        assert self.device in ("cuda", "cpu")
        # flash_attn 的 paged KV 接口要求 page size 是 256 的倍数
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 0.0 < self.gpu_memory_utilization <= 1.0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        assert self.hf_config.num_key_value_heads % self.tensor_parallel_size == 0
        assert self.hf_config.num_attention_heads % self.tensor_parallel_size == 0
        if self.device == "cpu":
            self.enforce_eager = True

    @property
    def torch_dtype(self):
        import torch

        if self.dtype is not None:
            return getattr(torch, self.dtype)
        if self.device == "cpu":
            return torch.float32
        dtype = getattr(self.hf_config, "dtype", None) or getattr(self.hf_config, "torch_dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return dtype or torch.bfloat16
