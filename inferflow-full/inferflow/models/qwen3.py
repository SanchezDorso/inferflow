import os

import torch
import torch.distributed as dist
from torch import nn

from inferflow.layers.activation import SiluAndMul
from inferflow.layers.attention import Attention
from inferflow.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from inferflow.layers.fused_qk_norm_rope import fused_qk_norm_rope
from inferflow.layers.layernorm import RMSNorm
from inferflow.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from inferflow.layers.rotary_embedding import get_rope
from inferflow.utils.context import get_context

try:
    import triton
except ImportError:
    triton = None


def get_rope_theta(config) -> float:
    rope_theta = getattr(config, "rope_theta", None)
    for attr in ("rope_parameters", "rope_scaling"):
        if rope_theta is None:
            rope_theta = (getattr(config, attr, None) or {}).get("rope_theta")
    return rope_theta if rope_theta is not None else 1000000.0


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        assert num_heads % tp_size == 0 and num_kv_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, num_heads, num_kv_heads, bias=qkv_bias)
        self.o_proj = RowParallelLinear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = get_rope(self.head_dim, self.head_dim, max_position, rope_theta)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache = self.attn.k_cache
        # prefill 且 KV 页已经分配时，旋转后的 K 直接写入页，不再经过单独的 store kernel。
        write_k_cache = (
            triton is not None
            and q.is_cuda
            and context.is_prefill
            and k_cache.numel() > 0
            and k_cache.dtype == q.dtype
            and context.slot_mapping is not None
            and context.slot_mapping.numel() == q.shape[0]
        )
        if triton is not None and q.is_cuda and os.environ.get("INFERFLOW_FUSED_QK", "1") != "0":
            # 前缀命中或混批时注意力从页里读 K，不必再把 K 写成一份连续张量。
            q, k_out = fused_qk_norm_rope(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_norm.eps,
                k_cache if write_k_cache else None,
                context.slot_mapping if write_k_cache else None,
                write_k_out=not (write_k_cache and context.block_tables is not None),
            )
            k = k_out
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)
            q, k = self.rotary_emb(positions, q, k)
            write_k_cache = False
        o = self.attn(q, k, v, k_in_cache=write_k_cache)
        return self.o_proj(o.flatten(1, -1))


class Qwen3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        assert hidden_act == "silu"
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=get_rope_theta(config),
        )
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        assert rope_scaling.get("rope_type", rope_scaling.get("type", "default")) == "default", "rope_scaling is not supported"
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
