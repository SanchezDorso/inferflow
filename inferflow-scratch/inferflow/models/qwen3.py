"""
Qwen3 模型定义。子模块命名必须与 HF checkpoint 中的参数名一致，loader 才能按名字加载：

  model.embed_tokens.weight
  model.layers.{i}.input_layernorm.weight
  model.layers.{i}.self_attn.{q,k,v,o}_proj.weight      q/k/v 在本实现里合并为 qkv_proj
  model.layers.{i}.self_attn.{q,k}_norm.weight          Qwen3 特有：对每个头做 RMSNorm
  model.layers.{i}.post_attention_layernorm.weight
  model.layers.{i}.mlp.{gate,up,down}_proj.weight       gate/up 合并为 gate_up_proj
  model.norm.weight
  lm_head.weight                                        tie_word_embeddings=True 时与 embed_tokens 共享
"""

import torch
import torch.distributed as dist
from torch import nn

from inferflow.layers.activation import SiluAndMul
from inferflow.layers.attention import Attention
from inferflow.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from inferflow.layers.layernorm import RMSNorm
from inferflow.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from inferflow.layers.rotary_embedding import get_rope


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
        # TODO(Step4): 需要的成员
        #   num_heads / num_kv_heads：本 rank 的头数（总头数 / dist.get_world_size()）
        #   head_dim（默认 hidden_size // num_heads）、q_size、kv_size、scaling = head_dim ** -0.5
        #   qkv_proj   = QKVParallelLinear(hidden_size, head_dim, 总 q 头数, 总 kv 头数, bias=qkv_bias)
        #   o_proj     = RowParallelLinear(总 q 头数 * head_dim, hidden_size)
        #   rotary_emb = get_rope(head_dim, head_dim, max_position, rope_theta)
        #   attn       = Attention(num_heads, head_dim, scaling, num_kv_heads)
        #   q_norm / k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        raise NotImplementedError

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # TODO(Step4): qkv_proj → split → view 成 [N, heads, head_dim] → q_norm/k_norm → RoPE → attn → flatten → o_proj
        raise NotImplementedError


class Qwen3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        assert hidden_act == "silu"
        # TODO(Step4): gate_up_proj（MergedColumnParallelLinear）、down_proj（RowParallelLinear）、act_fn（SiluAndMul）
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        # TODO(Step4): self_attn、mlp、input_layernorm、post_attention_layernorm
        #   提示：head_dim 用 getattr(config, "head_dim", None)，rope_theta 用 get_rope_theta(config)
        raise NotImplementedError

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        """
        TODO(Step4): pre-norm 结构，利用 RMSNorm(x, residual) 融合 residual add：
          第一层 residual 为 None：residual = hidden_states；hidden_states = input_layernorm(hidden_states)
          之后：hidden_states, residual = input_layernorm(hidden_states, residual)
          attn → post_attention_layernorm(hidden_states, residual) → mlp
          返回 (hidden_states, residual)
        """
        raise NotImplementedError


class Qwen3Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # TODO(Step4): embed → 逐层 (hidden, residual) → norm(hidden, residual)
        raise NotImplementedError


class Qwen3ForCausalLM(nn.Module):
    # HF checkpoint 名 -> (合并后的模块名, shard_id)，供 loader 使用
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
